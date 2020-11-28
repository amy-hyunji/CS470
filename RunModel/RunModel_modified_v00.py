'''
!python RunModel_modified.py
로 test하면서 실행해볼 것! (맨 아래에 test해놓게 해당 파일에서만 돌아가도록 만들어놓은 __main__ 있음)
'''

import os
# import random
import numpy as np
import torch
from easydict import EasyDict as edict

from evaluator import Evaluator
from sam.sa_m4c import SAM4C, BertConfig
from sam.datasets import DatasetMapTrain
from sam.task_utils import (clip_gradients, forward_model,
                            get_optim_scheduler)
from tools.registry import registry

from torch.utils.data import ConcatDataset, DataLoader, RandomSampler
from pytorch_transformers.tokenization_bert import BertTokenizer

import h5py
from PIL import Image
from googleapiclient.discovery import build
from io import BytesIO
import base64

#torch2trt
import torch
#from torch2trt import torch2trt

import getpass

APIKEY = getpass.getpass("Enter API Key:")


class RunModel:
    def __init__(self, textVQA_config="", textVQA_pretrained=""):
        self.textVQA_config = textVQA_config
        self.textVQA_pretrained = textVQA_pretrained

        assert self.textVQA_pretrained != "", "No Pretrained Model"
        assert self.textVQA_config != "", "No Configration"

        with open(textVQA_config, "r") as f:
            self.task_cfg = edict(yaml.safe_load(f))
        #         seed = self.task_cfg["seed"]
        #         random.seed(seed)
        #         np.random.seed(seed)
        #         torch.manual_seed(seed)

        self.task_cfg["batch_size"]=1
        
        # Add all configs to registry
        registry.update({"pretrained_eval": textVQA_pretrained, "config": textVQA_config})
        registry.update(self.task_cfg)

        # Setting device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()
        print("Number of GPUs : {self.n_gpu}")

        # Setting Model
        mmt_config = BertConfig.from_dict(self.task_cfg["SA-M4C"])
        text_bert_config = BertConfig.from_dict(self.task_cfg["TextBERT"])

        self.model = SAM4C(mmt_config, text_bert_config)
        self.model.to(self.device)
        
        pretrained=torch.load(self.textVQA_pretrained)
        self.model.load_state_dict(pretrained["model_state_dict"])
        self.model.eval()

        if n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

    def textVQA(self, question, img_path):
        """
        images : input image files
        Using self.textVQA_pretrained, self.model
        """
        # self.task.cfg, (self.textVQA_config, self.textVQA_pretrained).self.save_path, self.checkpoint_path,self.base_lr

        assert os.path.exists(self.textVQA_pretrained), "No path of Pretrained Model"

        ##### image->h5 file
        save_path = './numpy.hdf5'
        hf = h5py.File(save_path, 'a')
        img_np = np.array(Image.open(img_path))

        dset = hf.create_dataset('default', data=img_np)
        hf.close()
        self.task_cfg.textvqa_obj = save_path
        self.task_cfg.textvqa_ocr = save_path ### 이 부분이 맞는지 모르겠다. textvqa_dataset.py의 135번째 줄 참고
        ####################

        ##### get ocr tokens
        tokens = self._get_ocr_tokens(img_path)
        ###################

        ##### image 사이즈 구하기
        img = Image.open(img_path)
        image_width, image_height = img.size
        ###################

        ##### TEST(우리의 문제에 맞게 추후에 변경할 부분) #######
        dataloaders = load_datasets(self.task_cfg, ["test"], question, image_height, image_width, tokens) #우선 dataset을 통해 batch를 만들었다. batch size = 1
        #####################################################

        beam_size = 5  # the best beam_size in the Paper

        # Set beam_size
        self.model.moule.set_beam_size(beam_size)

        with torch.no_grad():
            for batch in dataloaders["test"]:
                loss,score,_,predictions = forward_model(self.task_cfg, self.device, self.model, None, "pred", batch, True)
                
        return predictions # deal with just one image at one time.

    def _get_image_bytes(self, image_path):
        image_path = image_path

        with BytesIO() as output:
            with Image.open(image_path) as img:
                img.save(output, 'JPEG')
            data = base64.b64encode(output.getvalue()).decode('utf-8')
        return data


    def _get_ocr_tokens(self, image_path):
        vision_service = build("vision", "v1", developerKey=APIKEY)
        request = vision_service.images().annotate(body={
            'requests': [{
                'image': {
                    'content': self._get_image_bytes(image_path)
                },
                'features': [{
                    'type': 'TEXT_DETECTION',
                    'maxResults': 15,
                }]
            }],
        })
        responses = request.execute(num_retries=5)
        tokens = []

        if 'textAnnotations' not in responses['responses'][0]:
            print("Either no OCR tokens detected by Google Cloud Vision or "
                  "the request to Google Cloud Vision failed. "
                  "Predicting without tokens.")
            print(responses)
            return []

        for token in responses['responses'][0]['textAnnotations'][1:]:
            tokens += token['description'].split('\n')
        return tokens


# 원문코드의 sam/task_utils.py에 있는 함수들
def get_loader(task_cfg, tokenizer, split, question, image_height, image_width, tokens):

    dataset_names = task_cfg[f"{split}_on"]
    assert isinstance(dataset_names, list)

    datasets = []
    for dset in dataset_names:
        _dataset = DatasetMapTrain[dset](
            split=split, tokenizer=tokenizer, task_cfg=task_cfg, question=question, image_height=image_height, image_width=image_width, tokens=tokens
        )     # sam/datasets/testvqa_dataset.py
        datasets.append(_dataset)

    if len(datasets) > 1:
        dataset_instance = ConcatDataset(datasets)
    else:
        dataset_instance = datasets[0]

    random_sampler = RandomSampler(dataset_instance)
    loader = DataLoader(
        dataset_instance,
        sampler=random_sampler if split == "train" else None,
        batch_size=1,#batch size를 1로 설정하였다.
        num_workers=task_cfg["num_workers"],
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )       # 여기서 dataloader 만든다. 우리 img랑 question 입력하는 방법?
    return loader

def load_datasets(task_cfg, splits, question, image_height, image_width, tokens):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    loaders = {}
    for split in splits:
        loaders[split] = get_loader(task_cfg, tokenizer, split, question, image_height, image_width, tokens)
    return loaders


if __mian__=="__name__":
    model = RunModel(textVQA_config="configs/train-stvqa-eval-stvqa-c3.yml", textVQA_pretrained="data/pretrained-models/best_model.tars")
    print(model.textVQA(question="What is this?",img_path="tools/sam-textvqa-large.png"))