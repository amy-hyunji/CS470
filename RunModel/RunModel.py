import os
# import random
import numpy as np
import torch
from easydict import EasyDict as edict

from evaluator import Evaluator
from sam.sa_m4c import SAM4C, BertConfig
from sam.datasets import DatasetMapTrain
from sam.task_utils import (clip_gradients, forward_model,
                            get_optim_scheduler, load_datasets)
from tools.registry import registry

#torch2trt
import torch
from torch2trt import torch2trt

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

        # Add all configs to registry
        registry.update({pretrained_eval: textVQA_pretrained, config: textVQA_config})
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
        self.model.load_state_dict(torch.load(self.textVQA_pretrained)) #이 부분에서 문제 발생
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

        image = Image.open(img_path)


        ##### TEST(우리의 문제에 맞게 추후에 변경할 부분) #######
        dataloaders = load_datasets(self.task_cfg, ["test"]) #우선 dataset을 통해 batch를 만들었다. batch size = 1
        #pred_data = dataloaders[0] #
        #pred_data["data"][0]["question"] = question
        #####################################################

        beam_size = 5  # the best beam_size in the Paper

        # Set beam_size
        self.model.moule.set_beam_size(beam_size)



        with torch.no_grad():
            for batch in self.dataloaders["test"]:
                loss,score,_,predctions=forward_model(task_cfg, device, self.model, None, "pred", batch_dict, True)
    # 해당 대회를 분석한다. predictions가 unanswerable이 아닌 것 중에서, 가장 높은 score를 가진 것을 선택한다....???
        for pred in predictions:
            if pred["pred_answer"]!="unanswerable":
                return pred["pred_answer"]
        return pred["pred_answer"]
        #return answer


# 원문코드의 sam/task_utils.py에 있는 함수들
def get_loader(task_cfg, tokenizer, split):

    dataset_names = task_cfg[f"{split}_on"]
    assert isinstance(dataset_names, list)

    datasets = []
    for dset in dataset_names:
        _dataset = DatasetMapTrain[dset](
            split=split, tokenizer=tokenizer, task_cfg=task_cfg
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

def load_datasets(task_cfg, splits):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    loaders = {}
    for split in splits:
        loaders[split] = get_loader(task_cfg, tokenizer, split)
    return loaders

