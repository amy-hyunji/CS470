import argparse
import os.path
from datetime import datetime

import numpy as np ## Error: mkl-service + Intel(R) MKL: MKL_THREADING_LAYER=INTEL 

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import yaml
from torch.autograd import Variable
from tqdm import tqdm

import models
import utils
###################################################################################################
###################################################################################################
#from datasets import vqa_dataset
#####################################################################################################

import json
import os
import os.path

import h5py
import torch
import torch.utils.data as data
################################################################
#from datasets.features import FeaturesDataset
################################################################
class FeaturesDataset(data.Dataset):

    def __init__(self, features_path, mode):
        self.path_hdf5 = features_path

        assert os.path.isfile(self.path_hdf5), \
            'File not found in {}, you must extract the features first with images_preprocessing.py'.format(
                self.path_hdf5)

        self.hdf5_file = h5py.File(self.path_hdf5, 'r')
        self.dataset_features = self.hdf5_file[mode]  # noatt or att (attention)

    def __getitem__(self, index):
        return torch.from_numpy(self.dataset_features[index].astype('float32'))

    def __len__(self):
        return self.dataset_features.shape[0]
################################################################

################################################################
#from preprocessing.preprocessing_utils import prepare_questions, prepare_answers, encode_question, encode_answers
################################################################
import re


def prepare_questions(annotations):
    """ Filter, Normalize and Tokenize question. """

    prepared = []
    questions = [q['question'] for q in annotations]

    for question in questions:
        # lower case
        question = question.lower()

        # define desired replacements here
        punctuation_dict = {'.': ' ', "'": '', '?': ' ', '_': ' ', '-': ' ', '/': ' ', ',': ' '}
        conversational_dict = {"thank you": '', "thanks": '', "thank": '', "please": '', "hello": '',
                               "hi ": ' ', "hey ": ' ', "good morning": '', "good afternoon": '', "have a nice day": '',
                               "okay": '', "goodbye": ''}

        rep = punctuation_dict
        rep.update(conversational_dict)

        # use these three lines to do the replacement
        rep = dict((re.escape(k), v) for k, v in rep.items())
        pattern = re.compile("|".join(rep.keys()))
        question = pattern.sub(lambda m: rep[re.escape(m.group(0))], question)

        # sentence to list
        question = question.split(' ')

        # remove empty strings
        question = list(filter(None, question))

        prepared.append(question)

    return prepared


def prepare_answers(annotations):
    answers = [[a['answer'] for a in ans_dict['answers']] for ans_dict in annotations]
    prepared = []

    for sample_answers in answers:
        prepared_sample_answers = []
        for answer in sample_answers:
            # lower case
            answer = answer.lower()

            # define desired replacements here
            punctuation_dict = {'.': ' ', "'": '', '?': ' ', '_': ' ', '-': ' ', '/': ' ', ',': ' '}

            rep = punctuation_dict
            rep = dict((re.escape(k), v) for k, v in rep.items())
            pattern = re.compile("|".join(rep.keys()))
            answer = pattern.sub(lambda m: rep[re.escape(m.group(0))], answer)
            prepared_sample_answers.append(answer)

        prepared.append(prepared_sample_answers)
    return prepared


def encode_question(question, token_to_index, max_length):
    question_vec = torch.zeros(max_length).long()
    length = min(len(question), max_length)
    for i in range(length):
        token = question[i]
        index = token_to_index.get(token, 0)
        question_vec[i] = index
    # empty encoded questions are a problem when packed,
    # if we set min length 1 we feed a 0 token to the RNN
    # that is not a problem since the token 0 does not represent a word
    return question_vec, max(length, 1)


def encode_answers(answers, answer_to_index):
    answer_vec = torch.zeros(len(answer_to_index))
    for answer in answers:
        index = answer_to_index.get(answer)
        if index is not None:
            answer_vec[index] += 1
    return answer_vec
################################################################

def get_loader(config, split):
    """ Returns the data loader of the specified dataset split """
    split = VQADataset(
        config,
        split
    )

    loader = torch.utils.data.DataLoader(
        split,
        batch_size=config['training']['batch_size'],
        shuffle=True if split == 'train' or split == 'trainval' else False,  # only shuffle the data in training
        pin_memory=True,
        num_workers=config['training']['data_workers'],
        collate_fn=collate_fn,
    )
    return loader


def collate_fn(batch):
    # Sort samples in the batch based on the question lengths in descending order.
    # This allows to pack the pack_padded_sequence when encoding questions using RNN
    batch.sort(key=lambda x: x['q_length'], reverse=True)
    return data.dataloader.default_collate(batch)


class VQADataset(data.Dataset):
    """ VQA dataset, open-ended """

    def __init__(self, config, split):
        super(VQADataset, self).__init__()

        with open(config['annotations']['path_vocabs'], 'r') as fd:
            vocabs = json.load(fd)

        annotations_dir = config['annotations']['dir']

        path_ann = os.path.join(annotations_dir, split + ".json")
        with open(path_ann, 'r') as fd:
            self.annotations = json.load(fd)

        self.max_question_length = config['annotations']['max_length']
        self.split = split

        # vocab
        self.vocabs = vocabs
        self.token_to_index = self.vocabs['question']
        self.answer_to_index = self.vocabs['answer']

        # pre-process questions and answers
        self.questions = prepare_questions(self.annotations)
        self.questions = [encode_question(q, self.token_to_index, self.max_question_length) for q in
                          self.questions]  # encode questions and return question and question lenght

        if self.split != 'test':
            self.answers = prepare_answers(self.annotations)
            self.answers = [encode_answers(a, self.answer_to_index) for a in
                            self.answers]  # create a sparse vector of len(self.answer_to_index) for each question containing the occurances of each answer

        if self.split == "train" or self.split == "trainval":
            self._filter_unanswerable_samples()

        # load image names in feature extraction order
        with h5py.File(config['images']['path_features'], 'r') as f:
            img_names = f['img_name'][()]
        self.name_to_id = {name: i for i, name in enumerate(img_names)}

        # names in the annotations, will be used to get items from the dataset
        self.img_names = [s['image'] for s in self.annotations]
        # load features
        self.features = FeaturesDataset(config['images']['path_features'], config['images']['mode'])

    def _filter_unanswerable_samples(self):
        """
        Filter during training the samples that do not have at least one answer
        """
        a = []
        q = []
        annotations = []
        for i in range(len(self.answers)):
            if len(self.answers[i].nonzero()) > 0:
                a.append(self.answers[i])
                q.append(self.questions[i])

                annotations.append(self.annotations[i])
        self.answers = a
        self.questions = q
        self.annotations = annotations

    @property
    def num_tokens(self):
        return len(self.token_to_index) + 1  # add 1 for <unknown> token at index 0

    def __getitem__(self, i):

        item = {}
        item['question'], item['q_length'] = self.questions[i]
        if self.split != 'test':
            item['answer'] = self.answers[i]
        img_name = self.img_names[i]
        feature_id = self.name_to_id[img_name]
        item['img_name'] = self.img_names[i]
        item['visual'] = self.features[feature_id]
        # collate_fn sorts the samples in order to be possible to pack them later in the model.
        # the sample_id is returned so that the original order can be restored during when evaluating the predictions
        item['sample_id'] = i

        return item

    def __len__(self):
        return len(self.questions)


#####################################################################################################

#####################################################################################################
#####################################################################################################
#####################################################################################################
def train(model, loader, optimizer, tracker, epoch, split):
    model.train()

    tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}
    tq = tqdm(loader, desc='{} E{:03d}'.format(split, epoch), ncols=0)
    loss_tracker = tracker.track('{}_loss'.format(split), tracker_class(**tracker_params))
    acc_tracker = tracker.track('{}_acc'.format(split), tracker_class(**tracker_params))
    log_softmax = nn.LogSoftmax(dim=1).cuda()

    for item in tq:
        v = item['visual']
        q = item['question']
        a = item['answer']
        q_length = item['q_length']

        v = Variable(v.cuda(async=True))
        q = Variable(q.cuda(async=True))
        a = Variable(a.cuda(async=True))
        q_length = Variable(q_length.cuda(async=True))

        out = model(v, q, q_length)

        # This is the Soft-loss described in https://arxiv.org/pdf/1708.00584.pdf

        nll = -log_softmax(out)

        loss = (nll * a / 10).sum(dim=1).mean()
        acc = utils.vqa_accuracy(out.data, a.data).cpu()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_tracker.append(loss.item())
        acc_tracker.append(acc.mean())
        fmt = '{:.4f}'.format
        tq.set_postfix(loss=fmt(loss_tracker.mean.value), acc=fmt(acc_tracker.mean.value))


def evaluate(model, loader, tracker, epoch, split):
    model.eval()
    tracker_class, tracker_params = tracker.MeanMonitor, {}

    predictions = []
    samples_ids = []
    accuracies = []

    tq = tqdm(loader, desc='{} E{:03d}'.format(split, epoch), ncols=0)
    loss_tracker = tracker.track('{}_loss'.format(split), tracker_class(**tracker_params))
    acc_tracker = tracker.track('{}_acc'.format(split), tracker_class(**tracker_params))
    log_softmax = nn.LogSoftmax(dim=1).cuda()

    with torch.no_grad():
        for item in tq:
            v = item['visual']
            q = item['question']
            a = item['answer']
            sample_id = item['sample_id']
            q_length = item['q_length']

            v = Variable(v.cuda(async=True))
            q = Variable(q.cuda(async=True))
            a = Variable(a.cuda(async=True))
            q_length = Variable(q_length.cuda(async=True))

            out = model(v, q, q_length)

            # This is the Soft-loss described in https://arxiv.org/pdf/1708.00584.pdf

            nll = -log_softmax(out)

            loss = (nll * a / 10).sum(dim=1).mean()
            acc = utils.vqa_accuracy(out.data, a.data).cpu()

            # save predictions of this batch
            _, answer = out.data.cpu().max(dim=1)

            predictions.append(answer.view(-1))
            accuracies.append(acc.view(-1))
            # Sample id is necessary to obtain the mapping sample-prediction
            samples_ids.append(sample_id.view(-1).clone())

            loss_tracker.append(loss.item())
            acc_tracker.append(acc.mean())
            fmt = '{:.4f}'.format
            tq.set_postfix(loss=fmt(loss_tracker.mean.value), acc=fmt(acc_tracker.mean.value))

        predictions = list(torch.cat(predictions, dim=0))
        accuracies = list(torch.cat(accuracies, dim=0))
        samples_ids = list(torch.cat(samples_ids, dim=0))

    eval_results = {
        'answers': predictions,
        'accuracies': accuracies,
        'samples_ids': samples_ids,
        'avg_accuracy': acc_tracker.mean.value,
        'avg_loss': loss_tracker.mean.value
    }

    return eval_results


def main():
    # Load config yaml file
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_config', default='config/default.yaml', type=str,
                        help='path to a yaml config file')
    args = parser.parse_args()

    if args.path_config is not None:
        with open(args.path_config, 'r') as handle:
            config = yaml.load(handle)

    # generate log directory
    dir_name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    path_log_dir = os.path.join(config['logs']['dir_logs'], dir_name)

    if not os.path.exists(path_log_dir):
        os.makedirs(path_log_dir)

    print('Model logs will be saved in {}'.format(path_log_dir))

    cudnn.benchmark = True

    # Generate datasets and loaders
    train_loader = get_loader(config, split='train')
    val_loader = get_loader(config, split='val')
    
    model = nn.DataParallel(models.Model(config, train_loader.dataset.num_tokens)).cuda()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 config['training']['lr'])

    # Load model weights if necessary
    if config['model']['pretrained_model'] is not None:
        print("Loading Model from %s" % config['model']['pretrained_model'])
        log = torch.load(config['model']['pretrained_model'])
        dict_weights = log['weights']
        model.load_state_dict(dict_weights)

    tracker = utils.Tracker()

    min_loss = 10
    max_accuracy = 0

    path_best_accuracy = os.path.join(path_log_dir, 'best_accuracy_log.pth')
    path_best_loss = os.path.join(path_log_dir, 'best_loss_log.pth')
    
    print("number of total epochs :",config['training']['epochs'])

    for i in range(config['training']['epochs']):

        train(model, train_loader, optimizer, tracker, epoch=i, split=config['training']['train_split'])
        # If we are training on the train split (and not on train+val) we can evaluate on val
        if config['training']['train_split'] == 'train':
            eval_results = evaluate(model, val_loader, tracker, epoch=i, split='val')

            # save all the information in the log file
            log_data = {
                'epoch': i,
                'tracker': tracker.to_dict(),
                'config': config,
                'weights': model.state_dict(),
                'eval_results': eval_results,
                'vocabs': train_loader.dataset.vocabs,
            }

            # save logs for min validation loss and max validation accuracy
            if eval_results['avg_loss'] < min_loss:
                torch.save(log_data, path_best_loss)  # save model
                min_loss = eval_results['avg_loss']  # update min loss value

            if eval_results['avg_accuracy'] > max_accuracy:
                torch.save(log_data, path_best_accuracy)  # save model
                max_accuracy = eval_results['avg_accuracy']  # update max accuracy value

    # Save final model
    log_data = {
        'tracker': tracker.to_dict(),
        'config': config,
        'weights': model.state_dict(),
        'vocabs': train_loader.dataset.vocabs,
    }

    path_final_log = os.path.join(path_log_dir, 'final_log.pth')
    torch.save(log_data, path_final_log)


if __name__ == '__main__':
    main()
