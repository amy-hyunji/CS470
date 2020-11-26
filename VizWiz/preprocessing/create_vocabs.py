import argparse
import itertools
import json
import os
from collections import Counter
from itertools import takewhile
from pprint import pprint

import yaml

###############################################################
#from preprocessing.preprocessing_utils import prepare_questions, prepare_answers
###############################################################

import re

import torch


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


###############################################################

def create_question_vocab(questions, min_count=0):
    """
    Extract vocabulary used to tokenize and encode questions.
    """
    words = itertools.chain.from_iterable([q for q in questions])  # chain('ABC', 'DEF') --> A B C D E F
    counter = Counter(words)

    counted_words = counter.most_common()
    # select only the words appearing at least min_count
    selected_words = list(takewhile(lambda x: x[1] >= min_count, counted_words))

    vocab = {t[0]: i for i, t in enumerate(selected_words, start=1)}

    return vocab


def create_answer_vocab(annotations, top_k):
    answers = itertools.chain.from_iterable(prepare_answers(annotations))

    counter = Counter(answers)
    counted_ans = counter.most_common(top_k)
    # start from labels from 0
    vocab = {t[0]: i for i, t in enumerate(counted_ans, start=0)}

    return vocab


parser = argparse.ArgumentParser()
parser.add_argument('--path_config', default='config/default.yaml', type=str,
                    help='path to a yaml config file')


def main():
    # Load and visualize config from yaml file
    global args
    args = parser.parse_args()

    if args.path_config is not None:
        with open(args.path_config, 'r') as handle:
            config = yaml.load(handle)

    pprint(config)

    # Load annotations
    dir_path = config['annotations']['dir']

    # vocabs are created based on train (trainval) split only
    train_path = os.path.join(dir_path, config['training']['train_split'] + '.json')
    with open(train_path, 'r') as fd:
        train_ann = json.load(fd)

    questions = prepare_questions(train_ann)

    question_vocab = create_question_vocab(questions, config['annotations']['min_count_word'])
    answer_vocab = create_answer_vocab(train_ann, config['annotations']['top_ans'])

    # Save pre-processing vocabs
    vocabs = {
        'question': question_vocab,
        'answer': answer_vocab,
    }

    with open(config['annotations']['path_vocabs'], 'w') as fd:
        json.dump(vocabs, fd)

    print("vocabs saved in {}".format(config['annotations']['path_vocabs']))


if __name__ == '__main__':
    main()
