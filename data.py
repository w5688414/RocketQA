# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import paddle

from paddlenlp.utils.log import logger


def create_dataloader(dataset,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None,
                      trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == 'train' else False
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)

    return paddle.io.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)

def _truncate_seq_pair(tokens_a, tokens_b, max_length):

    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def convert_train_example(example,
                          tokenizer,
                          query_max_seq_length=32,
                          title_max_seq_length=128):
    """
    Builds model inputs from a sequence.
        
    A BERT sequence has the following format:

    - single sequence: ``[CLS] X [SEP]``

    Args:
        example(obj:`list(str)`): The list of text to be converted to ids.
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer` 
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        max_seq_len(obj:`int`): The maximum total input sequence length after tokenization. 
            Sequences longer than this will be truncated, sequences shorter will be padded.
        is_test(obj:`False`, defaults to `False`): Whether the example contains label or not.

    Returns:
        input_ids(obj:`list[int]`): The list of query token ids.
        token_type_ids(obj: `list[int]`): List of query sequence pair mask.
    """
    tokens_title_pos = tokenizer._tokenize(example["pos_title"])
    tokens_para_pos = tokenizer._tokenize(example["pos_para"])
    _truncate_seq_pair(tokens_title_pos, tokens_para_pos, title_max_seq_length - 3)

    encoded_inputs = tokenizer(
        text=example["query"], max_seq_len=query_max_seq_length)
    query_input_ids = encoded_inputs["input_ids"]
    query_token_type_ids = encoded_inputs["token_type_ids"]

    # encoded_inputs = tokenizer(
    #     text=example["pos_title"],
    #     text_pair=example["pos_para"],
    #     max_seq_len=title_max_seq_length)
    encoded_inputs = tokenizer(
        text="",
        text_pair=tokens_title_pos+tokens_para_pos,
        max_seq_len=title_max_seq_length)
    # print(example["pos_title"])
    # print(example["pos_para"])
    # print(encoded_inputs)
    pos_title_input_ids = encoded_inputs["input_ids"]
    pos_title__token_type_ids = encoded_inputs["token_type_ids"]

    tokens_title_neg = tokenizer._tokenize(example["neg_title"])
    tokens_para_neg = tokenizer._tokenize(example["neg_para"])
    _truncate_seq_pair(tokens_title_neg, tokens_para_neg, title_max_seq_length - 3)
    encoded_inputs = tokenizer(
        text="",
        text_pair=tokens_title_neg+tokens_para_neg,
        max_seq_len=title_max_seq_length)

    # print(example["neg_title"])
    # print(example["neg_para"])
    # print(encoded_inputs)
    neg_title_input_ids = encoded_inputs["input_ids"]
    neg_title__token_type_ids = encoded_inputs["token_type_ids"]

    result = [
        query_input_ids, query_token_type_ids, pos_title_input_ids,
        pos_title__token_type_ids, neg_title_input_ids,
        neg_title__token_type_ids
    ]

    return result


def convert_inference_example(example, tokenizer, max_seq_length=128):
    """
    Builds model inputs from a sequence.
        
    A BERT sequence has the following format:

    - single sequence: ``[CLS] X [SEP]``

    Args:
        example(obj:`list(str)`): The list of text to be converted to ids.
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer` 
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        max_seq_len(obj:`int`): The maximum total input sequence length after tokenization. 
            Sequences longer than this will be truncated, sequences shorter will be padded.
        is_test(obj:`False`, defaults to `False`): Whether the example contains label or not.

    Returns:
        input_ids(obj:`list[int]`): The list of query token ids.
        token_type_ids(obj: `list[int]`): List of query sequence pair mask.
    """

    encoded_inputs = tokenizer(text=example["text"], max_seq_len=max_seq_length)
    text_input_ids = encoded_inputs["input_ids"]
    text_token_type_ids = encoded_inputs["token_type_ids"]

    result = [text_input_ids, text_token_type_ids]
    # print("text_input_ids:{}".format(text_input_ids))
    # print("text_token_type_ids:{}".format(text_token_type_ids))

    # f = open('tid', 'a')
    # for tid in range(len(text_input_ids)):
        # f.write(str(text_input_ids[tid]) + '\t' + example["text"][tid] + '\n')
            # f.write(str(token_ids_q[tid]) + ' ')
    # f.write('\t')


    return result


def convert_inference_example_para(example, tokenizer, max_seq_length=128):
    """
    Builds model inputs from a sequence.
        
    A BERT sequence has the following format:

    - single sequence: ``[CLS] X [SEP]``

    Args:
        example(obj:`list(str)`): The list of text to be converted to ids.
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer` 
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        max_seq_len(obj:`int`): The maximum total input sequence length after tokenization. 
            Sequences longer than this will be truncated, sequences shorter will be padded.
        is_test(obj:`False`, defaults to `False`): Whether the example contains label or not.

    Returns:
        input_ids(obj:`list[int]`): The list of query token ids.
        token_type_ids(obj: `list[int]`): List of query sequence pair mask.
    """

    encoded_inputs = tokenizer(text="",text_pair=example["text"], max_seq_len=max_seq_length)
    text_input_ids =encoded_inputs["input_ids"]
    text_token_type_ids =encoded_inputs["token_type_ids"]

    result = [text_input_ids, text_token_type_ids]
    # print(len(text_input_ids))
    # print("text_input_ids:{}".format(text_input_ids))
    # print("text_token_type_ids:{}".format(text_token_type_ids))

    # f = open('tid', 'a')
    # for tid in range(len(text_input_ids)):
        # f.write(str(text_input_ids[tid]) + '\t' + example["text"][tid] + '\n')
            # f.write(str(token_ids_q[tid]) + ' ')
    # f.write('\t')


    return result


def read_train_data(data_path):
    """Reads data."""
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.rstrip().split("\t")
            # print(len(data))
            if len(data) != 6:
                continue
            yield {
                'query': data[0],
                'pos_title': data[1],
                'pos_para': data[2],
                'neg_title': data[3],
                'neg_para': data[4]
            }


def read_text(data_path):
    """Reads data."""
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.rstrip().split("\t")
            if len(data) != 1:
                continue
            yield {'text': data[0]}

def read_dev_text(data_path):
    """Reads data."""
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.rstrip().split("\t")
            yield {'text': data[0]}

def read_passage_text(data_path):
    """Reads data."""
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.rstrip().split("\t")
            yield {'text': data[2]}

def gen_id2corpus(corpus_file):
    id2corpus = {}
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            id2corpus[idx] = line.rstrip()
    return id2corpus
