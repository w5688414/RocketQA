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

from functools import partial
import argparse
import sys
import os
import random
import time

import numpy as np
import paddle
import paddle.nn.functional as F
import paddlenlp as ppnlp
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad

from data import read_text, convert_inference_example, create_dataloader
from model import DualEncoder

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--text_file", type=str, required=True, help="The full path of input file")
parser.add_argument("--output_file", default="output", type=str, required=True, help="The full path of output file to save text embedddings")
parser.add_argument("--params_path", type=str, required=True, help="The path to model parameters to be loaded.")
parser.add_argument("--max_seq_length", default=64, type=int, help="The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--output_emb_size", default=None, type=int, help="output_embedding_size")
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--pad_to_max_seq_len", action="store_true", help="Whether to pad to max seq length.")
args = parser.parse_args()
# yapf: enable


def predict(model, data_loader):
    """
    Predicts the data labels.

    Args:
        model (obj:`DualEncoder`): A model to extract text embedding or calculate similarity of text pair.
        data_loaer (obj:`List(Example)`): The processed data ids of text pair: [query_input_ids, query_token_type_ids, title_input_ids, title_token_type_ids]
    Returns:
        results(obj:`List`): cosine similarity of text pairs.
    """

    embeddings = []

    model.eval()

    with paddle.no_grad():
        for batch_data in data_loader:
            text_input_ids, text_token_type_ids = batch_data

            text_input_ids = paddle.to_tensor(text_input_ids)
            text_token_type_ids = paddle.to_tensor(text_token_type_ids)

            batch_embedding = model.get_pooled_embedding(
                text_input_ids, text_token_type_ids).numpy()

            embeddings.append(batch_embedding)

        embeddings = np.concatenate(embeddings, axis=0)

        return embeddings


if __name__ == "__main__":
    paddle.set_device(args.device)

    tokenizer = ppnlp.transformers.ErnieTokenizer.from_pretrained(
        'ernie-2.0-en')

    trans_func = partial(
        convert_inference_example,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # query_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # query_segment
    ): [data for data in fn(samples)]

    valid_ds = load_dataset(read_text, data_path=args.text_file, lazy=False)

    valid_data_loader = create_dataloader(
        valid_ds,
        mode='predict',
        batch_size=args.batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    pretrained_model = ppnlp.transformers.ErnieModel.from_pretrained(
        'ernie-2.0-en')

    model = DualEncoder(pretrained_model, output_emb_size=args.output_emb_size)

    if args.params_path and os.path.isfile(args.params_path):
        state_dict = paddle.load(args.params_path)
        #print("paddle_load state_dict:{}".format(state_dict))
        #print("model state_dict before set_dict:{}".format(model.state_dict()))

        model.set_dict(state_dict)

        #print("model state_dict after set_dict:{}".format(model.state_dict()))
        print("Loaded parameters from %s" % args.params_path)
        #for name, param in model.named_parameters():
        #    print("{}:{}".format(name, param.shape))
        #    if name == "ernie.embeddings.word_embeddings.weight":
        #        print("{}:{}".format("101 embedding", param[101, :20]))
        #        print("embeding {}:{}".format(name, param))
        #    # print("{}:{}".format(name, param))
    else:
        raise ValueError(
            "Please set --params_path with correct pretrained model file")

    embeddings = predict(model, valid_data_loader)
    print("final embedding:{}".format(embeddings[0, :20]))
    """
    with open(args.output_file + '.npy', 'wb') as f:
        np.save(f, np.array(embeddings))
        print("succeed save {} array to output file".format(
            np.array(embeddings).shape))
   """
