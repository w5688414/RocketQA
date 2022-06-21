# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import os
from functools import partial

import paddle
import paddle.nn.functional as F
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad
from tqdm import tqdm 
from paddlenlp.transformers import AutoTokenizer,AutoModel

from utils import convert_example, read_train_set, create_dataloader
from model import CrossEncoder

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--params_path", type=str, required=True, default="checkpoints/model_900/model_state.pdparams", help="The path to model parameters to be loaded.")
parser.add_argument("--max_seq_length", type=int, default=128, help="The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU/CPU for training.")
parser.add_argument("--test_set", type=str, required=True, help="The full path of test_set.")
parser.add_argument('--device', choices=['cpu', 'gpu', 'xpu', 'npu'], default="gpu", help="Select which device to train model, defaults to gpu.")
args = parser.parse_args()
# yapf: enable


def predict(model, data_loader):
    results = []
    model.eval()
    with paddle.no_grad():
        for batch in tqdm(data_loader):
            input_ids, token_type_ids,label = batch
            logits = model(input_ids, token_type_ids)
            loss, probs = F.softmax_with_cross_entropy(logits=logits, label=label,return_softmax=True)
            probs = probs.numpy()
            results.extend(probs[:, 1])
    return results


if __name__ == "__main__":
    paddle.set_device(args.device)

    test_ds = load_dataset(read_train_set, data_path=args.test_set, lazy=False)

    pretrained_model = AutoModel.from_pretrained(
        'ernie-1.0')
    tokenizer = AutoTokenizer.from_pretrained('ernie-1.0')

    model = CrossEncoder(pretrained_model,num_classes=2)

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        is_pair=True)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
        Stack(dtype="int64")  # label
    ): [data for data in fn(samples)]

    test_data_loader = create_dataloader(
        test_ds,
        mode='predict',
        batch_size=args.batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    if args.params_path and os.path.isfile(args.params_path):
        state_dict = paddle.load(args.params_path)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % args.params_path)

    results = predict(model, test_data_loader)
    file = open('result.txt','w')
    for score in results:
        print(score)
        file.write(str(score)+'\n')