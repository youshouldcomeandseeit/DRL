import copy
import re
import json
# import matplotlib.pyplot as plt
import numpy as np
import config
from transformers import RobertaTokenizerFast
from tool import tools
import torch

import string

string_mask = r""""#&*+-:;<=>?@[\]^_`{|}~"""
def load_data(file_path):
    with open(file_path, 'r') as f:
        # sentence = []
        # entities_features = []
        datas = []
        for dic in f:
            data = json.loads(dic)
            if len(data['sentence']) < 512:
                datas.append(data)
            # sentence.append(data['sentence'])
            # entities_features.append(data['entities'])

    # 输出的tokens 是一个个的列表
    return datas



class Dataset(torch.utils.data.Dataset):
    def __init__(self, path,is_training=True):
        dataset = load_data(path)
        self.dataset = dataset
        self.is_training = is_training



    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i:int):
        sentence = self.dataset[i]['sentence']
        entities_features = self.dataset[i]['entities']
        return sentence,entities_features



def generate_entity_label(seq_length:int,modified_entities,entities_size):
    """"""
    span_label_list = []

    if len(modified_entities) != 0:
        for batch_entity in modified_entities:
            span = np.zeros((seq_length, seq_length,entities_size))

            for entity in batch_entity:
                span[entity['start'],entity['end'],entity['type']] = 1

            span_label_list.append(torch.tensor(span, dtype=torch.float))

    span_labels = torch.stack(span_label_list, dim=0)

    return span_labels


def collate_fn(data,label2id,tokenizer):
    sentence = [i[0]for i in data]
    entities = [i[1] for i in data]

    encode_dic = tokenizer.batch_encode_plus(sentence,
                                        pad_to_max_length=True,
                                        padding=True,
                                        return_offsets_mapping=True,
                                      is_split_into_words=False)
    input_ids = torch.tensor(encode_dic['input_ids'],dtype=torch.long)
    attention_mask = torch.tensor(encode_dic['attention_mask'], dtype=torch.long)
    seq_length = input_ids.shape[1]
    modified_entities = []

    span_mask = []
    default_span = []
    for index in range(len(encode_dic['input_ids'])):
        offset_mapping = encode_dic['offset_mapping'][index]

        """-----------span_mask-----------"""
        token_start_mask, token_end_mask = [], []
        for i, (start_char, end_char) in enumerate(offset_mapping):

            token = tokenizer.convert_ids_to_tokens(encode_dic['input_ids'][index][i])


            if start_char == end_char:
                token_start_mask.append(0)
                token_end_mask.append(0)

            else:
                if end_char-start_char == 1:
                    token_start_mask.append(int(token not in string_mask))
                    token_end_mask.append(int(token not in string_mask))
                else:
                    token_start_mask.append(1)
                    token_end_mask.append(1)


        assert len(token_start_mask) == len(token_end_mask)
        # assert len(negative_list) == len(token_start_mask)

        default_span_mask = [
            [
                (j - i >= 0) * s * e for j, e in enumerate(token_end_mask)
            ]
            for i, s in enumerate(token_start_mask)
        ]
        span_negative_mask = [[x[:] for x in default_span_mask] for _ in label2id]  # for multi-label classification
        span_mask.append(span_negative_mask)
        default_span.append(default_span_mask)
        """-----------span_mask-----------"""
        assert len(offset_mapping) == seq_length
        current_entity = entities[index]
        modified_entity = tools.span_offset(offset_mapping, copy.deepcopy(current_entity),label2id)
        modified_entities.append(modified_entity)


    assert input_ids.shape == attention_mask.shape
    span_labels = generate_entity_label(seq_length, modified_entities,len(label2id))
    default_span_mask = torch.tensor(default_span, dtype=torch.bool)
    span_mask = torch.tensor(span_mask, dtype=torch.bool).permute(0,2,3,1)     # span_mask中标点符号和pad为0

    return input_ids,attention_mask,span_labels,span_mask,default_span_mask

