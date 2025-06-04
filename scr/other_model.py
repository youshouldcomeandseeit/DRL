import numpy as np
import torch.nn as nn
import torch
import tool
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from transformers import BertModel, AutoModel, RobertaModel

from tool import tools


class Global_Attention(nn.Module):
    def __init__(self, encoder_path, ent_type_size, inner_dim=64):
        super().__init__()
        self.encoder = BertModel.from_pretrained(encoder_path, output_hidden_states=True)
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = self.encoder.config.hidden_size
        self.dense = nn.Linear(self.hidden_size, self.ent_type_size * self.inner_dim * 2)

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)
        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(self.device)
        return embeddings

    def forward(self, input_ids, attention_mask, token_type_ids):
        self.device = input_ids.device

        # last_hidden_state:(batch_size, seq_len, inner_dim)
        hidden = self.encoder(input_ids, attention_mask, token_type_ids)
        last_hidden_state = torch.sum(torch.stack(hidden[2][-4:]), dim=0)

        batch_size = last_hidden_state.size()[0]
        seq_len = last_hidden_state.size()[1]

        # outputs:(batch_size, seq_len, ent_type_size*inner_dim*2)
        outputs = self.dense(last_hidden_state)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        # outputs:(batch_size, seq_len, ent_type_size, inner_dim*2)
        outputs = torch.stack(outputs, dim=-2)
        # qw,kw:(batch_size, seq_len, ent_type_size, inner_dim)
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]

        # pos_emb:(batch_size, seq_len, inner_dim)
        pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim)
        # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)
        cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
        sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
        qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
        qw2 = qw2.reshape(qw.shape)
        qw = qw * cos_pos + qw2 * sin_pos
        kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
        kw2 = kw2.reshape(kw.shape)
        kw = kw * cos_pos + kw2 * sin_pos

        logits = torch.einsum('bmhd,bnhd->bmnh', qw, kw) / self.inner_dim ** 0.5

        return logits


class NoiseMatrixLayer(torch.nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super().__init__()
        self.num_classes = num_classes

        self.noise_layer = nn.Linear(self.num_classes, self.num_classes, bias=False)
        # initialization
        self.noise_layer.weight.data.copy_(torch.eye(self.num_classes))

        init_noise_matrix = torch.eye(self.num_classes)
        self.noise_layer.weight.data.copy_(init_noise_matrix)

        self.eye = torch.eye(self.num_classes).cuda()
        self.scale = scale

    def forward(self, x):
        noise_matrix = self.noise_layer(self.eye)
        # noise_matrix = noise_matrix ** 2
        noise_matrix = F.normalize(noise_matrix, dim=0)
        noise_matrix = F.normalize(noise_matrix, dim=1)
        return noise_matrix * self.scale

class Biaffine(nn.Module):
    def __init__(self, model_name, num_ner_tag, biaffine_size=120, width_embeddings_dim=20):
        super(Biaffine, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        hidden_size = self.bert.config.hidden_size
        hsz = biaffine_size*2+2 + width_embeddings_dim
        self.head_mlp = nn.Sequential(
            nn.Dropout(0.33),
            nn.Linear(hidden_size, biaffine_size),
            nn.LeakyReLU(),
        )
        self.tail_mlp = nn.Sequential(
            nn.Dropout(0.33),
            nn.Linear(hidden_size, biaffine_size),
            nn.LeakyReLU(),
        )
        self.width_embeddings = torch.nn.Embedding(256, width_embeddings_dim, padding_idx=0)
        self.dropout = nn.Dropout(0.1)
        self.U = nn.Parameter(nn.init.xavier_normal_(torch.randn(num_ner_tag, biaffine_size, biaffine_size)))
        self.W = torch.nn.Parameter(torch.empty(num_ner_tag, hsz))
        torch.nn.init.xavier_normal_(self.W.data)
        self.cls = nn.Linear(hsz, num_ner_tag)


    def forward(self, input_ids, attention_mask):
        self.device = input_ids.device
        hidden = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        state = torch.sum(torch.stack(hidden[2][-4:]), dim=0)
        b, seq_len, _ = state.shape
        head_state = self.head_mlp(state)
        tail_state = self.tail_mlp(state)
        scores1 = torch.einsum('bxi,oij,byj->bxyo', head_state, self.U, tail_state)

        head_state = torch.cat([head_state, torch.ones_like(head_state[..., :1])], dim=-1)
        tail_state = torch.cat([tail_state, torch.ones_like(tail_state[..., :1])], dim=-1)
        affined_cat = torch.cat([head_state.unsqueeze(2).expand(-1, -1, tail_state.size(1), -1),
                                 tail_state.unsqueeze(1).expand(-1, head_state.size(1), -1, -1)], dim=-1)

        position_ids = (torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1) + 1).to(self.device)
        span_width_embeddings = self.width_embeddings(position_ids * (position_ids > 0))
        affined_cat = torch.cat([affined_cat,span_width_embeddings.unsqueeze(0).expand(b, -1, -1, -1)],dim=-1)

        scores2 = self.cls(affined_cat)  # bsz x dim x L x L
        scores = scores2 + scores1   # bsz x dim x L x L

        return scores

















































