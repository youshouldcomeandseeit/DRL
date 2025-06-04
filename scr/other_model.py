import numpy as np
import torch.nn as nn
import torch
import tool
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from transformers import BertModel, AutoModel, RobertaModel

from tool import tools


class _Encoder(nn.Module):
    def __init__(self, ent_type_size, encoder):
        super(_Encoder, self).__init__()
        self.ent_type_size = ent_type_size
        self.encoder = encoder



    def forward(self, input_ids, attention_mask, token_type_ids):
        hidden = self.encoder(input_ids, attention_mask, token_type_ids)
        token_embeddings = torch.sum(torch.stack(hidden[2][-4:]), dim=0)


        return token_embeddings




class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.encoder_path, output_hidden_states=True)

        self.hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(config.dropout)
        self._encoder = _Encoder(config.ent_type_size, self.bert)
        self.ent_type_size = config.ent_type_size
        self.span_linear = torch.nn.Linear(self.hidden_size * 2 , config.linear_size)
        self.cls = torch.nn.Linear(config.linear_size , self.ent_type_size)
        self.width_embeddings = torch.nn.Embedding(config.max_span_width, config.linear_size, padding_idx=0)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.bert.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.bert.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, token_type_ids,mask):
        self.device = input_ids.device
        token_embeddings = self._encoder(input_ids, attention_mask, token_type_ids)  # [b,seq,dim]



        b, seq_len, _ = token_embeddings.shape

        span_embeddings = torch.cat(
            [
                token_embeddings.unsqueeze(2).expand(-1, -1, seq_len, -1),
                token_embeddings.unsqueeze(1).expand(-1, seq_len, -1, -1),],dim=3)

        span_linear = self.span_linear(span_embeddings) # [b,seq,seq,dim]
        scores = self.cls(span_linear).permute(0,-1,1,2)
        return scores









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





class LayerNorm(nn.Module):
    def __init__(self, input_dim, cond_dim=0, center=True, scale=True, epsilon=None, conditional=False,
                 hidden_units=None, hidden_activation='linear', hidden_initializer='xaiver', **kwargs):
        super(LayerNorm, self).__init__()
        """
        input_dim: inputs.shape[-1]
        cond_dim: cond.shape[-1]
        """
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        self.hidden_initializer = hidden_initializer
        self.epsilon = epsilon or 1e-12
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        if self.center:
            self.beta = nn.Parameter(torch.zeros(input_dim))
        if self.scale:
            self.gamma = nn.Parameter(torch.ones(input_dim))

        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = nn.Linear(in_features=self.cond_dim, out_features=self.hidden_units, bias=False)
            if self.center:
                self.beta_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)
            if self.scale:
                self.gamma_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)

        self.initialize_weights()

    def initialize_weights(self):

        if self.conditional:
            if self.hidden_units is not None:
                if self.hidden_initializer == 'normal':
                    torch.nn.init.normal(self.hidden_dense.weight)
                elif self.hidden_initializer == 'xavier':  # glorot_uniform
                    torch.nn.init.xavier_uniform_(self.hidden_dense.weight)

            if self.center:
                torch.nn.init.constant_(self.beta_dense.weight, 0)
            if self.scale:
                torch.nn.init.constant_(self.gamma_dense.weight, 0)

    def forward(self, inputs, cond=None):
        if self.conditional:
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)

            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(1)  # cond = K.expand_dims(cond, 1)

            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = torch.mean(outputs, dim=-1).unsqueeze(-1)
            outputs = outputs - mean
        if self.scale:
            variance = torch.mean(outputs ** 2, dim=-1).unsqueeze(-1)
            std = (variance + self.epsilon) ** 0.5
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta

        return outputs


class ConvolutionLayer(nn.Module):
    def __init__(self, input_size, channels, dilation, dropout=0.1):
        super(ConvolutionLayer, self).__init__()
        self.base = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(input_size, channels, kernel_size=1),
            nn.GELU(),
        )

        self.convs = nn.ModuleList(
            [nn.Conv2d(channels, channels, kernel_size=3, groups=channels, dilation=d, padding=d) for d in dilation])

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.base(x)

        outputs = []
        for conv in self.convs:
            x = conv(x)
            x = F.gelu(x)
            outputs.append(x)
        outputs = torch.cat(outputs, dim=1)
        outputs = outputs.permute(0, 2, 3, 1).contiguous()
        return outputs


class _Biaffine_(nn.Module):
    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(_Biaffine_, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        weight = torch.zeros((n_out, n_in + int(bias_x), n_in + int(bias_y)))
        nn.init.xavier_normal_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        # remove dim 1 if n_out == 1
        s = s.permute(0, 2, 3, 1)

        return s


class MLP(nn.Module):
    def __init__(self, n_in, n_out, dropout=0):
        super().__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        x = self.activation(x)
        return x


class CoPredictor(nn.Module):
    def __init__(self, cls_num, hid_size, biaffine_size, channels, ffnn_hid_size, dropout=0):
        super().__init__()
        self.mlp1 = MLP(n_in=hid_size, n_out=biaffine_size, dropout=dropout)
        self.mlp2 = MLP(n_in=hid_size, n_out=biaffine_size, dropout=dropout)
        self.biaffine = _Biaffine_(n_in=biaffine_size, n_out=cls_num, bias_x=True, bias_y=True)
        self.mlp_rel = MLP(channels, ffnn_hid_size, dropout=dropout)
        self.linear = nn.Linear(ffnn_hid_size, cls_num)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y, z):
        h = self.dropout(self.mlp1(x))
        t = self.dropout(self.mlp2(y))
        o1 = self.biaffine(h, t)

        z = self.dropout(self.mlp_rel(z))
        o2 = self.linear(z)
        return o1 + o2


class W2WModel(nn.Module):
    def __init__(self, encoder_path,num_ner_tag):
        super(W2WModel, self).__init__()
        self.use_bert_last_4_layers = True

        self.lstm_hid_size = 512
        self.conv_hid_size = 128

        self.bert = BertModel.from_pretrained(encoder_path, output_hidden_states=True)

        self.encoder = nn.LSTM(self.bert.config.hidden_size, self.lstm_hid_size // 2, num_layers=1, batch_first=True,
                               bidirectional=True)


        self.convLayer = ConvolutionLayer(self.lstm_hid_size, self.conv_hid_size, [1, 2, 3, 4], 0.5)
        self.dropout = nn.Dropout(0.5)
        self.predictor = CoPredictor(num_ner_tag, self.lstm_hid_size, 512,
                                     self.conv_hid_size * 4, 384,0.33)
        #
        self.cln = LayerNorm(self.lstm_hid_size, self.lstm_hid_size, conditional=True)

    def forward(self, input_ids, attention_mask, token_type_ids,mask):

        bert_embs = self.bert(input_ids, attention_mask, token_type_ids)
        bert_embs = torch.stack(bert_embs[2][-4:], dim=-1).mean(-1)


        sent_length = torch.sum(attention_mask,dim=-1)

        word_reps = self.dropout(bert_embs)
        packed_embs = pack_padded_sequence(word_reps, sent_length.cpu(), batch_first=True, enforce_sorted=False)
        packed_outs, (hidden, _) = self.encoder(packed_embs)
        word_reps, _ = pad_packed_sequence(packed_outs, batch_first=True, total_length=sent_length.max())

        cln = self.cln(word_reps.unsqueeze(2), word_reps)

        conv_inputs = torch.masked_fill(cln, mask.eq(0).unsqueeze(-1), 0.0)
        conv_outputs = self.convLayer(conv_inputs)
        conv_outputs = torch.masked_fill(conv_outputs, mask.eq(0).unsqueeze(-1), 0.0)
        outputs = self.predictor(word_reps, word_reps, conv_outputs)

        return outputs.permute(0,-1,1,2)













































