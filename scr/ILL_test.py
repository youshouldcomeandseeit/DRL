import argparse

import math
import torch
import transformers
import wandb
# import wandb
from transformers import BertTokenizerFast, RobertaTokenizerFast
from transformers import AutoTokenizer




import os
import random
from functools import partial
import numpy as np
import torch.nn.functional as F
import torch
import config
from other_model import Biaffine, Global_Attention, NoiseMatrixLayer
import loss_func
from data_processing import collate_fn, Dataset
from tqdm import tqdm
import model


from tool import tools
from tool.logger import logger
from tool.tools import MetricsCalculator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def load_data(dataset):
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=config.batch_size,
                                         collate_fn=partial(collate_fn, label2id=config.label2id,tokenizer=tokenizer),
                                         shuffle=True,
                                         drop_last=False,
                                         num_workers=8,
                                         )
    return loader
metrics = MetricsCalculator()

class Trainer(object):
    def __init__(self, model, noise_model, loss_function):
        self.model = model
        self.criterion = loss_function
        self.noise_model = noise_model
        bert_params = set(self.model.bert.parameters())
        other_params = list(set(self.model.parameters()) - bert_params)
        no_decay = ['bias', 'LayerNorm.weight']
        params = [
            {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': config.bert_learning_rate,
             'weight_decay': 0.0},
            {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': config.bert_learning_rate,
             'weight_decay': 0.0},
            {'params': other_params,
             'lr': config.model_learning_rate,
             'weight_decay': 0.0},
        ]

        self.optimizer = torch.optim.AdamW(params,lr=config.model_learning_rate,weight_decay=0.0)
        self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer,
                                                                 num_warmup_steps=0.1 * ((len(train_dataset) // config.batch_size) * config.epochs),
                                                                 num_training_steps=((len(train_dataset) // config.batch_size) * config.epochs))

        self.noise_optimizer = torch.optim.AdamW(self.noise_model.parameters(), lr=config.model_learning_rate, weight_decay=0.0)
        self.noise_scheduler = transformers.get_linear_schedule_with_warmup(self.noise_optimizer,num_warmup_steps=0.1 * ((len(train_dataset) // config.batch_size) * config.epochs),
                                                                      num_training_steps=((len(train_dataset) // config.batch_size) * config.epochs))
    def train(self,epoch,train_dataset):
        train_dataloader = load_data(train_dataset)
        self.model.train()
        self.noise_model.train()
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        loss_list = []
        for batch_ind, batch_data in pbar:

            batch_data = [data.cuda() for data in batch_data]
            input_ids, attention_mask, span_labels,span_mask,default_span_mask = batch_data

            logits_w, logits_s = self.model(input_ids, attention_mask)
            logits_w = logits_w[span_mask == 1].view(-1, config.ent_type_size)
            logits_s = logits_s[span_mask == 1].view(-1, config.ent_type_size)
            span_labels = span_labels[default_span_mask == 1].long()
            noise_matrix = self.noise_model(logits_w)

            probs_x_w = logits_w.softmax(dim=-1).clamp(min=1e-7, max=1.0).detach()

            # convert logits_s to probs
            probs_x_s = logits_s.softmax(dim=-1).clamp(min=1e-7, max=1.0)

            with torch.no_grad():
                # model p(y_hat | y, x) p(y|x)
                noise_matrix_col = noise_matrix.softmax(dim=-1)[:, span_labels].detach().transpose(0, 1)
                em_y = probs_x_w * noise_matrix_col
                em_y = em_y / em_y.sum(dim=1, keepdim=True)
            em_probs_x_s = probs_x_s * noise_matrix_col
            em_probs_x_s = em_probs_x_s / em_probs_x_s.sum(dim=1, keepdim=True)

            # compute observed noisy labels
            noise_matrix_row = noise_matrix.softmax(dim=0)
            noisy_probs_x_w = torch.matmul(logits_w.softmax(dim=-1).clamp(min=1e-7, max=1.0), noise_matrix_row)
            noisy_probs_x_w = noisy_probs_x_w / noisy_probs_x_w.sum(dim=-1, keepdims=True)

            # compute noisy loss
            noise_loss = torch.mean(-torch.sum(F.one_hot(span_labels, config.ent_type_size) * torch.log(noisy_probs_x_w), dim=-1))

            em_loss = torch.mean(-torch.sum(em_y * torch.log(em_probs_x_s), dim=-1), dim=-1)
            max_probs, hard_pseudo_label = torch.max(probs_x_w, dim=-1)
            mask = max_probs.ge(0.6).float()
            con_loss = torch.mean(-(probs_x_w * torch.log_softmax(logits_s, dim=-1)).sum(dim=-1) * mask)

            loss = noise_loss + em_loss + con_loss
            # avg_prediction = torch.mean(logits_s.softmax(dim=-1), dim=0)
            # prior_distr = 1.0/config.ent_type_size * torch.ones_like(avg_prediction)
            # avg_prediction = torch.clamp(avg_prediction, min = 1e-6, max = 1.0)
            # balance_kl = torch.mean(-(prior_distr * torch.log(avg_prediction)).sum(dim=0))

            entropy_loss = loss_func.SCELoss(alpha=1,beta=1,num_classes=config.ent_type_size)(logits_s,span_labels)

            loss += entropy_loss

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            self.noise_optimizer.step()
            self.optimizer.zero_grad()
            self.noise_optimizer.zero_grad()
            self.scheduler.step()
            self.noise_scheduler.step()

            # torch.cuda.empty_cache()
            loss_list.append(loss.cpu().item())

            avg_loss = sum(loss_list) / len(loss_list)
            pbar.set_description(
                f' Epoch: {epoch + 1}/{config.epochs}, Step: {batch_ind + 1}/{len(train_dataloader)}')
            pbar.set_postfix(loss=avg_loss, lr=self.optimizer.param_groups[0]["lr"])

            if batch_ind % 50 == 0:
                logger.info({
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "learning_rate": self.optimizer.param_groups[0]['lr'],
                })
        # wandb.log({'epoch': epoch, 'training_loss': sum(loss_list) / len(loss_list)})
        return sum(loss_list) / len(loss_list)


    def eval(self,epoch,dev_dataset=None,test_dataset=None,is_dev=False,is_test=False):
        self.model.eval()
        if is_dev:
            dev_dataloader = load_data(dev_dataset)
            all_intersection = 0
            all_pred = 0
            all_true = 0
            for batch_data in tqdm(dev_dataloader, desc="Validating"):
                batch_data = [data.cuda() for data in batch_data]
                input_ids, attention_mask, label, span_mask, default_span_mask = batch_data
                with torch.no_grad():
                    logits_w, logits_s = self.model(input_ids, attention_mask)

                    probability = logits_w.softmax(dim=-1)
                    probability.clamp_(0, 1)  # 限制值在[0, 1]之间，避免精度问题
                    pred = torch.argmax(probability, dim=-1)
                    pred = pred[default_span_mask]
                    label = label[default_span_mask]

                    pred = F.one_hot(pred.long(), config.ent_type_size)
                    label = F.one_hot(label.long(), config.ent_type_size)

                    batch_intersection, batch_p, batch_t = metrics.get_evaluate_fpr_single(pred, label)
                    all_intersection += batch_intersection
                    all_pred += batch_p
                    all_true += batch_t
            if all_pred == 0 or all_true == 0 or all_intersection == 0:
                return 0
            else:
                precision = all_intersection / all_pred
                recall = all_intersection / all_true
                f1 = 2 * (precision * recall) / (precision + recall)

                print("******************************************")
                print(
                    f'dev_precision: {round(precision * 100, 2)}, dev_recall: {round(recall * 100, 2)}, dev_f1: {round(f1 * 100, 2)}')
                print("******************************************")

                logger.info({"dev_precision": precision, "dev_recall": recall, "dev_f1": f1})
                return round(f1 * 100, 2)

        if is_test:
            all_intersection = 0
            all_pred = 0
            all_true = 0
            test_dataloader = load_data(test_dataset)
            for batch_data in tqdm(test_dataloader, desc="Testing"):
                batch_data = [data.cuda() for data in batch_data]
                input_ids,attention_mask, label, span_mask,default_span_mask = batch_data

                with torch.no_grad():
                    logits_w, logits_s = self.model(input_ids, attention_mask)

                    probability = logits_s[span_mask == 1].view(-1, config.ent_type_size).softmax(dim=-1)
                    probability.clamp_(0, 1)  # 限制值在[0, 1]之间，避免精度问题
                    pred = torch.argmax(probability,dim=-1)
                    label = label[default_span_mask]

                    pred = F.one_hot(pred.long(), config.ent_type_size)
                    label = F.one_hot(label.long(), config.ent_type_size)

                    batch_intersection, batch_p, batch_t = metrics.get_evaluate_fpr_single(pred, label)
                    all_intersection += batch_intersection
                    all_pred += batch_p
                    all_true += batch_t
            if all_pred == 0 or all_true == 0 or all_intersection == 0:
                return 0,0,0
            else:
                precision = all_intersection / all_pred
                recall = all_intersection / all_true
                f1 = 2 * (precision * recall) / (precision + recall)

                print("******************************************")
                print(
                    f'precision: {round(precision * 100, 2)}, recall: {round(recall * 100, 2)}, f1: {round(f1 * 100, 2)}')
                print("******************************************")

            logger.info({"precision": precision, "recall": recall, "f1": f1})
            return round(precision * 100, 2), round(recall * 100, 2), round(f1 * 100, 2)


def loss_function(pred, true, mask, default_span_mask, epoch):
    # beta = 0.1 * math.log(epoch + 1)
    # sample_loss, class_loss = loss_func.TwoWayLoss(beta,robust=False)(pred, true, mask)
    # _alpha = 1. - math.exp(- 0.12 * (epoch + 1))
    # loss = (_alpha) * class_loss + (1 - _alpha) * sample_loss

    # loss = loss_func.Boundary_smoothing(sb_epsilon=0.2, sb_size=1)(pred, true, mask)

# 59.01 60.54 58.96 59.06 58.61 59.06 58.72
    # loss = loss_func.Hill(lamb=1.5, margin=0.5, gamma=2.)(pred,true,mask)

    # loss = loss_func.Symmetric_CELoss(alpha=1.,beta=1.)(pred,true.float(),mask)
    # loss = loss_func.AsymmetricLossOptimized(gamma_neg=2, gamma_pos=1, clip=0.05)(pred,true,mask)


    # loss = loss_func.BCELossWithLabelSmoothing(alpha=_alpha)(pred, true, mask)
    # loss = loss_func.BCELossWithLabelSmoothing(alpha=0.1)(pred, true, mask)

    pred = pred[mask].view(-1,config.ent_type_size)
    true = true[mask].view(-1,config.ent_type_size)
    loss = loss_func.multilabel_categorical_crossentropy(pred,true)
    # loss = loss_func.BCELoss(pred, true, mask)
    # loss = loss_func.BCEFocalLoss(gamma=2)(pred,true,mask)
    # loss = loss_func.ASLLoss(gamma_neg=2, gamma_pos=1)(pred,true,mask)
    # loss = loss_func.GHMC(bins=30)(pred,true,mask.float())
    # return loss, sample_loss, class_loss
    return loss

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/conll03_crowd.json')
    # parser.add_argument('--save_path', type=str, default='./model.pt')
    # parser.add_argument('--predict_path', type=str, default='./output.json')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--encoder_path', type=str)
    parser.add_argument('--train_datasets_path', type=str)
    parser.add_argument('--dev_datasets_path', type=str)
    parser.add_argument('--test_datasets_path', type=str)
    parser.add_argument('--linear_size', type=int)
    parser.add_argument('--ent_type_size', type=int)
    parser.add_argument('--label2id', type=dict)
    parser.add_argument('--label_encode_list', type=list)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    config = config.Config(args)
    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.encoder_path,use_fast=True)
    train_dataset = Dataset(config.train_datasets_path,is_training=True)
    dev_dataset = Dataset(config.dev_datasets_path,is_training=False)
    test_dataset = Dataset(config.test_datasets_path)
    score_list = []
    # logger.info(config)
    for i in range(8):
            torch.cuda.empty_cache()
            # wandb.init(project='conll_weak',
            #            name=f"analysis_large_loss_matter",
            #            config=config,
            #            resume='allow')
            best_dev_f1 = 0.
            max_f1 = 0.
            max_p = 0.
            max_r = 0.
            # set_seed(config.seed)
            my_model = Biaffine(config.encoder_path, config.ent_type_size, biaffine_size=120,
                                width_embeddings_dim=20).cuda()
            noise_model = NoiseMatrixLayer(config.ent_type_size,1.).cuda()
            trainer = Trainer(my_model, noise_model, loss_function)

            for epoch in range(config.epochs):

                # pos_count_bin = torch.zeros(10)
                # neg_count_bin = torch.zeros(10)
                torch.cuda.empty_cache()
                trainer.train(epoch, train_dataset)
                dev_f1 = trainer.eval(epoch, dev_dataset=dev_dataset, test_dataset=test_dataset, is_dev=True, is_test=False)

                if dev_f1 > best_dev_f1:
                    best_dev_f1 = dev_f1
                    test_p, test_r, test_f1 = trainer.eval(epoch, test_dataset=test_dataset, is_dev=False, is_test=True)
                    max_f1 = test_f1
                    max_p = test_p
                    max_r = test_r
            score_list.append(max_f1)

            del my_model
            torch.cuda.empty_cache()
            # wandb.finish()

    print(score_list)
    # logger.info(score_list)


