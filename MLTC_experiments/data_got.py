import numpy as np
from mxnet.contrib import text
import torch.utils.data as data_utils
import torch
# from gensim.models import KeyedVectors
def load_data(batch_size=64):
    X_tst = np.load("/tmp/chinese_ner/MLTC_experiments/data/X_test.npy")
    X_trn = np.load("/tmp/chinese_ner/MLTC_experiments/data/X_train.npy")
    Y_trn = np.load("/tmp/chinese_ner/MLTC_experiments/data/y_train.npy")
    Y_tst = np.load("/tmp/chinese_ner/MLTC_experiments/data/y_test.npy")
    label_embed = np.load("/tmp/chinese_ner/MLTC_experiments/data/label_embed.npy")
    embed = text.embedding.CustomEmbedding('/tmp/chinese_ner/MLTC_experiments/data/word_embed.txt')

    train_data = data_utils.TensorDataset(torch.from_numpy(X_trn).type(torch.LongTensor),
                                          torch.from_numpy(Y_trn).type(torch.LongTensor))
    test_data = data_utils.TensorDataset(torch.from_numpy(X_tst).type(torch.LongTensor),
                                         torch.from_numpy(Y_tst).type(torch.LongTensor))
    train_loader = data_utils.DataLoader(train_data, batch_size, shuffle=True, drop_last=True)
    test_loader = data_utils.DataLoader(test_data, batch_size, drop_last=True)
    return train_loader, test_loader, label_embed, embed.idx_to_vec.asnumpy(), X_tst, embed.token_to_idx, Y_tst, Y_trn

