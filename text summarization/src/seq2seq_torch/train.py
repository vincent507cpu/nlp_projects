import torch

import sys
sys.path.append('/Users/wenjiazhai/Documents/GitHub/nlp_projects/text summarization')

from src.seq2seq_torch.seq2seq_model import Seq2Seq
from src.seq2seq_torch.train_helper import train_model
from src.utils.gpu_utils import config_gpu
from src.utils.params_utils import get_params
from src.utils.wv_loader import Vocab

def train(params, epochs=25):
    # config gpu
    # config_gpu()

    # load vocab for training
    vocab = Vocab(params['vocab_path'], params['vocab_size'])

    # construct model
    print("Building the model ...")
    model = Seq2Seq(params, vocab)

    # train model
    train_model(model, vocab, params, epochs=epochs)

if __name__ == '__main__':
    # load params
    params = get_params()
    params['mode'] == 'train'

    # train
    train(params, epochs=1)