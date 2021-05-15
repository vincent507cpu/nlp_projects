import torch
from torch import optim
from torch import nn
import time
from functools import partial
import sys
import os

sys.path.append('/Users/wenjiazhai/Documents/GitHub/nlp_projects/text summarization')
from src.seq2seq_torch.seq2seq_batcher import train_batch_generator

def train_model(model, vocab, params, epochs=None):
    if not epochs:
        epochs = params['epochs']

    pad_index = vocab.word2id[vocab.PAD_TOKEN]

    # get vocab size
    params['vocab_size'] = vocab.count

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    train_dataset, val_dataset, train_steps_per_epoch, val_steps_per_epoch = train_batch_generator(
        params['batch_size'], params['max_enc_len'], params['max_dec_len']
    )

    # train
    for epoch in range(epochs):
        start = time.time()
        enc_hidden = model.encoder.init_hidden()

        total_loss = 0.
        running_loss = 0.
        step = 0
        while step * train_steps_per_epoch < len(train_dataset):
            inputs, targets = train_dataset[step * train_steps_per_epoch:(step + 1) * train_steps_per_epoch]

            batch_loss = train_step(model, inputs, targets, enc_hidden,
                                    criterion=partial(criterion, pad_index=pad_index),
                                    optimizer=optimizer)
            total_loss += batch_loss

            if step % 50 == 1:
                print(f'Epoch {epoch + 1} Batch {step + 1} Loss {(total_loss - running_loss) / 50}')
                running_loss = total_loss

            step += 1

        if not os.path.isdir('/Users/wenjiazhai/Documents/GitHub/nlp_projects/text summarization/ckpt'):
            os.mkdir('/Users/wenjiazhai/Documents/GitHub/nlp_projects/text summarization/ckpt')
        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2:
            ckpt_save_path = f'/Users/wenjiazhai/Documents/GitHub/nlp_projects/text summarization/ckpt/checkpoint_epoch_{epoch}.pt'
            torch.save({'epoch':epoch,
                        'model_state_dict':model.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),
                        'loss':total_loss}, ckpt_save_path)

            print(f'Saving checkpoint for epoch {epoch} at {ckpt_save_path}')
        
        valid_loss = evaluate(model, val_dataset, val_steps_per_epoch)
        print(f'Epoch {epoch + 1} Loss {total_loss / train_steps_per_epoch:.2f} ')


# define loss function
def criterion(real, pred, pad_index):
    loss_object = nn.CrossEntropyLoss(reduction='none')
    mask = torch.logical_not(real == pad_index)
    loss_ = loss_object(pred, real)
    mask = mask.type(loss_.type())
    loss_ *= mask
    return loss_.sum()

def train_step(model, enc_input, dec_target, enc_hidden, criterion, optimizer, mode='train'):
    model.train() if mode == 'train' else model.eval()

    optimizer.zero_grad()

    enc_output, enc_hidden = model.encoder(enc_input, enc_hidden)

    # 第一个隐藏层输入
    dec_hidden = enc_hidden

    # 逐个预测序列
    pred = model.teacher_decoder(dec_hidden, enc_output, dec_target)

    batch_loss = criterion(pred, dec_target[:, 1:])

    if mode == 'train':
        batch_loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        optimizer.step()

    return batch_loss

def evaluate(model, val_dataset, val_steps_per_epoch):
    print('Starting evaluation...')
    total_loss = 0.
    enc_hidden = model.encoder.init_hidden()
    step = 0

    while step * val_steps_per_epoch < len(val_dataset):
    # for inputs, targets in val_dataset[]:
        inputs, targets = val_dataset[step * val_steps_per_epoch:(step + 1) * val_steps_per_epoch]
        batch_loss = train_step(model, inputs, targets, enc_hidden, model='eval')
        total_loss += batch_loss
        step += 1
    return total_loss / val_steps_per_epoch
