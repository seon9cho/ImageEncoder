from src.process_data import index2sentence
from src.loss import ComputeLoss
from src.image import tensor2image
from src.encoderdecoder import save_model
from src.dataset import Batch

import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import os
import gc

def run_instance(batch, model, criterion, optimizer):
    output, _ = model(
        batch.src, batch.trg, 
        src_mask=batch.src_pad_mask, 
        trg_mask=batch.trg_attn_mask)

    loss = ComputeLoss(output, batch.trg_y, batch.trg_ntokens, criterion)
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    return loss

def eval_instance(decoded, image_tensor, checkpoint, 
        start, name, src, trg_y, src_index2word, 
        trg_index2word, avg_loss, i, N, e):
    
    elapsed = checkpoint - start
    remaining = (elapsed / (i+1)) * (N - (i+1))
    print_str1 = "Epoch: {}, Iteration: {}, loss: {:.4f}, elapsed: {:.2f}, remaining: {:.2f}"\
                    .format(e, i, avg_loss, elapsed, remaining)
    src_len = (src != 0).sum().item()
    print_str2 = " Input: " + index2sentence(src[:src_len].tolist(), src_index2word)
    print_str3 = "Output: " + index2sentence(decoded[1:].tolist(), trg_index2word)
    trg_len = (trg_y!=0).sum().item()
    print_str4 = "Target: " + index2sentence(trg_y[:trg_len].tolist(), trg_index2word)

    log = open("../../outputs/logs/{}.log".format(name), 'a')
    log.write(print_str1 + "\n")
    log.write(print_str2 + "\n")
    log.write(print_str3 + "\n")
    log.write(print_str4 + "\n\n")
    log.close()

    image = tensor2image(image_tensor[0])
    fig = plt.figure()
    plt.imshow(image)
    plt.savefig("../../outputs/images/{}/image-{}-{}.png".format(name, e, i))
    plt.close(fig)

def train(data_loader, val_loader, model, criterion, 
        optimizer, src_index2word, trg_index2word, 
        num_epochs, name, parallel, eval_every=100, 
        cp_every=100000):
    
    history = []
    val_history = []
    start = time.time()
    if not os.path.exists('../../outputs/images/' + name):
        os.mkdir('../../outputs/images/' + name)
    loader_len = len(data_loader)
    cp_counter = 1
    try:
        for e in range(num_epochs):
            temp_history = []
            loop = tqdm(total=loader_len, position=0, leave=False)
            for i, batch in enumerate(data_loader):
                loss = run_instance(batch, model, criterion, optimizer)
                temp_history.append(loss.item())
                loop.set_description("Epoch: {}, Iteration: {}, loss: {:.4f}"\
                                    .format(e, i, loss.item()))
                loop.update(1)
                if i % eval_every == 0:                
                    if parallel:
                        model.module.eval()
                        decoded, image = model.module.greedy_decode(batch.src[0], batch.src_pad_mask[0])
                        model.module.train()
                    else:
                        model.eval()
                        decoded, image = model.greedy_decode(batch.src[0], batch.src_pad_mask[0])
                        model.train()
                    avg_loss = np.mean(temp_history)
                    eval_instance(decoded, image, time.time(), start, 
                        name, batch.src[0], batch.trg_y[0], src_index2word, trg_index2word, 
                        avg_loss, i, loader_len, e)
                    history.append(avg_loss)
                    temp_history = []
                    
                del batch
                gc.collect()

            # Model Checkpoint
            model.eval()
            val_loss = validate(val_loader, model, criterion, name, cp_counter)
            val_history.append(val_loss)
            model.train()
            save_model(model, name + "-cp_{}".format(cp_counter))
            cp_counter+=1

            loop.close()

    except KeyboardInterrupt:
        return model, history, val_history
    
    return model, history, val_history

def validate(data, model, criterion, name, cp_counter):
    loss_history = []
    for _, batch in enumerate(data):
        output, _ = model(
            batch.src, batch.trg, 
            src_mask=batch.src_pad_mask, 
            trg_mask=batch.trg_attn_mask)
        with torch.no_grad():
            loss = ComputeLoss(output, batch.trg_y, batch.trg_ntokens, criterion)
        loss_history.append(loss.item())
    avg_loss = np.mean(loss_history)
    log = open("../../outputs/logs/{}.log".format(name), 'a')
    log.write("Validation loss, cp_{}: {}".format(cp_counter, avg_loss) + "\n\n")
    return avg_loss
