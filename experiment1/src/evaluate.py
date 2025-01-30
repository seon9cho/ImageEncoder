from src.process_data import index2sentence
from src.image import tensor2image

import os
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def run_instance_two_models(batch, src_model, trg_model, src_index2word, trg_index2word, name, i):
    features, image_tensor, _ = src_model.encode(batch.src, batch.src_pad_mask)
    decoded = trg_model.greedy_decode_from_memory(batch.src, features)

    src_len = (batch.src[0] != 0).sum().item()
    print_str1 = " Input: " + index2sentence(batch.src[0][:src_len].tolist(), src_index2word)
    print_str2 = "Output: " + index2sentence(decoded[1:].tolist(), trg_index2word)
    trg_len = (batch.trg_y[0]!=0).sum().item()
    print_str3 = "Target: " + index2sentence(batch.trg_y[0][:trg_len].tolist(), trg_index2word)

    log = open("../../outputs/eval/{}.log".format(name), 'a')
    log.write(print_str1 + "\n")
    log.write(print_str2 + "\n")
    log.write(print_str3 + "\n\n")
    log.close()

    image = tensor2image(image_tensor[0])
    fig = plt.figure()
    plt.imshow(image)
    plt.savefig("../../outputs/images/{}/image-{}.png".format(name, i))
    plt.close(fig)

def run_instance(batch, model, src_index2word, trg_index2word, name, i):
    decoded, image_tensor, _ = model.greedy_decode(batch.src[0], batch.src_mask[0])

    src_len = (batch.src[0] != 0).sum().item()
    print_str1 = " Input: " + index2sentence(batch.src[0][:src_len].tolist(), src_index2word)
    print_str2 = "Output: " + index2sentence(decoded[1:].tolist(), trg_index2word)
    trg_len = (batch.trg_y[0]!=0).sum().item()
    print_str3 = "Target: " + index2sentence(batch.trg_y[0][:trg_len].tolist(), trg_index2word)

    log = open("../../outputs/eval/{}.log".format(name), 'a')
    log.write(print_str1 + "\n")
    log.write(print_str2 + "\n")
    log.write(print_str3 + "\n\n")
    log.close()

    image = tensor2image(image_tensor[0])
    fig = plt.figure()
    plt.imshow(image)
    plt.savefig("../../outputs/images/{}/image-{}.png".format(name, i))
    plt.close(fig)


def evaluate_translation_two_models(src_model, trg_model, 
        dataloader, src_index2word, trg_index2word, name):

    history = []
    if not os.path.exists('../../outputs/images/' + name):
        os.mkdir('../../outputs/images/' + name)
    log = open("../../outputs/logs/{}.log".format(name), 'w')
    log.close()
    loader_len = len(dataloader)
    loop = tqdm(total=loader_len, position=0, leave=False)
    for i, batch in enumerate(dataloader):
        run_instance_two_models(
            batch, 
            src_model, 
            trg_model, 
            src_index2word, 
            trg_index2word,
            name, i
        )
        #history.append(score)
        loop.update(1)
    loop.close()
    
    return history

def evaluate_translation(mode, dataloader, src_index2word, trg_index2word, name):
    history = []
    if not os.path.exists('../../outputs/images/' + name):
        os.mkdir('../../outputs/images/' + name)
    log = open("../../outputs/logs/{}.log".format(name), 'w')
    log.close()
    loader_len = len(dataloader)
    loop = tqdm(total=loader_len, position=0, leave=False)
    for i, batch in enumerate(dataloader):
        run_instance(
            batch, 
            model,  
            src_index2word, 
            trg_index2word,
            name, i
        )
        #history.append(score)
        loop.update(1)
    loop.close()
    
    return history
