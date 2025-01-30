import numpy as np
import torch
import random
from torch.autograd import Variable
import copy
import json
from tqdm import tqdm
from collections import Counter

PAD = 0
SOS = 1
EOS = 2
UNK = 3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def read_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.read().strip().lower().split('\n')
    return lines

def generate_vocab(data, min_freq=20):
    vocab_cnt = Counter()
    for line in data:
        for word in line.split():
            vocab_cnt[word] += 1
    vocab = [word for word in vocab_cnt if vocab_cnt[word] > min_freq]
    return vocab

def generate_maps(vocab):
    word2index = {'PAD':PAD, 'SOS':SOS, 'EOS':EOS}
    index2word = {PAD:'PAD', SOS:'SOS', EOS:'EOS'}
    for i, w in enumerate(vocab):
        word2index[w] = i + 3
        index2word[i+3] = w
    return word2index, index2word

def sentence2index(s, _map):
    out = []
    for w in s.split():
        if w in _map:
            out.append(_map[w])
    return out

def index2sentence(ind, _map):
    out = []
    for i in ind:
        out.append(_map[i])
    return ' '.join(out)

def convert_lines(lines, _map, max_len=50):
    out = []
    for line in lines:
        if len(line.split()) <= max_len:
            out.append(sentence2index(line, _map))
    return out

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def keys2int(x):
    return {int(k): v for k, v in x.items()}

def save_maps(word2index, index2word, name, map_dir="../../outputs/maps/"):
    with open(map_dir + 'word2index_{}.json'.format(name), 'w') as json_file:
        json.dump(word2index, json_file)
    with open(map_dir + 'index2word_{}.json'.format(name), 'w') as json_file:
        json.dump(index2word, json_file)
        
def load_maps(name, map_dir="../../outputs/maps/"):
    with open(map_dir + 'word2index_{}.json'.format(name), 'r') as json_file:
        word2index = json.load(json_file)
    with open(map_dir + 'index2word_{}.json'.format(name), 'r') as json_file:
        index2word = json.load(json_file, object_hook=keys2int)
    return word2index, index2word

