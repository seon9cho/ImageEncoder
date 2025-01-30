from src.process_data import save_maps, load_maps
from src.dataset import TranslationDataset, AutoencoderDataset, padding_collate_fn
from src.encoderdecoder import EncoderDecoder, save_model, load_model
from src.loss import LabelSmoothing
from src.optimizer import get_std_opt
from src.train import train
from src.evaluate import evaluate_translation_two_models, evaluate_translation

from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def autoencoder_new_session(args):

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    if args.parallel:
        train_name = args.filename.replace('.', '_') + "-parallel-" + timestamp
    else:
        train_name = args.filename.replace('.', '_') + '-' + timestamp
    print(train_name)
    
    print("Creating train dataset.")
    start_time = time.time()
    file_path = args.base_dir + args.data_dir + args.filename
    dataset = AutoencoderDataset(file_path, min_freq_vocab=args.min_freq_vocab, max_len=args.max_len)
    dataset.init_with_new_maps()
    vocab_size = len(dataset.word2index)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        pin_memory=True, 
        collate_fn=padding_collate_fn, 
        shuffle=True
    )
    print("Number of lines:", len(dataset))
    print("vocab size: ", vocab_size)
    print("elapsed: ", time.time() - start_time)

    print("Creating validation set.")
    val_path = args.base_dir + args.data_dir + args.val_name
    val_dataset = AutoencoderDataset(val_path, max_len=args.max_len)
    val_dataset.init_using_existing_maps(dataset.vocab, dataset.word2index, dataset.index2word)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size, 
        pin_memory=True, 
        collate_fn=padding_collate_fn, 
        shuffle=True
    )
    print("Number of lines:", len(val_dataset))
    print("elapsed:", time.time() - start_time)

    model = EncoderDecoder(
        vocab_size, vocab_size, 
        args.d_model, args.d_ff,  
        args.h, args.N, args.image_layers,
        eval(args.activation), args.dropout
    )
    print("Model created.")
    total_param = sum(p.numel() for p in model.parameters())
    print("Total number of parameters:", total_param)

    criterion = LabelSmoothing(size=vocab_size, padding_idx=0, smoothing=0.1)
    if args.parallel:
        model = nn.DataParallel(model)
        optimizer = get_std_opt(model.module, args.warmup)
    else:
        optimizer = get_std_opt(model, args.warmup)

    print("Start training.")
    save_maps(dataset.word2index, dataset.index2word, train_name)

    model.to(device)
    model, train_history, val_history = train(
        dataloader, val_loader, model, criterion, 
        optimizer, dataset.index2word, dataset.index2word, 
        args.num_epochs, train_name, args.parallel, 
        eval_every = args.eval_every, cp_every=args.cp_every
    )
    save_model(model, train_name)
    np.save("../../outputs/history/train_history_{}.npy".format(train_name), np.array(train_history))
    np.save("../../outputs/history/val_history_{}.npy".format(train_name), np.array(val_history))


def autoencoder_continue_session(args):

    print("Creating train dataset.")
    start_time = time.time()
    file_path = args.base_dir + args.data_dir + args.filename
    dataset = AutoencoderDataset(file_path, min_freq_vocab=args.min_freq_vocab, max_len=args.max_len)
    word2index, index2word = load_maps(args.train_name, map_dir=args.map_dir)
    dataset.init_using_existing_maps(None, word2index, index2word)
    vocab_size = len(dataset.word2index)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        pin_memory=True, 
        collate_fn=padding_collate_fn, 
        shuffle=True
    )
    print("Number of lines:", len(dataset))
    print("vocab size: ", vocab_size)
    print("elapsed: ", time.time() - start_time)

    print("Creating validation set.")
    val_path = args.base_dir + args.data_dir + args.val_name
    val_dataset = AutoencoderDataset(val_path, max_len=args.max_len)
    val_dataset.init_using_existing_maps(None, word2index, index2word)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size, 
        pin_memory=True, 
        collate_fn=padding_collate_fn, 
        shuffle=True
    )
    print("Number of lines:", len(val_dataset))
    print("elapsed:", time.time() - start_time)

    model = load_model(args.train_name + "-cp_{}".format(args.check_point), model_dir=args.model_dir)
    model.to(device)
    print("Model Loaded.")

    criterion = LabelSmoothing(size=vocab_size, padding_idx=0, smoothing=0.1)
    if args.parallel:
        model = nn.DataParallel(model)
        optimizer = get_std_opt(model.module, args.warmup)
    else:
        optimizer = get_std_opt(model, args.warmup)

    print("Start training.")

    log = open("../../outputs/logs/{}.log".format(args.train_name), 'a')
    log.write("Training resumed from model check point {}.".format(args.check_point) + "\n\n")
    log.close()
    
    model.to(device)
    model, train_history, val_history = train(
        dataloader, val_loader, model, criterion, 
        optimizer, dataset.index2word, dataset.index2word, 
        args.num_epochs, args.train_name, args.parallel, 
        eval_every = args.eval_every, cp_every=args.cp_every
    )
    save_model(model, args.train_name)
    np.save("../../outputs/history/train_history_{}.npy".format(args.train_name), np.array(train_history))
    np.save("../../outputs/history/val_history_{}.npy".format(args.train_name), np.array(val_history))




def sup_trans_training_exp(args):

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    filename = '.'.join(args.src_filename.split('.')[:-1]) + '_' + \
        '.'.join(args.trg_filename.split('.')[:-1])
    train_name = filename.replace('.', '_') + '-' + timestamp
    print(train_name)

    src_path = args.data_dir + "train/" + args.src_filename
    trg_path = args.data_dir + "train/" + args.trg_filename
    dataset = TranslationDataset(
        src_path, 
        trg_path, 
        min_freq_vocab=args.min_freq_vocab
    )
    dataset.init_with_new_maps()
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        collate_fn=padding_collate_fn,
        shuffle=True
    )
    src_vocab_size = len(dataset.src_word2index)
    trg_vocab_size = len(dataset.trg_word2index)
    print("Number of lines:", len(dataset))
    print("src vocab size: ", src_vocab_size)
    print("trg vocab size: ", trg_vocab_size)

    model = EncoderDecoder(
        src_vocab_size, trg_vocab_size, 
        args.d_model, args.d_ff,  
        args.h, args.N, args.image_layers,
        eval(args.activation), args.dropout,
        autoencoder=False
    )
    print("Model created.")
    total_param = sum(p.numel() for p in model.parameters())
    print("Total number of parameters:", total_param)
    
    criterion = LabelSmoothing(size=trg_vocab_size, padding_idx=0, smoothing=0.1)
    if args.parallel:
        model = nn.DataParallel(model)
        optimizer = get_std_opt(model.module, args.warmup)
    else:
        optimizer = get_std_opt(model, args.warmup)

    print("Start training.")
    model.to(device)
    model, history = train(
        dataloader, model, criterion, optimizer, 
        dataset.src_index2word, dataset.trg_index2word, 
        args.num_epochs, train_name, args.parallel
    )
    save_maps(dataset.src_word2index, dataset.src_index2word, train_name + "-src")
    save_maps(dataset.trg_word2index, dataset.trg_index2word, train_name + "-trg")
    save_model(model, train_name)

def autoencoder_eval_exp(args):
    file_path = args.base_dir + args.data_dir + args.filename
    dataset = AutoencoderDataset(file_path, min_freq_vocab=args.min_freq_vocab)
    word2index, index2word = load_maps(args.train_name, map_dir=args.map_dir)
    dataset.init_using_existing_maps(word2index, index2word)
    vocab_size = len(dataset.word2index)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        pin_memory=True, 
        collate_fn=padding_collate_fn, 
        shuffle=False
    )
    print("Number of lines:", len(dataset))
    print("vocab size: ", vocab_size)

    model = load_model(args.train_name, model_dir=args.model_dir)
    model.to(device)
    #evaluate_model(model, dataloader, dataset)
    