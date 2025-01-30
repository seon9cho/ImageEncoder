from src.process_data import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PAD = 0
SOS = 1
EOS = 2
UNK = 3

class TranslationDataset():
    def __init__(self, src_path, trg_path, min_freq_vocab=20, max_len=50):
        self.src_lines = read_lines(src_path)
        self.trg_lines = read_lines(trg_path)
        self.min_freq_vocab = min_freq_vocab
        self.max_len = max_len
        self.src_vocab = None
        self.src_word2index = None
        self.src_index2word = None
        self.src_data = None
        
        self.trg_vocab = None
        self.trg_word2index = None
        self.trg_index2word = None
        self.trg_data = None

    def init_with_new_maps(self):
        print("Generating vocab...")
        self.src_vocab = generate_vocab(self.src_lines, min_freq=self.min_freq_vocab)
        self.trg_vocab = generate_vocab(self.trg_lines, min_freq=self.min_freq_vocab)
        print("Generating maps...")
        self.src_word2index, self.src_index2word = generate_maps(self.src_vocab)
        self.trg_word2index, self.trg_index2word = generate_maps(self.trg_vocab)
        print("Converting lines to indices...")
        self.src_data = convert_lines(self.src_lines, self.src_word2index)
        self.trg_data = convert_lines(self.trg_lines, self.trg_word2index)
    
    def init_using_existing_maps(self, src_vocab, src_word2index, src_index2word, 
                                trg_vocab, trg_word2index, trg_index2word):
        self.src_vocab = src_vocab
        self.src_word2index = src_word2index
        self.src_index2word = src_index2word
        self.trg_vocab = trg_vocab
        self.trg_word2index = trg_word2index
        self.trg_index2word = trg_index2word
        print("Converting lines to indices...")
        self.src_data = convert_lines(self.src_lines, self.src_word2index)
        self.trg_data = convert_lines(self.trg_lines, self.trg_word2index)
    
    def __getitem__(self, idx):
        return self.src_data[idx], self.trg_data[idx]
    
    def __len__(self):
        return len(self.src_data)

class AutoencoderDataset():
    def __init__(self, path, min_freq_vocab=5, max_len=50):
        self.lines = read_lines(path)
        self.min_freq_vocab = min_freq_vocab
        self.max_len = max_len
        self.vocab = None
        self.word2index = None
        self.index2word = None
        self.data = None

    def init_with_new_maps(self):
        print("Generating vocab...")
        self.vocab = generate_vocab(self.lines, min_freq=self.min_freq_vocab)
        print("Generating maps...")
        self.word2index, self.index2word = generate_maps(self.vocab)
        print("Converting lines to indices...")
        self.data = convert_lines(self.lines, self.word2index, self.max_len)
    
    def init_using_existing_maps(self, vocab, word2index, index2word):
        self.vocab = vocab
        self.word2index = word2index
        self.index2word = index2word
        print("Converting lines to indices...")
        self.data = convert_lines(self.lines, self.word2index, self.max_len)
    
    def __getitem__(self, idx):
        return self.data[idx], self.data[idx]
    
    def __len__(self):
        return len(self.data)

class Batch:
    def __init__(self, batch, bidirectional=False):
        batch_size = len(batch)
        max_length1 = max([len(item[0]) for item in batch]) + 1
        max_length2 = max([len(item[1]) for item in batch]) + 1

        self.src = torch.zeros((batch_size, max_length1), dtype=torch.float)
        self.src_y = torch.zeros((batch_size, max_length1), dtype=torch.float)
        self.trg = torch.zeros((batch_size, max_length2), dtype=torch.float)
        self.trg_y = torch.zeros((batch_size, max_length2), dtype=torch.float)
        
        for i, seq in enumerate(batch):
            src_seq = seq[0].copy()
            src_seq.insert(0, SOS)
            src_y_seq = seq[0].copy()
            src_y_seq.append(EOS)

            trg_seq = seq[1].copy()
            trg_seq.insert(0, SOS)
            trg_y_seq = seq[1].copy()
            trg_y_seq.append(EOS)

            self.src[i, :len(seq[0])+1] = torch.tensor(src_seq)
            self.src_y[i, :len(seq[0])+1] = torch.tensor(src_y_seq)
            self.trg[i, :len(seq[1])+1] = torch.tensor(trg_seq)
            self.trg_y[i, :len(seq[1])+1] = torch.tensor(trg_y_seq)
        
        self.src = self.src.long().to(device)
        self.src_y = self.src_y.long().to(device)
        self.src_pad_mask = ((self.src != PAD) & (self.src != SOS)).unsqueeze(-2)
        self.src_attn_mask = self.make_std_mask(self.src)
        self.src_ntokens = (self.src_y != PAD).data.sum()

        self.trg = self.trg.long().to(device)
        self.trg_y = self.trg_y.long().to(device)
        self.trg_pad_mask = ((self.trg != PAD) & (self.trg != SOS)).unsqueeze(-2)
        self.trg_attn_mask = self.make_std_mask(self.trg)
        self.trg_ntokens = (self.trg_y != PAD).data.sum()
    
    @staticmethod
    def make_std_mask(trg):
        "Create a mask to hide padding and future words."
        trg_mask = (trg != PAD).unsqueeze(-2)
        trg_mask = trg_mask & Variable(
            subsequent_mask(trg.size(-1)).type_as(trg_mask.data))
        return trg_mask

def padding_collate_fn(data_batch):
    batch = Batch(data_batch)
    return batch