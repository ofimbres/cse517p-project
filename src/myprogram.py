#!/usr/bin/env python
import os
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
import string
import random
import torch
import torch.nn as nn
import torch.optim as optim
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datasets import load_dataset
import unicodedata
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import re
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm


import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(CharLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        output, hidden = self.lstm(x, hidden)
        output = self.fc(output)
        return output, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim))


class CharDataset(Dataset):
    def __init__(self, text, seq_length):
        self.text = text
        self.seq_length = seq_length
        self.vocab_size = len(set(text))
        self.char2idx = {char: idx for idx, char in enumerate(sorted(set(text)))}
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        self.encoded_text = [self.char2idx[c] for c in text]

    def __len__(self):
        return len(self.encoded_text) - self.seq_length

    def __getitem__(self, idx):
        x = self.encoded_text[idx:idx+self.seq_length]
        y = self.encoded_text[idx+1:idx+self.seq_length+1]
        return torch.tensor(x), torch.tensor(y)
    

class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """
    vocab = None
    char2idx = None
    idx2char = None
    vocab_size = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None

    @classmethod
    def load_training_data(cls, seq_len=10, test_split=0.1):
        dataset_name = "unshuffled_deduplicated" # Remove "en" from the name if you want to load multiple languages

        # Load datasets for multiple languages
        languages = ["en", "es", "fr"]
        datasets = {}
        for lang in languages:
            datasets[lang] = load_dataset("oscar-corpus/OSCAR-2201", language=lang, split="train", streaming=True)
            datasets[lang] = datasets[lang].take(2500)

        # Combine the text from different languages
        text_lines = []
        for lang in languages:
            for d in datasets[lang]:
                text_lines.append(d['text'])

        single_text_string = "".join(text_lines)

        normalized_text_string = normalize_text(single_text_string, lowercase=True, allowed_chars=allowed_chars)
        return normalized_text_string        

    @classmethod
    def load_test_data(cls, fname):
        # your code here
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                print(inp)
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir, epochs=1, lr=0.003):
        # your code here
        chars = sorted(set(data))
        vocab_size = len(chars)
        embedding_dim = 128
        hidden_dim = 256
        num_layers = 2

        model = CharLSTM(vocab_size=vocab_size,
                 embedding_dim=embedding_dim,
                 hidden_dim=hidden_dim,
                 num_layers=num_layers)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)


        n_epochs = 10
        epoch_losses = []

        seq_length = 100
        batch_size = 64
        dataset = CharDataset(train_data, seq_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.003)

        print('Using device {}'.format(self.device))
        self.model.train()

        # Modify training loop to use DataLoader
        for epoch in range(n_epochs):

            total_loss = 0
            hidden = tuple(h.to(device) for h in hidden)
            for x, y in tqdm(dataloader):
                x, y = x.to(device), y.to(device)
                hidden = tuple(h.detach() for h in hidden)

                model.zero_grad()
                output, hidden = model(x, hidden)
                loss = criterion(output.view(-1, vocab_size), y.view(-1))

                total_loss += loss.item()

                loss.backward()
                optimizer.step()

            average_loss = total_loss / len(dataloader)
            epoch_losses.append(average_loss)

            print(f'Epoch {epoch+1}/{epochs} - Loss: {loss.item()}')

            # Save the model checkpoint
            checkpoint_path = os.path.join(work_dir, f'model_epoch_{epoch+1}.pt')
            checkpoint = {
                'vocab_size': vocab_size,
                'embedding_dim': 128,
                'hidden_dim': 256,
                'num_layers': 2,
                'state_dict': model.state_dict(),
                'char2idx': dataset.char2idx, # Also good to save character mappings
                'idx2char': dataset.idx2char
            }
            torch.save(checkpoint, checkpoint_path)
            print(f'Model checkpoint saved to {checkpoint_path}')


    def run_pred(self, data):
        # Predict top-k next characters for each context in data using the provided predict_top_chars logic
        self.model.eval()
        preds = []
        device = self.device
        char2idx = self.char2idx
        idx2char = self.idx2char
        model = self.model

        def predict_top_chars(model, input_text, char2idx, idx2char, top_k=3, hidden=None):
            input_seq = torch.tensor([char2idx[c] for c in input_text]).unsqueeze(0).to(device)
            if hidden is None:
                hidden = (torch.zeros(model.num_layers, 1, model.hidden_dim).to(device),
                          torch.zeros(model.num_layers, 1, model.hidden_dim).to(device))
            output, hidden = model(input_seq, hidden)
            last_output = output[:, -1, :]
            top_indices = torch.topk(last_output, top_k).indices.squeeze().tolist()
            if isinstance(top_indices, int):
                top_indices = [top_indices]
            predicted_chars = [idx2char[idx] for idx in top_indices]
            return predicted_chars, hidden

        with torch.no_grad():
            for context in data:
                context = normalize_text(context, lowercase=True, allowed_chars=allowed_chars)
                pred_chars, _ = predict_top_chars(model, context, char2idx, idx2char)
                preds.append(''.join(pred_chars))
        return preds

    def save(self, work_dir):
        # your code here
        # this particular model has nothing to save, but for demonstration purposes we will save a blank file
        with open(os.path.join(work_dir, 'model.checkpoint'), 'wt') as f:
            f.write('dummy save')

    @classmethod
    def load(cls, work_dir, model_name='model.pt'):
        """
        Loads a model from a checkpoint, using the provided normalized_text_string to get vocab if needed.
        """
        model_path = os.path.join(work_dir, model_name)
        device = cls.device

        with open(model_path, 'rb') as f:
            checkpoint = torch.load(f, map_location=device)

        print(checkpoint.keys()) 

        model = CharLSTM(checkpoint['vocab_size'], checkpoint['embedding_dim'], checkpoint['hidden_dim'], checkpoint['num_layers'])
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()

        mymodel = cls()
        mymodel.model = model
        mymodel.char2idx = checkpoint['char2idx']
        mymodel.idx2char = checkpoint['idx2char']
        mymodel.vocab_size = checkpoint['vocab_size']
        mymodel.device = device
        print(f"Model loaded from {model_path} with vocab size {mymodel.vocab_size}")
        return mymodel

def normalize_text(text, lowercase=True, allowed_chars=None):
    # Apply Unicode NFKC normalization first
    text = unicodedata.normalize("NFKC", text)

    if lowercase:
        text = text.lower()
    if allowed_chars is not None:
        text = ''.join([ch for ch in text if ch in allowed_chars])
    else:
        # Default: keep lowercase letters, digits, basic punctuation, and whitespace
        text = re.sub(r"[^a-z0-9 .,;:!?'\"\n]", '', text)
    # Standardize whitespace
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    return text

# Allowed_chars for English, Spanish, and French
allowed_chars = "abcdefghijklmnopqrstuvwxyz0123456789 .,;:!?'\n’"
# Spanish and French specific characters
allowed_chars += "áéíóúüñ¡¿àèìòùçéèêëàâäôöûüîï"

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instatiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
