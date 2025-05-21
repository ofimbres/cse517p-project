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
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset
import unicodedata
from tqdm import tqdm
from sklearn.model_selection import train_test_split

class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128):
        super(CharLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        logits = self.fc(out[:, -1, :])
        return logits, hidden

class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """
    vocab = sorted(set(string.printable))
    char2idx = {ch: idx for idx, ch in enumerate(vocab)}
    idx2char = {idx: ch for ch, idx in char2idx.items()}
    vocab_size = len(vocab)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self):
        self.model = CharLSTM(self.vocab_size).to(self.device)

    @classmethod
    def load_training_data_multilingual(cls, seq_len=30, test_split=0.1, lang='en', sample_size=10000, pair_fraction=0.1):
        """
        Load and preprocess training data from the OSCAR dataset.
        :param seq_len: Length of context string.
        :param test_split: Fraction for test set.
        :param lang: Language code (e.g., 'en', 'es', 'fr', 'zh').
        :param sample_size: Number of lines to use.
        :param pair_fraction: Fraction of (context, next_char) pairs to sample.
        :return: Training data as list of (context, next_char) tuples.
        """
        
        print(f"Loading OSCAR dataset for language: {lang}")
        dataset = load_dataset("oscar", f"unshuffled_deduplicated_{lang}", split="train", trust_remote_code=True)

        print("Filtering and normalizing text...")
        def normalize(text):
            text = unicodedata.normalize("NFKC", text)
            return ''.join(c for c in text if not unicodedata.category(c).startswith('C'))

        # Filter and normalize
        lines = [normalize(sample['text']) for sample in dataset if sample['text'].strip() != ""]
        sampled_lines = lines[:sample_size]  # limit size for memory

        text = "\n".join(sampled_lines)
        print(f"Combined sample text length: {len(text)}")

        # Generate (context, next_char) pairs
        num_possible = len(text) - seq_len
        num_pairs = int(num_possible * pair_fraction)
        indices = random.sample(range(num_possible), num_pairs)

        data = []
        for i in tqdm(indices, desc="Generating pairs"):
            context = text[i:i + seq_len]
            next_char = text[i + seq_len]
            if all(c in cls.char2idx for c in context + next_char):
                data.append((context, next_char))

        print(f"Total valid training pairs: {len(data)}")
        train_data, _ = train_test_split(data, test_size=test_split, random_state=42)
        return train_data
        

    @classmethod
    def load_training_data(cls, seq_len=10, test_split=0.1):
        # your code here
        dataset_name="wikitext-2-v1"
        #dataset_name = "wikitext-103-v1"
        sample_fraction=0.5
        pair_fraction=0.1

        dataset = load_dataset("wikitext", dataset_name, trust_remote_code=True)
        lines = dataset['train']['text']
        filtered_lines = [line for line in lines if line.strip() != ""]

        # Reduce the amount of text by sampling lines
        num_lines = int(len(filtered_lines) * sample_fraction)
        sampled_lines = random.sample(filtered_lines, num_lines)
        text = "\n".join(sampled_lines)

        print(f"Preparing your training data for a character-level language model (seq_len={seq_len})...")
        # Sample after splitting: randomly select starting indices
        num_possible = len(text) - seq_len
        num_pairs = int(num_possible * pair_fraction)
        if num_pairs < 1:
            num_pairs = 1
        indices = random.sample(range(num_possible), num_pairs)
        data = []
        for i in tqdm(indices, desc="Generating pairs"): # num_possible
            context = text[i:i+seq_len]
            next_char = text[i+seq_len]
            if all(c in cls.char2idx for c in context + next_char):
                data.append((context, next_char))

        print("Splits the dataset into a training set and a test set...")
        train_data, _ = train_test_split(data, test_size=test_split, random_state=42)
        return train_data        

    @classmethod
    def load_test_data(cls, fname):
        # your code here
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                print(inp)
                data.append(inp)

                # for i in range(len(line) - sequence_length):
                #     context = line[i:i+sequence_length]
                #     next_char = line[i+sequence_length]
                #     if all(c in cls.char2idx for c in context + next_char):
                #         data.append((context, next_char))
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir, epochs=1, lr=0.003):
        # your code here
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        print('Training on {} samples'.format(len(data)))
        print('Using device {}'.format(self.device))
        self.model.train()

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            total_loss = 0
            for context, target in tqdm(data):
                x = torch.tensor([[self.char2idx[c] for c in context]], dtype=torch.long).to(self.device)
                y = torch.tensor([self.char2idx[target]], dtype=torch.long).to(self.device)

                optimizer.zero_grad()
                logits, _ = self.model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

        torch.save({
            'model_state_dict': self.model.state_dict(),
        }, os.path.join(work_dir, 'model.pt'))

    def run_pred(self, data):
        # your code here
        self.model.eval()
        preds = []
        with torch.no_grad():
            for context in data:
                x = torch.tensor([[self.char2idx.get(c, 0) for c in context]], dtype=torch.long).to(self.device)
                logits, _ = self.model(x)
                probs = torch.softmax(logits, dim=-1)
                top3 = torch.topk(probs, 3).indices[0].tolist()
                pred_chars = [self.idx2char[i] for i in top3]
                preds.append(''.join(pred_chars))
        return preds
        #for inp in data:
         #  # this model just predicts a random character each time
          #  top_guesses = [random.choice(all_chars) for _ in range(3)]
           # preds.append(''.join(top_guesses))
        #return preds

    def save(self, work_dir):
        # your code here
        # this particular model has nothing to save, but for demonstration purposes we will save a blank file
        with open(os.path.join(work_dir, 'model.checkpoint'), 'wt') as f:
            f.write('dummy save')

    @classmethod
    def load(cls, work_dir):
        # your code here
        # this particular model has nothing to load, but for demonstration purposes we will load a blank file
        #with open(os.path.join(work_dir, 'model.checkpoint')) as f:
         #   dummy_save = f.read()
        #return MyModel()
        model = cls()
        checkpoint = torch.load(os.path.join(work_dir, 'model.pt'), map_location=cls.device)
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.model.to(cls.device)
        model.model.eval()
        return model


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
        train_data = MyModel.load_training_data_multilingual()
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
