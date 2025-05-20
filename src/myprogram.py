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
    def load_training_data(cls,seq_len=10, test_split=0.1):
        # your code here
        # this particular model doesn't train
        #dataset = load_dataset("tiny_shakespeare",trust_remote_code=True)
        #text = dataset['train'][0]['text']
        dataset = load_dataset("wikitext","wikitext-103-v1",trust_remote_code=True)
        lines = dataset['train']['text']
        # Filter out empty lines
        filtered_lines = [line for line in lines if line.strip() != ""]
        text = "\n".join(filtered_lines)
        data = []
        for i in range(len(text) - seq_len):
            context = text[i:i+seq_len]
            next_char = text[i+seq_len]
            if all(c in cls.char2idx for c in context + next_char):
                data.append((context, next_char))
        
        sample = True
        if sample:
            # Randomly sample 1% of the data
            reduced_data = random.sample(data, int(len(data) * 0.2))
            train_data, _ = train_test_split(reduced_data, test_size=test_split, random_state=42)
        else:
            train_data, _ = train_test_split(data, test_size=test_split, random_state=42)

        return train_data        

    @classmethod
    def load_test_data(cls, fname, sequence_length=10):
        # your code here
        data = []
        with open(fname) as f:
            for line in f:
                line = line.strip()
                #inp = line[:-1]  # the last character is a newline
                #data.append(inp)
                for i in range(len(line) - sequence_length):
                    context = line[i:i+sequence_length]
                    next_char = line[i+sequence_length]
                    if all(c in cls.char2idx for c in context + next_char):
                        data.append((context, next_char))
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir, epochs=3, lr=0.003):
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
        all_chars = string.ascii_letters
        with torch.no_grad():
            for context, _ in data:
                x = torch.tensor([[self.char2idx.get(c, 0) for c in context]], dtype=torch.long).to(self.device)
                logits, _ = self.model(x)
                probs = torch.softmax(logits, dim=-1)
                top3 = torch.topk(probs, 3).indices[0].tolist()
                pred_chars = [self.idx2char[i] for i in top3]
                preds.append(''.join(pred_chars))
        return preds
        #for inp in data:
         #   # this model just predicts a random character each time
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
