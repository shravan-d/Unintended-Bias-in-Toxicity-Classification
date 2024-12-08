import pandas as pd
import re
import torch
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize, WhitespaceTokenizer
from contractions import fix
import numpy as np
from tqdm import tqdm


MAX_LENGTH = 220
BATCH_SIZE = 32

class JigsawDataset(Dataset):
    def __init__(self, comments, labels, glove_vocab, max_length):
        self.texts = comments.tolist()
        self.labeAls = labels.tolist()
        self.glove_vocab = glove_vocab
        self.max_length = max_length
        self.tokenizer = WhitespaceTokenizer()
        self.processed_texts = [self._preprocess(text) for text in self.texts]

    def _preprocess(self, text):
        # Expand contractions
        text = fix(text)
        # Convert to lower case
        text = text.lower()
        # Replace underscores with spaces
        text = re.sub(r'[_]', ' ', text)
        # Removing characters that usually don't add meaning to a sentence
        text = re.sub(r"[^?$.-:()%@!&=+/><,a-zA-Z\s0-9\w]", '', text)
        # Changes multiple occurrences of these special characters to only one occurrence. For example '???' to '?'
        text = re.sub(r'([?.!#$%&()*+,-/:;_<=>@[^`|])\1+', r'\1', text)
        # Inserts a space before and after special characters so embeddings can catch them
        text = re.sub(r'([?.!#$%&()*+,-/:;_<=>@[^`|])', r' \1 ', text)
        # Removes extra spaces that may have come in from the previous operation
        text = re.sub(r'([\s])\1+', r'\1', text)
        # Tokenize
        tokens = self.tokenizer.tokenize(text)
        # Filter tokens not in GloVe vocab
        tokens = [token if token in self.glove_vocab else '<unk>' for token in tokens]
        return tokens

    def __len__(self):
        return len(self.processed_texts)

    def __getitem__(self, idx):
        # TODO: Update this function based on the format required for the model
        tokens = self.processed_texts[idx]
        # Pad or truncate to max_length
        if len(tokens) < self.max_length:
            tokens += ['<pad>'] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]
        # Convert tokens to indices
        indices = [self.glove_vocab[token] for token in tokens]
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return torch.tensor(indices, dtype=torch.long), label


# Load GloVe vocabulary
def load_glove_vocab(filepath='../data/glove.6B/glove.6B.50d.txt'):
    glove_vocab = {}
    embeddings = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            values = line.split()
            word = values[0]
            vector = [float(x) for x in values[1:]]
            glove_vocab[word] = idx
            embeddings.append(vector)
    glove_vocab['<pad>'] = len(embeddings)
    embeddings.append([0.0] * len(vector))
    glove_vocab['<unk>'] = len(embeddings)
    embeddings.append([1.0] * len(vector))
    return glove_vocab, torch.tensor(embeddings, dtype=torch.float)


# Example usage
def main():
    df = pd.read_csv('data/train.csv')

    # Load GloVe vocab
    glove_vocab, embedding_matrix = load_glove_vocab('data/glove.6B/glove.6B.100d.txt') #TODO: Tune for optimal distance (50, 100, 200, 300)

    # Define dataset and DataLoader
    dataset = JigsawDataset(df['comment_text'], df['target'], glove_vocab, max_length=MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Check DataLoader
    for batch in dataloader:
        inputs, labels = batch
        print("Input shape:", inputs.shape)
        print("Label shape:", labels.shape)
        break

if __name__ == "__main__":
    main()
