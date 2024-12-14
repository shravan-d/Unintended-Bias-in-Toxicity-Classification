import pandas as pd
import re
import torch
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize, WhitespaceTokenizer
from contractions import fix
import os


MAX_WORD_LENGTH = 220
BATCH_SIZE = 32

def split_dataframe(df, toxicity_threshold=0.4, val_size=0.05):
    """
    Split the dataframe into training and validation sets. The split is done such that the validation set will contain an 30% toxic and 70% non-toxic samples. The remaining samples will be assigned to the training set. Ideally we wanted 50% in each, but the total count of toxic comments is too less so there wouldn't be enough left in the training set.

    Parameters
    ----------
    df : The dataframe to split
    toxicity_threshold : The threshold for determining whether a comment is toxic (default: 0.5)
    val_size : The proportion of the dataframe to assign to the validation set (default: 0.1)

    Returns
    -------
    tuple
        A tuple of dataframes, where the first element is the training set and the second element is the validation set
    """
    df = df.dropna(subset=['comment_text'])
    df['is_toxic'] = df['target'] >= toxicity_threshold
    toxic_comments = df[df['is_toxic'] == True]
    non_toxic_comments = df[df['is_toxic'] == False]

    # Determine the number of toxic comments for validation set
    toxic_count = int(len(df) * val_size * 0.3)
    non_toxic_count = int(len(df) * val_size * 0.7)

    # Sample toxic and non-toxic comments for validation
    toxic_set = toxic_comments.sample(n=toxic_count, random_state=42)
    non_toxic_set = non_toxic_comments.sample(n=non_toxic_count, random_state=42)

    validation_df = pd.concat([toxic_set, non_toxic_set])
    train_df = df.drop(validation_df.index)

    return train_df, validation_df


class JigsawDataset(Dataset):
    def __init__(self, comments, labels, glove_vocab, max_length=MAX_WORD_LENGTH, only_predict=False):
        self.glove_vocab = glove_vocab
        self.max_length = max_length
        self.tokenizer = WhitespaceTokenizer()
        if only_predict:
            return
        self.texts = comments.tolist()
        self.labels = labels.tolist()
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
    """
    Loads GloVe embeddings from a file and returns a vocabulary dictionary and embeddings tensor.

    Args:
        filepath (str): Path to the GloVe file. Defaults to '../data/glove.6B/glove.6B.50d.txt'.

    Returns:
        tuple: A tuple containing:
            - glove_vocab (dict): A dictionary mapping words to their index in the embeddings.
            - embeddings (torch.Tensor): A tensor of word vectors including special tokens <pad> and <unk>.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError('Glove Embeddings file not found in data dir')

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
    dataset = JigsawDataset(df['comment_text'], df['target'], glove_vocab, max_length=MAX_WORD_LENGTH)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Check DataLoader
    for batch in dataloader:
        inputs, labels = batch
        print("Input shape:", inputs.shape)
        print("Label shape:", labels.shape)
        break

if __name__ == "__main__":
    main()
