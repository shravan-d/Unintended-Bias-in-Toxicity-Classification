import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from dataset import JigsawDataset, load_glove_vocab, MAX_WORD_LENGTH, clean_text_for_roberta
from transformers import RobertaTokenizer, RobertaForSequenceClassification

class ToxicityClassifierLSTM(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, output_dim, num_layers, dropout=None):
        """
        Initialize the ToxicityClassifierLSTM model.

        Parameters
        ----------
        embedding_matrix : torch.Tensor
            Pre-trained word embeddings to be used as input embeddings.
        hidden_dim : int
            The number of features in the hidden state of the LSTM.
        output_dim : int
            The size of the output layer.
        num_layers : int
            The number of recurrent layers in the LSTM.
        dropout : float, optional
            The dropout probability for the LSTM layers. Default is None, which means no dropout.
        """
        super(ToxicityClassifierLSTM, self).__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        self.dropout = nn.Dropout1d(0.3)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if dropout is not None else 0)
        self.lstm_layer_1 = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.lstm_layer_2 = nn.LSTM(hidden_dim * 2, hidden_dim, bidirectional=True, batch_first=True)
        self.linear1 = nn.Linear(4 * hidden_dim, 4 * hidden_dim)
        self.linear2 = nn.Linear(4 * hidden_dim, 4 * hidden_dim)
        self.fc = nn.Linear(4 * hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        lstm_out, _ = self.lstm_layer_1(embedded)
        lstm_out, _ = self.lstm_layer_2(lstm_out)
        
        # Global average pooling
        avg_pool = torch.mean(lstm_out, 1)
        # Global max pooling
        max_pool, _ = torch.max(lstm_out, 1)

        pool_out = torch.cat((max_pool, avg_pool), 1)
        linear1_out  = F.relu(self.linear1(pool_out))
        linear2_out  = F.relu(self.linear2(pool_out))
        
        final_hidden_state = pool_out + linear1_out + linear2_out
        
        return self.fc(final_hidden_state)


class EnsembleClassifier(nn.Module):
    def __init__(self, checkpoints_dir, DEVICE):
        """
        Parameters
        ----------
        checkpoints_dir : str
            The directory containing the model checkpoints.
        DEVICE : torch.device
            The device on which the model should be loaded.

        Attributes
        ----------
        glove_vocab : dict
            The vocabulary of GloVe vectors.
        lstm : ToxicityClassifierLSTM
            The LSTM model.
        tokenizer : transformers.PreTrainedTokenizer
            The tokenizer for the Roberta model.
        roberta : transformers.PreTrainedModel
            The Roberta model.
        device : torch.device
            The device on which the models are loaded.
        """
        super(ToxicityClassifierLSTM, self).__init__()
        self.glove_vocab, glove_embeddings = load_glove_vocab('data/glove.6B/glove.6B.100d.txt')
        self.lstm = ToxicityClassifierLSTM(
            embedding_matrix=glove_embeddings,
            hidden_dim=128,
            output_dim=7,
            num_layers=2,
        )
        self.lstm.load_state_dict(torch.load(f'{checkpoints_dir}/LSTM.pth', weights_only=False, map_location=DEVICE))
        self.lstm = self.lstm.to(DEVICE)

        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.roberta = RobertaForSequenceClassification.from_pretrained(checkpoints_dir)
        self.roberta = self.roberta.to(DEVICE)

        self.device = DEVICE

    def forward(self, sentence):
        lstm_out = predict_LSTM(sentence, self.lstm, self.device, self.glove_vocab)
        roberta_out = predict_roberta(sentence, self.roberta, self.device, self.tokenizer)

        result = 0.65 * roberta_out + 0.35 * lstm_out
        return result


def loadModel(model_name, checkpoints_dir, DEVICE):
    """
    Load a model given the model name and the path to the checkpoint directory.

    Parameters
    ----------
    model_name : str
        The name of the model to be loaded. Options are 'lstm', 'roberta', 'ensemble'.
    checkpoints_dir : str
        The path to the directory containing the model checkpoint.
    DEVICE : torch.device
        The device on which the model should be loaded.

    Returns
    -------
    model : torch.nn.Module
        The loaded model.
    glove_vocab : dict or None
        The vocabulary of GloVe vectors, if the model is an LSTM.
    tokenizer : transformers.PreTrainedTokenizer or None
        The tokenizer for the model, if the model is a Roberta.

    Raises
    ------
    ValueError
        If the model name is not one of 'lstm', 'roberta', 'ensemble'.
    """
    glove_vocab, tokenizer = None, None
    if model_name == 'lstm':
        glove_vocab, glove_embeddings = load_glove_vocab('data/glove.6B/glove.6B.100d.txt')
        model = ToxicityClassifierLSTM(
            embedding_matrix=glove_embeddings,
            hidden_dim=128,
            output_dim=7,
            num_layers=2,
        )
        model.load_state_dict(torch.load(f'{checkpoints_dir}/LSTM.pth', weights_only=False, map_location=DEVICE))
        model = model.to(DEVICE)
    elif model_name == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForSequenceClassification.from_pretrained(checkpoints_dir)
        model = model.to(DEVICE)
    elif model_name == 'ensemble':
        model = EnsembleClassifier(checkpoints_dir, DEVICE)
    else:
        raise ValueError('Model name given has to be one of lstm, roberta or ensemble')
    
    return model, glove_vocab, tokenizer

def sigmoid(x):
    """
    Compute the sigmoid of a given input x.

    Args:
        x (float or np.ndarray): The input value(s) to compute the sigmoid of.

    Returns:
        float or np.ndarray: The sigmoid of the input value(s).
    """
    return 1 / (1 + np.exp(-x))


def predict_LSTM(sentence, model, DEVICE, vocab):
    """
    Predict toxicity score of a given sentence using the LSTM model.

    Args:
        sentence (str): The sentence to classify.
        model (nn.Module): The LSTM model to use for prediction.
        DEVICE (torch.device): The device to run the model on.
        vocab (dict): The vocabulary of words to their indices.

    Returns:
        float: The predicted toxicity score of the sentence.
    """
    model = model.to(DEVICE)

    dataset = JigsawDataset([], [], vocab, only_predict=True)
    
    tokens = dataset._preprocess(sentence)
    if len(tokens) < MAX_WORD_LENGTH:
        tokens += ['<pad>'] * (MAX_WORD_LENGTH - len(tokens))
    else:
        tokens = tokens[:MAX_WORD_LENGTH]

    input_tokens = torch.tensor([vocab[token] for token in tokens], dtype=torch.long).to(DEVICE)
    predictions = model(input_tokens.unsqueeze(0)).squeeze(1)
    predictions = torch.tensor([pred[0] for pred in predictions]).to(DEVICE)
            
    return sigmoid(predictions[0].detach().cpu().numpy())


def predict_roberta(sentence, model, DEVICE, tokenizer):
    """
    Predict the toxicity score of a sentence using the RoBERTa model.

    Args:
        sentence (str): The sentence to classify.
        model (nn.Module): The RoBERTa model used for prediction.
        DEVICE (torch.device): The device to run the model on (CPU or GPU).
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to preprocess the sentence.

    Returns:
        float: The predicted toxicity score of the sentence.
    """
    model = model.to(DEVICE)
    
    sentence = clean_text_for_roberta(sentence)
    tokens = tokenizer([sentence], truncation=True, padding=True, max_length=200, return_tensors='pt')
    input_tokens = tokens['input_ids']
    attention_mask = tokens['attention_mask']
    
    input_tokens, attention_mask = input_tokens.to(DEVICE), attention_mask.to(DEVICE)
    predictions = model(input_tokens, attention_mask=attention_mask).logits.squeeze(0)

    return predictions[0].detach().cpu().numpy()


def predict(sentence, model_name, model, DEVICE, vocab=None, tokenizer=None):
    """
    Predicts the toxicity of a sentence based on the model name, model, device, vocab and tokenizer.

    Parameters
    ----------
    sentence : str
        The sentence to predict the toxicity of
    model_name : str
        The name of the model, either 'lstm', 'roberta', or 'ensemble'
    model : nn.Module
        The PyTorch model to use for prediction
    DEVICE : torch.device
        The device to use for prediction, either cpu or cuda
    vocab : dict
        The vocabulary to use for prediction, only required for LSTM model
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer to use for prediction, only required for Roberta model

    Returns
    -------
    tuple
        A tuple of two elements, the first is the toxicity level, the second is the confidence of the model
    """
    if model_name == 'lstm':
        result = predict_LSTM(sentence, model, DEVICE, vocab)
    elif model_name == 'roberta':
        result = predict_roberta(sentence, model, DEVICE, tokenizer)
    elif model_name == 'ensemble':
        result = model(sentence)
    else:
        raise ValueError('Model name given has to be one of lstm, roberta or ensemble')
    
    if result > 0.98:
        return "I can't believe you tried that", result
    if result > 0.8:
        return 'Severely Toxic', result
    if result > 0.5:
        return 'Toxic', result
    if result > 0.2:
        return "Not too bad, but ummmmmm", result

    return "Not toxic", result
