import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from dataset import JigsawDataset, load_glove_vocab, MAX_WORD_LENGTH

class ToxicityClassifierLSTM(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, output_dim, num_layers, dropout=None):
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


def loadModel(model_name, checkpoints_dir, DEVICE):
    glove_vocab = None
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
    

    return model, glove_vocab

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def predictLSTM(sentence, model, DEVICE, vocab):
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



def predict(sentence, model_name, model, DEVICE, vocab=None):
    if model_name == 'lstm':
        result = predictLSTM(sentence, model, DEVICE, vocab)
    
    if result > 0.98:
        return "I can't believe you tried that", result
    if result > 0.8:
        return 'Severely Toxic', result
    if result > 0.4:
        return 'Toxic', result
    if result > 0.2:
        return "Not too bad, but ummmmmm", result

    return "Not toxic", result