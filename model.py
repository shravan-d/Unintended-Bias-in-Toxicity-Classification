import torch
import torch.nn as nn
from torch.nn import functional as F

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
