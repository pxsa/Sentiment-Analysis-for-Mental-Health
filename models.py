import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==================================================
# ==================================================

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, 
                 n_layers=2, bidirectional=False, dropout=0.5, 
                 pretrained_embeddings=None, freeze_embeddings=False):
        super(LSTMClassifier, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        # Embedding layer
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Fully connected layer
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, output_dim)

    
    def forward(self, text):
        # text: [batch_size, seq_len]
        embedded = self.dropout(self.embedding(text))
        # embedded: [batch_size, seq_len, embedding_dim]
        
        output, (hidden, cell) = self.lstm(embedded)
        # output: [batch_size, seq_len, hidden_dim * num_directions]
        # hidden: [n_layers * num_directions, batch_size, hidden_dim]
        
        if self.bidirectional:
            # Concatenate the final forward and backward hidden states
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        
        # hidden: [batch_size, hidden_dim * num_directions]
        hidden = self.dropout(hidden)
        output = self.fc(hidden)
        
        return output
    

# ==================================================
# ==================================================

class xLSTMCell(nn.Module):
    """
    Extended LSTM Cell with exponential gating and modifed memory mixing
    Based on xLSTM paper concepts
    """
    def __init__(self, input_size, hidden_size):
        super(xLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Input gate
        self.W_ii = nn.Linear(input_size, hidden_size, bias=True)
        self.W_hi = nn.Linear(hidden_size, hidden_size, bias=False)

        # Forget gate
        self.W_if = nn.Linear(input_size, hidden_size, bias=True)
        self.W_hf = nn.Linear(hidden_size, hidden_size, bias=False)

        # Cell gate
        self.W_ig = nn.Linear(input_size, hidden_size, bias=True)
        self.W_hg = nn.Linear(hidden_size, hidden_size, bias=False)

        # Output gate
        self.W_io = nn.Linear(input_size, hidden_size, bias=True)
        self.W_ho = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Normalization layers
        self.layer_norm_c = nn.LayerNorm(hidden_size)
        self.layer_norm_h = nn.LayerNorm(hidden_size)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-std, std)


    def forward(self, x, hidden):
        h_prev, c_prev = hidden
        
        # Input gate
        i = torch.sigmoid(self.W_ii(x) + self.W_hi(h_prev))
        
        # Forget gate
        f = torch.sigmoid(self.W_if(x) + self.W_hf(h_prev))
        
        # Cell gate
        g = torch.tanh(self.W_ig(x) + self.W_hg(h_prev))
        
        # Output gate
        o = torch.sigmoid(self.W_io(x) + self.W_ho(h_prev))
        
        # Exponential gate (key innovation in xLSTM)
        e = torch.exp(torch.tanh(self.W_ie(x) + self.W_he(h_prev)))
        
        # Update cell state with exponential gating
        c = f * c_prev + i * g * e
        c = self.layer_norm_c(c)
        
        # Update hidden state
        h = o * torch.tanh(c)
        h = self.layer_norm_h(h)
        
        return h, c

# ==================================================
# ==================================================

class xLSTM(nn.Module):
    """Extended LSTM layer"""
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, dropout=0.0):
        super(xLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        
        # Create xLSTM cells
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            self.cells.append(xLSTMCell(layer_input_size, hidden_size))
        
        if dropout > 0 and num_layers > 1:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = None
    
    def forward(self, x, hidden=None):
        # x: [batch_size, seq_len, input_size] if batch_first
        if self.batch_first:
            batch_size, seq_len, _ = x.size()
        else:
            seq_len, batch_size, _ = x.size()
            x = x.transpose(0, 1)
        
        # Initialize hidden states if not provided
        if hidden is None:
            h = [torch.zeros(batch_size, self.hidden_size, device=x.device) 
                 for _ in range(self.num_layers)]
            c = [torch.zeros(batch_size, self.hidden_size, device=x.device) 
                 for _ in range(self.num_layers)]
        else:
            h, c = hidden
            h = list(h)
            c = list(c)
        
        # Process sequence
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            for layer in range(self.num_layers):
                h[layer], c[layer] = self.cells[layer](x_t, (h[layer], c[layer]))
                x_t = h[layer]
                
                if self.dropout_layer is not None and layer < self.num_layers - 1:
                    x_t = self.dropout_layer(x_t)
            
            outputs.append(h[-1].unsqueeze(1))
        
        output = torch.cat(outputs, dim=1)
        h_final = torch.stack(h)
        c_final = torch.stack(c)
        
        return output, (h_final, c_final)

# ==================================================
# ==================================================

class xLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 n_layers=2, dropout=0.5, pretrained_embeddings=None, 
                 freeze_embeddings=False):
        super(xLSTMClassifier, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False
        
        # xLSTM layer
        self.xlstm = xLSTM(embedding_dim, hidden_dim, n_layers, 
                          batch_first=True, dropout=dropout)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim, 1)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):
        # text: [batch_size, seq_len]
        embedded = self.dropout(self.embedding(text))
        # embedded: [batch_size, seq_len, embedding_dim]
        
        output, (hidden, cell) = self.xlstm(embedded)
        # output: [batch_size, seq_len, hidden_dim]
        
        # Apply attention
        attention_weights = torch.softmax(self.attention(output), dim=1)
        # attention_weights: [batch_size, seq_len, 1]
        
        attended = torch.sum(attention_weights * output, dim=1)
        # attended: [batch_size, hidden_dim]
        
        attended = self.dropout(attended)
        output = self.fc(attended)
        
        return output


# ==================================================
# ==================================================

class xBiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 n_layers=2, dropout=0.5, pretrained_embeddings=None,
                 freeze_embeddings=False):
        super(xBiLSTMClassifier, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False
        
        # Forward and backward xLSTM layers
        self.xlstm_forward = xLSTM(embedding_dim, hidden_dim, n_layers,
                                   batch_first=True, dropout=dropout)
        self.xlstm_backward = xLSTM(embedding_dim, hidden_dim, n_layers,
                                    batch_first=True, dropout=dropout)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, text):
        # text: [batch_size, seq_len]
        embedded = self.dropout(self.embedding(text))
        # embedded: [batch_size, seq_len, embedding_dim]
        
        # Forward pass
        output_forward, _ = self.xlstm_forward(embedded)
        # output_forward: [batch_size, seq_len, hidden_dim]
        
        # Backward pass (reverse the sequence)
        embedded_backward = torch.flip(embedded, [1])
        output_backward, _ = self.xlstm_backward(embedded_backward)
        output_backward = torch.flip(output_backward, [1])
        # output_backward: [batch_size, seq_len, hidden_dim]
        
        # Concatenate forward and backward outputs
        output = torch.cat([output_forward, output_backward], dim=2)
        # output: [batch_size, seq_len, hidden_dim * 2]
        
        # Apply attention
        attention_weights = torch.softmax(self.attention(output), dim=1)
        # attention_weights: [batch_size, seq_len, 1]
        
        attended = torch.sum(attention_weights * output, dim=1)
        # attended: [batch_size, hidden_dim * 2]
        
        attended = self.dropout(attended)
        output = self.fc(attended)
        
        return output

# ==================================================
# ==================================================
