import torch.nn as nn


class ANN(nn.Module):
    "Feed-forward artificial neural network."
    def __init__(self, input_size, hidden_size, output_size, dropout_prob):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x = self.fc1(x)
        x = self.elu(x)
        x = self.fc2(x)
        x = self.elu(x)
        x = self.dropout(x) 
        x = self.fc3(x)
        x = self.sigmoid(x)        
        
        return x