import torch
import torch.nn as nn


class TemporalModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True
        )

        self.fc = nn.Linear(256,1)


    def forward(self,x):

        out,_ = self.lstm(x)

        scores = self.fc(out)

        scores = scores.squeeze(-1)

        return scores
