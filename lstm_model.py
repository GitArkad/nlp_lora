import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):


    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128, num_layers=1,
                 dropout=0.3, bidirectional=False, num_classes=2):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * (2 if bidirectional else 1), 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)             # один логит (как в твоей модели)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)

        out, (h, c) = self.lstm(x)
        h_last = h[-1]  # последний слой LSTM

        logit = self.fc(h_last)  # [batch, 1]

        # Конвертируем 1 логит → 2 логита для softmax:
        # класс 0 = -logit, класс 1 = +logit
        logits = torch.cat([-logit, logit], dim=1)  # [batch, 2]

        return logits


def accuracy(preds, y):
    preds = torch.sigmoid(preds)
    preds = (preds >= 0.5).float()
    return (preds == y).float().mean()