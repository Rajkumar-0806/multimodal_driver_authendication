# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Gait LSTM Encoder
# -------------------------
class GaitLSTM(nn.Module):
    """
    Input: batch x seq_len x feat_dim (e.g., 150 x 6)
    Output: embedding vector (batch x out_dim)
    """
    def __init__(self, in_dim=6, hidden_dim=128, num_layers=2, out_dim=128, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: (B, T, in_dim)
        out, (hn, cn) = self.lstm(x)         # hn: (num_layers, B, hidden_dim)
        last = hn[-1]                       # (B, hidden_dim)
        emb = self.proj(last)               # (B, out_dim)
        return emb

# -------------------------
# Voice LSTM Encoder (MFCC -> LSTM)
# -------------------------
class VoiceLSTM(nn.Module):
    """
    Input: batch x T_v x F  (e.g., mfcc frames)
    Output: embedding vector (batch x out_dim)
    """
    def __init__(self, in_dim=39, hidden_dim=128, num_layers=2, out_dim=128, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: (B, T_v, F)
        out, (hn, cn) = self.lstm(x)
        last = hn[-1]
        emb = self.proj(last)
        return emb

# -------------------------
# Face CNN Encoder (Lightweight ResNet-like)
# -------------------------
class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(identity)
        out += identity
        return F.relu(out)

class FaceCNN(nn.Module):
    """
    Input: batch x 3 x H x W (e.g., 3x160x160)
    Output: embedding vector (batch x out_dim)
    """
    def __init__(self, out_dim=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 7, 2, 3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1)
        )
        self.layer1 = self._make_layer(32, 64, stride=2)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, out_dim),
            nn.ReLU(),
        )

    def _make_layer(self, in_ch, out_ch, stride=1):
        return nn.Sequential(
            BasicBlock(in_ch, out_ch, stride=stride),
            BasicBlock(out_ch, out_ch, stride=1)
        )

    def forward(self, x):
        # normalize expecting x in [0,1]
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(out)
        emb = self.fc(out)
        return emb

# -------------------------
# Fusion Head (concat embeddings -> classifier)
# -------------------------
class FusionHead(nn.Module):
    """
    Input: concatenated embeddings from the three encoders
    Output: logits (batch x n_classes)
    """
    def __init__(self, emb_dim=128, num_modalities=3, hidden=256, n_classes=20, dropout=0.3):
        super().__init__()
        in_dim = emb_dim * num_modalities
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes)
        )

    def forward(self, g_emb, f_emb, v_emb):
        x = torch.cat([g_emb, f_emb, v_emb], dim=1)
        logits = self.mlp(x)
        return logits
