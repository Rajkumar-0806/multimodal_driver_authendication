# train.py
import argparse
import os
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

from dataset import MultimodalDataset
from models import GaitLSTM, FaceCNN, VoiceLSTM, FusionHead

def compute_eer(y_true, y_score):
    # y_true: {0,1} where 1=pos, 0=neg; y_score: probability for positive
    fpr, tpr, thresh = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    # EER is where FPR ~= FNR
    idx = np.nanargmin(np.abs(fpr - fnr))
    return (fpr[idx] + fnr[idx]) / 2.0, thresh[idx]

def evaluate(model_components, loader, device):
    gait_enc, face_enc, voice_enc, fusion = model_components
    gait_enc.eval(); face_enc.eval(); voice_enc.eval(); fusion.eval()
    y_true = []
    y_prob = []
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            gait = batch['gait'].to(device)           # (B,T,6)
            face = batch['face'].to(device)           # (B,3,160,160)
            voice = batch['voice'].to(device)         # (B,16000) or (B,Tv,feat)
            labels = batch['label'].to(device)
            # if voice is raw waveform, map to shape (B,16000) and pass to voice encoder expecting MFCC or raw linear
            # Here we assume voice encoder expects fixed-length vector (train_stub design)
            g_emb = gait_enc(gait)
            f_emb = face_enc(face)
            # if voice is 1D waveform, reshape for voice encoder stub (simple fc)
            if voice.dim() == 2:
                # voice encoder in models expects (B, T, F) OR (B, len) depending on implementation
                # In our VoiceLSTM we expect (B, T, F). For simplicity, if 1D we reshape to (B, 16000)
                v_emb = voice_enc(voice)
            else:
                v_emb = voice_enc(voice)
            logits = fusion(g_emb, f_emb, v_emb)
            probs = torch.softmax(logits, dim=1)
            # predict top class (for closed-set)
            preds = torch.argmax(probs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            # For EER, treat true class as positive: compute probability assigned to correct class
            for i in range(labels.size(0)):
                lab = labels[i].item()
                prob = probs[i, lab].item()
                y_true.append(1)   # positive for genuine sample (we use pairs approach usually; here demo)
                y_prob.append(prob)
    acc = correct / total if total > 0 else 0.0
    # EER computed on the probs array (synthetic; in practice compute on genuine vs impostor trials)
    eer, thr = compute_eer(y_true, y_prob)
    return acc, eer

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = MultimodalDataset(args.data_dir, seq_len_gait=150, voice_len=16000, mfcc=False)
    n = len(ds)
    n_train = int(n * 0.7)
    n_val = n - n_train
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Instantiate models
    gait_enc = GaitLSTM(in_dim=6, hidden_dim=128, out_dim=128).to(device)
    face_enc = FaceCNN(out_dim=128).to(device)
    # Simple voice encoder: if using raw waveform, adapt model; here use a small MLP to map 16000 -> 128
    voice_enc = nn.Sequential(nn.Flatten(), nn.Linear(16000, 512), nn.ReLU(), nn.Linear(512,128)).to(device)
    fusion = FusionHead(emb_dim=128, num_modalities=3, hidden=256, n_classes=len(ds.meta['participant'].unique())).to(device)

    # optimizer and loss
    opt = torch.optim.Adam(list(gait_enc.parameters()) + list(face_enc.parameters()) + list(voice_enc.parameters()) + list(fusion.parameters()), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        gait_enc.train(); face_enc.train(); voice_enc.train(); fusion.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            gait = batch['gait'].to(device)         # (B,T,6)
            face = batch['face'].to(device)         # (B,3,160,160)
            voice = batch['voice'].to(device)       # (B,16000)
            labels = batch['label'].to(device)

            g_emb = gait_enc(gait)
            f_emb = face_enc(face)
            v_emb = voice_enc(voice)
            logits = fusion(g_emb, f_emb, v_emb)
            loss = loss_fn(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=epoch_loss/(pbar.n+1))

        # evaluate
        val_acc, val_eer = evaluate((gait_enc, face_enc, voice_enc, fusion), val_loader, device)
        print(f"Epoch {epoch+1}: Val Acc: {val_acc:.4f}, Val EER: {val_eer:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # save checkpoint
            ckpt = {
                'gait_enc': gait_enc.state_dict(),
                'face_enc': face_enc.state_dict(),
                'voice_enc': voice_enc.state_dict(),
                'fusion': fusion.state_dict()
            }
            torch.save(ckpt, os.path.join(args.out_dir, "best_model.pth"))
    print("Training completed. Best val acc:", best_val_acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./multimodal_synthetic_dataset")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out_dir", type=str, default="./checkpoints")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    train(args)
