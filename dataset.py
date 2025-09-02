# dataset.py
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import soundfile as sf
import torch

# optional: use librosa if you need MFCC extraction later
try:
    import torchaudio
    have_torchaudio = True
except Exception:
    have_torchaudio = False
    import librosa

def zscore(a):
    a = np.array(a, dtype=np.float32)
    mu = a.mean(axis=0, keepdims=True)
    sd = a.std(axis=0, keepdims=True) + 1e-8
    return (a - mu) / sd

class MultimodalDataset(Dataset):
    """
    Expects directory structure:
    data_dir/
      gait/*.csv
      face/*.png
      voice/*.wav
      metadata.csv  (columns: participant,gait,face,voice,label)
    """
    def __init__(self, data_dir, seq_len_gait=150, voice_len=16000, mfcc=False, mfcc_params=None, transform=None):
        self.data_dir = data_dir
        meta = pd.read_csv(os.path.join(data_dir, "metadata.csv"))
        self.meta = meta
        self.seq_len_gait = seq_len_gait
        self.voice_len = voice_len
        self.mfcc = mfcc
        self.mfcc_params = mfcc_params or {'n_mfcc':13, 'hop_length':512, 'n_fft':2048}
        self.transform = transform

    def __len__(self):
        return len(self.meta)

    def _load_gait(self, fname):
        path = os.path.join(self.data_dir, "gait", fname)
        df = pd.read_csv(path)
        arr = df[['ax','ay','az','gx','gy','gz']].values.astype(np.float32)
        arr = zscore(arr)
        # pad / truncate
        L = self.seq_len_gait
        if arr.shape[0] < L:
            pad = np.zeros((L - arr.shape[0], arr.shape[1]), dtype=np.float32)
            arr = np.vstack([arr, pad])
        else:
            arr = arr[:L]
        return arr  # (L,6)

    def _load_face(self, fname):
        path = os.path.join(self.data_dir, "face", fname)
        im = Image.open(path).convert("RGB")
        im = im.resize((160,160))
        arr = np.array(im).astype(np.float32) / 255.0  # HxWx3
        # transpose to CxHxW for torch
        arr = np.transpose(arr, (2,0,1))
        return arr

    def _load_voice(self, fname):
        path = os.path.join(self.data_dir, "voice", fname)
        sig, sr = sf.read(path)
        sig = sig.astype(np.float32)
        # normalize amplitude
        if sig.max() != 0:
            sig = sig / (np.max(np.abs(sig)) + 1e-9)
        # pad / truncate
        L = self.voice_len
        if len(sig) < L:
            sig = np.pad(sig, (0, L - len(sig)))
        else:
            sig = sig[:L]
        if self.mfcc:
            if have_torchaudio:
                mfcc = torchaudio.transforms.MFCC(sample_rate=sr, n_mfcc=self.mfcc_params['n_mfcc'],
                                                  melkwargs={'n_fft': self.mfcc_params['n_fft'],
                                                             'hop_length': self.mfcc_params['hop_length']})(torch.tensor(sig).float())
                # mfcc shape: (n_mfcc, time)
                mfcc = mfcc.transpose(0,1).numpy()  # (T, n_mfcc)
            else:
                mfcc = librosa.feature.mfcc(y=sig, sr=sr, n_mfcc=self.mfcc_params['n_mfcc'],
                                            n_fft=self.mfcc_params['n_fft'],
                                            hop_length=self.mfcc_params['hop_length']).T
            return mfcc.astype(np.float32)
        else:
            return sig.astype(np.float32)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        gait = self._load_gait(row['gait'])
        face = self._load_face(row['face'])
        voice = self._load_voice(row['voice'])
        label = int(row['label']) - 1  # zero-index
        sample = {
            'gait': torch.tensor(gait, dtype=torch.float32),
            'face': torch.tensor(face, dtype=torch.float32),
            'voice': torch.tensor(voice, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long)
        }
        return sample
