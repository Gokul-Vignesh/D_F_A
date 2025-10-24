# ================================
# 1. Import Libraries
# ================================
import os
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ================================
# 2. Load Metadata
# ================================
# You should first generate a CSV like this:
# file_id,file_path,label
# DF_E_2000011,ASVspoof2021_DF_eval/flac/DF_E_2000011.flac,spoof
# DF_E_2000053,ASVspoof2021_DF_eval/flac/DF_E_2000053.flac,bonafide

CSV_PATH = "asvspoof2021_df_eval.csv"
df = pd.read_csv(CSV_PATH)

# Map labels to numbers: bonafide=0, spoof=1
label_map = {"bonafide": 0, "spoof": 1}
df["label_idx"] = df["label"].map(label_map)

# ================================
# 3. Feature Extraction Function
# ================================
def extract_features(file_path, sr=16000, n_mels=64, max_len=400):
    """
    Convert audio file into log-mel spectrogram.
    - sr: target sample rate
    - n_mels: number of mel bands
    - max_len: time steps (pad/truncate for uniform size)
    """
    y, sr = librosa.load(file_path, sr=sr)  # load audio
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)  # convert to log scale
    
    # Pad or truncate to fixed length
    if log_mel.shape[1] < max_len:
        pad_width = max_len - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0,0),(0,pad_width)), mode='constant')
    else:
        log_mel = log_mel[:, :max_len]
    
    return log_mel

# ================================
# 4. Torch Dataset
# ================================
class AudioDataset(Dataset):
    def __init__(self, df):
        self.df = df
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        features = extract_features(row["file_path"])  # shape: (64, 400)
        features = torch.tensor(features, dtype=torch.float).unsqueeze(0)  # (1, 64, 400)
        label = torch.tensor(row["label_idx"], dtype=torch.long)
        return features, label

# Train-test split
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label_idx"], random_state=42)
train_dataset = AudioDataset(train_df)
test_dataset = AudioDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# ================================
# 5. CNN Model
# ================================
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),   # output shape ~ (16, 32, 200)

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)    # output shape ~ (32, 16, 100)
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 16 * 100, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # output: bonafide vs spoof
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel().to(device)

# ================================
# 6. Training
# ================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 5
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader):.4f}")

# ================================
# 7. Evaluation
# ================================
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for features, labels in test_loader:
        features, labels = features.to(device), labels.to(device)
        outputs = model(features)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\nClassification Report:\n", classification_report(all_labels, all_preds, target_names=["bonafide", "spoof"]))

# ================================
# 8. Prediction on New File
# ================================
def predict(file_path):
    model.eval()
    features = extract_features(file_path)
    features = torch.tensor(features, dtype=torch.float).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(features)
        pred = torch.argmax(output, dim=1).item()
    return "bonafide" if pred == 0 else "spoof"

# Example usage
print("Prediction:", predict(test_df.iloc[0]["file_path"]))
