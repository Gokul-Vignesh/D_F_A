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
from sklearn.utils.class_weight import compute_class_weight

# ================================
# 2. Paths & Dataset Loading
# ================================
AUDIO_DIR = "ASVspoof2021_DF_eval/flac"
CSV_PATH = "ASVspoof2021_DF_labels.csv"  # your metadata CSV

df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip().str.upper()

if "ID" not in df.columns or "LABEL" not in df.columns:
    raise ValueError("CSV must have headers: ID and LABEL")

label_map = {"bonafide": 0, "spoof": 1}
df["label_idx"] = df["LABEL"].map(label_map)
df["file_path"] = df["ID"].apply(lambda x: os.path.join(AUDIO_DIR, f"{x}.flac"))
df = df[df["file_path"].apply(os.path.exists)]

print(f"✅ Valid audio files found: {len(df)}")

# ================================
# 3. Balance Dataset
# ================================
min_count = df["LABEL"].value_counts().min()
df_balanced = pd.concat([
    df[df["LABEL"] == "bonafide"].sample(min_count, random_state=42),
    df[df["LABEL"] == "spoof"].sample(min_count, random_state=42)
])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
print("✅ Dataset balanced.")

# ================================
# 4. Feature Extraction
# ================================
def extract_features(file_path, sr=16000, n_mels=64, max_len=400):
    try:
        y, sr = librosa.load(file_path, sr=sr)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)

        if log_mel.shape[1] < max_len:
            pad_width = max_len - log_mel.shape[1]
            log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode="constant")
        else:
            log_mel = log_mel[:, :max_len]
        return log_mel
    except Exception as e:
        print(f"⚠️ Could not process file {file_path}: {e}")
        return np.zeros((n_mels, max_len), dtype=np.float32)

# ================================
# 5. Dataset Class
# ================================
class AudioDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        features = extract_features(row["file_path"])
        features = torch.tensor(features, dtype=torch.float).unsqueeze(0)
        label = torch.tensor(row["label_idx"], dtype=torch.long)
        return features, label

# ================================
# 6. Train/Test Split
# ================================
train_df, test_df = train_test_split(
    df_balanced, test_size=0.2, stratify=df_balanced["label_idx"], random_state=42
)

train_dataset = AudioDataset(train_df)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# ================================
# 7. Model Definition
# ================================
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 16 * 100, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ================================
# 8. Train Model
# ================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel().to(device)

classes = np.unique(df_balanced["label_idx"])
weights = compute_class_weight("balanced", classes=classes, y=df_balanced["label_idx"])
weights = torch.tensor(weights, dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)
EPOCHS = 10

print("\n🚀 Training Started...")
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
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {running_loss/len(train_loader):.4f}")
print("\n✅ Training Complete!")

# Optionally save the trained model
torch.save(model.state_dict(), "deepfake_detector.pth")
print("💾 Model saved as deepfake_detector.pth")

# ================================
# 9. User Audio Prediction
# ================================
def predict_user_audio(file_path):
    print(f"\n🔍 Analyzing: {file_path}")
    if not os.path.exists(file_path):
        print("❌ File not found.")
        return None

    features = extract_features(file_path)
    features = torch.tensor(features, dtype=torch.float).unsqueeze(0).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(features)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred].item() * 100

    result = "🧑 Bonafide Human Voice" if pred == 0 else "🤖 AI/Deepfake Voice"
    print(f"✅ Prediction: {result} ({confidence:.2f}% confidence)")
    return result

# ================================
# 10. Interactive User Input
# ================================
print("\n🎧 DeepFake Voice Detection System Ready!")
while True:
    user_input = input("\nEnter an audio file path (.flac or .wav) to analyze, or 'q' to quit:\n> ").strip()
    if user_input.lower() == "q":
        print("👋 Exiting.")
        break
    predict_user_audio(user_input)
# ================================
# 11. Save Model (for reuse in Streamlit)
# ================================
MODEL_PATH = "saved_model.pth"
torch.save(model.state_dict(), MODEL_PATH)
print(f"\n💾 Model saved to {MODEL_PATH}")

# ================================
# 12. Load Model (for Streamlit)
# ================================
def load_model(model_path="saved_model.pth"):
    model = CNNModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model
