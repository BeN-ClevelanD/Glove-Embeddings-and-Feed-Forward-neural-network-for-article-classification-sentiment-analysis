import os
import csv
import re
import unicodedata
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime

def remove_accents(content):
    return ''.join(letter for letter in unicodedata.normalize('NFKD', content) if not unicodedata.combining(letter))

def strip_non_basic_ascii(text):
    return ''.join(c for c in text if c in 'abcdefghijklmnopqrstuvwxyz 0')

def clean_text(text):
    result = text.lower()
    result = re.sub(r'[-]', ' ', result)
    result = re.sub(r'<[^>]+>', ' ', result)
    result = re.sub(r'&[a-zA-Z0-9#]+;', ' ', result)
    result = re.sub(r'\b(quot|nbsp|39|amp|lt|gt)\b', ' ', result)
    result = re.sub(r'\b\w*(http|www|href|html|com|net|asp|php)\w*\b', '', result)
    result = re.sub(r'\b(reuters|usatodaycom|forbescom|afp|ap|cnn|techweb|maccentral|spacecom)\b', '', result)
    result = re.sub(r'\d+', '', result)
    result = re.sub(r'[^\w\s]', ' ', result)
    result = re.sub(r'\s+', ' ', result)
    result = remove_accents(result)
    result = strip_non_basic_ascii(result)
    return result.strip()

def load_and_normalise(filepath):
    cleaned_data = []
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) >= 3:
                label = int(row[0]) - 1 
                description = row[2]
                cleaned_desc = clean_text(description)
                cleaned_data.append((label, cleaned_desc))
    return cleaned_data

def load_embeddings(embedding_path):
    word_to_vec = {}
    with open(embedding_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            word_to_vec[word] = vec
    return word_to_vec

def sentence_to_embedding(sentence, word_to_vec, embedding_dim=100):
    words = sentence.split()
    vectors = [word_to_vec[word] for word in words if word in word_to_vec]
    if not vectors:
        return np.zeros(embedding_dim, dtype=np.float32)
    return np.mean(vectors, axis=0)

def build_dataset(data, word_to_vec, embedding_dim=300):
    X = []
    y = []
    for label, sentence in data:
        emb = sentence_to_embedding(sentence, word_to_vec, embedding_dim)
        X.append(emb)
        y.append(label)
    return np.stack(X), np.array(y)

import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

def train_classifier(model, train_X, train_y, val_X, val_y,
                     epochs=15, lr=0.0032364051067221814, batch_size=128,
                     loss_curve_path="loss_curve.png"):
    device = torch.device('cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    train_dataset = TensorDataset(
        torch.tensor(train_X, dtype=torch.float32),
        torch.tensor(train_y, dtype=torch.long)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_X = torch.tensor(val_X, dtype=torch.float32).to(device)
    val_y = torch.tensor(val_y, dtype=torch.long).to(device)
    train_losses, val_losses = [], []
    for epoch in range(1, epochs+1):
        model.train()
        total_train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        model.eval()
        with torch.no_grad():
            val_logits = model(val_X)
            loss_val = loss_fn(val_logits, val_y).item()
            val_losses.append(loss_val)
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == val_y).float().mean().item()
        print(f"Epoch {epoch}/{epochs} — "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {loss_val:.4f}, "
              f"Val Acc: {val_acc:.4f}")

    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses,   label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.legend()
    plt.tight_layout()
    if loss_curve_path:
        plt.savefig(loss_curve_path)
    plt.close()
    return model

def evaluate_classifier(model, test_X, test_y, results_path="test_results.txt"):
    device = torch.device('cpu')
    model.eval()
    test_X = torch.tensor(test_X, dtype=torch.float32).to(device)
    test_y = torch.tensor(test_y, dtype=torch.long).to(device)
    with torch.no_grad():
        preds = model(test_X).argmax(dim=1).cpu().numpy()
        true = test_y.cpu().numpy()
    acc = accuracy_score(true, preds)
    cm = confusion_matrix(true, preds)
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write(f"Test Accuracy: {acc:.4f}\n")
        f.write("Confusion Matrix:\n")
        for row in cm:
            f.write(" ".join(str(val) for val in row) + "\n")
    print(f"Test Accuracy: {acc:.4f}")
    print(f" Results written to '{results_path}'")
    return cm

def plot_confusion_matrix(cm, labels, output_path="confusion_matrix.png"):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f" Confusion matrix saved to '{output_path}'")

class FFNNClassifier(nn.Module):
    def __init__(self, input_dim=500, hidden_dim1=320, hidden_dim2=128, hidden_dim3=320, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim3, num_classes)
        )
    def forward(self, x):
        return self.net(x)
    
class FFNNClassifier_tunable(nn.Module):
    def __init__(
        self,
        input_dim: int = 500,
        hidden_dims: list[int] = [400, 300, 150],
        dropout_rate: float = 0.3,
        num_classes: int = 4
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
