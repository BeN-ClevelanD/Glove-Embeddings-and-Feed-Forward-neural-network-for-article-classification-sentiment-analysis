import optuna
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from classifier_pipeline import (
    load_and_normalise,
    build_dataset,
    train_classifier,
    FFNNClassifier_tunable,
    load_embeddings
)

def normalize_word_vectors(word_to_vec):
    normalized = {}
    for word, vec in word_to_vec.items():
        norm = np.linalg.norm(vec)
        normalized[word] = vec / norm if norm > 0 else vec
    return normalized

EMBEDDINGS_PATH = "glove_embeddings.txt"
RAW_DATA_PATH = "ag_news_csv/train.csv"
word_to_vec = load_embeddings(EMBEDDINGS_PATH)
word_to_vec = normalize_word_vectors(word_to_vec)
data = load_and_normalise(RAW_DATA_PATH)
train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)
EMBEDDING_DIM = len(next(iter(word_to_vec.values())))


def objective(trial):
    n_layers = trial.suggest_int("n_layers", 2, 3)
    h1 = trial.suggest_int("hidden_dim1", 64, 512, step=64)
    h2 = trial.suggest_int("hidden_dim2", 64, 512, step=64)
    hidden_dims = [h1, h2]
    if n_layers == 3:
        h3 = trial.suggest_int("hidden_dim3", 64, 512, step=64)
        hidden_dims.append(h3)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5, step=0.1)
    lr           = trial.suggest_loguniform("lr", 1e-4, 1e-1)
    batch_size   = trial.suggest_categorical("batch_size", [32, 64, 128])
    X_train, y_train = build_dataset(train_data, word_to_vec, EMBEDDING_DIM)
    X_val,   y_val   = build_dataset(val_data,   word_to_vec, EMBEDDING_DIM)
    model = FFNNClassifier_tunable(
        input_dim=EMBEDDING_DIM,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        num_classes=4
    )
    model = train_classifier(
        model,
        X_train, y_train,
        X_val,   y_val,
        epochs=15,           
        lr=lr,
        batch_size=batch_size,
        loss_curve_path=None
    )
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_val, dtype=torch.float32))
        preds  = logits.argmax(dim=1).cpu().numpy()
    return (preds == y_val).mean()

def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=45)
    print("Best hyperparameters:", study.best_params)

if __name__ == "__main__":
    main()
