from oldnormalise_classification_data import load_and_normalise
import csv
import os
from text_normaliser import clean_line
import os
import torch
from classifier_pipeline import *
from glove_utils import (
    load_glove_embeddings,
    print_most_similar_for_selected_groups_glove,
    visualize_selected_word_groups_glove,
    visualize_3d_selected_words_glov
)
import random
random.seed(0)
torch.manual_seed(0)
from glove_utils import build_vocab_and_cooccurrence, train_glove, save_glove_embeddings



def load_and_normalise_csv(input_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:

        reader = csv.reader(infile)
        for row in reader:
            if len(row) < 3:
                continue 
            title, description = row[1], row[2]
            normalised = clean_line(description)
            if normalised:
                outfile.write(normalised + "\n")

def implement_glove():
    corpus_path = "data/normalised/train.txt"
    embedding_dim = 500
    window_size = 10
    x_max = 100
    alpha = 0.75
    epochs = 50
    lr = 0.05
    cooccurrence, word_to_id, id_to_word = build_vocab_and_cooccurrence(corpus_path, window_size=window_size)
    model = train_glove(cooccurrence, vocab_size=len(word_to_id), embedding_dim=embedding_dim,
                        x_max=x_max, alpha=alpha, epochs=epochs, lr=lr)
    with torch.no_grad():
        norms = (model.w_embeddings.weight.data + model.w_context.weight.data).norm(dim=1, keepdim=True)
        model.w_embeddings.weight.data /= norms
        model.w_context.weight.data /= norms
    save_glove_embeddings(model, id_to_word, path="glove_embeddings.txt")

def analyze_glove_embeddings():
    glove_path = "glove_embeddings.txt"
    word_to_vec = load_glove_embeddings(glove_path)
    print_most_similar_for_selected_groups_glove(word_to_vec, output_path="glove_selected_similar.txt")
    visualize_selected_word_groups_glove(word_to_vec, output_path="glove_selected_groups.png")

def run_glove_visualisations():
    word_to_vec = load_glove_embeddings("glove_embeddings.txt")
    visualize_3d_selected_words_glov(word_to_vec)
  
def normalize_word_vectors(word_to_vec):
    normalized = {}
    for word, vec in word_to_vec.items():
        norm = np.linalg.norm(vec)
        if norm == 0:
            normalized[word] = vec
        else:
            normalized[word] = vec / norm
    return normalized

def run_classification(embeddingspath):
    word_to_vec = load_embeddings(embeddingspath)
    word_to_vec = normalize_word_vectors(word_to_vec)
    train_data = load_and_normalise("ag_news_csv/train.csv")
    test_data = load_and_normalise("ag_news_csv/test.csv")
    train_subset, val_subset = train_test_split(train_data, test_size=0.1, random_state=42)
    X_train, y_train = build_dataset(train_subset, word_to_vec)
    X_val, y_val = build_dataset(val_subset, word_to_vec)
    X_test, y_test = build_dataset(test_data, word_to_vec)
    model = FFNNClassifier(input_dim=500)
    model = train_classifier(model, X_train, y_train, X_val, y_val, epochs=15)
    cm = evaluate_classifier(model, X_test, y_test)
    plot_confusion_matrix(cm, labels=["World", "Sports", "Business", "Sci/Tech"])

if __name__ == "__main__":
    train_input = "ag_news_csv/train.csv"
    train_output = "data/normalised/train.txt"
    load_and_normalise_csv(train_input, train_output)
    embpath3 = "glove_embeddings.txt"
    if os.path.exists(embpath3):
        print(" Found existing embeddings.")
        word_vectors = {}
        with open(embpath3, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.strip().split()
                word = values[0]
                vector = torch.tensor([float(x) for x in values[1:]], dtype=torch.float)
                word_vectors[word] = vector
    else:
        implement_glove()
    analyze_glove_embeddings()
    run_glove_visualisations()
    run_classification(embeddingspath=embpath3) 
 


    
