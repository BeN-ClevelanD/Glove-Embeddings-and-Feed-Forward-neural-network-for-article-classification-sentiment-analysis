import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter, defaultdict
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from umap import UMAP

def load_glove_embeddings(file_path):
    word_to_vec = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            word_to_vec[word] = vec
    return word_to_vec

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def build_vocab_and_cooccurrence(corpus_path, window_size=10, min_count=1):
    with open(corpus_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    tokens = [line.strip().split() for line in lines]
    flat_tokens = [token for line in tokens for token in line]
    print("window_size:", window_size)
    vocab_counter = Counter(flat_tokens)
    vocab = {word for word, count in vocab_counter.items() if count >= min_count}
    word_to_id = {word: i for i, word in enumerate(sorted(vocab))}
    id_to_word = {i: word for word, i in word_to_id.items()}
    cooccurrence = defaultdict(Counter)
    for sentence in tokens:
        indexed_sentence = [word_to_id[word] for word in sentence if word in word_to_id]
        for center_idx, center_word in enumerate(indexed_sentence):
            start = max(0, center_idx - window_size)
            end = min(len(indexed_sentence), center_idx + window_size + 1)
            for context_idx in range(start, end):
                if context_idx != center_idx:
                    context_word = indexed_sentence[context_idx]
                    distance = abs(center_idx - context_idx)
                    weight = 1.0 / distance
                    cooccurrence[center_word][context_word] += weight
    return cooccurrence, word_to_id, id_to_word

class GloVeModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.w_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.w_context = nn.Embedding(vocab_size, embedding_dim)
        self.b_w = nn.Embedding(vocab_size, 1)
        self.b_c = nn.Embedding(vocab_size, 1)
        nn.init.xavier_uniform_(self.w_embeddings.weight)
        nn.init.xavier_uniform_(self.w_context.weight)
        nn.init.zeros_(self.b_w.weight)
        nn.init.zeros_(self.b_c.weight)

    def forward(self, center_ids, context_ids):
        w = self.w_embeddings(center_ids)
        c = self.w_context(context_ids)
        bw = self.b_w(center_ids).squeeze()
        bc = self.b_c(context_ids).squeeze()
        dot = (w * c).sum(dim=1)
        return dot + bw + bc

def train_glove(cooccurrence, vocab_size, embedding_dim=500, x_max=100, alpha=0.75, epochs=50, lr=0.05, batch_size=4000):
    model = GloVeModel(vocab_size, embedding_dim)
    optimizer = optim.Adagrad(model.parameters(), lr=lr)
    data = []
    for center_id, contexts in cooccurrence.items():
        for context_id, x_ij in contexts.items():
            data.append((center_id, context_id, x_ij))
    for epoch in range(epochs):
        random.shuffle(data)
        total_loss = 0
        print(f"Epoch {epoch+1}/{epochs}")
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            center_ids = torch.tensor([x[0] for x in batch], dtype=torch.long)
            context_ids = torch.tensor([x[1] for x in batch], dtype=torch.long)
            Xij = torch.tensor([x[2] for x in batch], dtype=torch.float)
            fX = torch.where(Xij < x_max, (Xij / x_max) ** alpha, torch.ones_like(Xij))
            logX = torch.log(Xij)
            preds = model(center_ids, context_ids)
            loss = torch.mean(fX * (preds - logX) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    return model

def save_glove_embeddings(model, id_to_word, path="glove_embeddings.txt"):
    embeddings = model.w_embeddings.weight.data + model.w_context.weight.data
    with open(path, 'w', encoding='utf-8') as f:
        for idx, vec in enumerate(embeddings):
            word = id_to_word[idx]
            vec_str = ' '.join([f"{v:.4f}" for v in vec.tolist()])
            f.write(f"{word} {vec_str}\n")

def print_most_similar_for_selected_groups_glove(word_to_vec, output_path="glove_selected_similar.txt"):
    thematic_groups = {
        "business": ['oil', 'energy', 'market', 'stocks', 'dollar'],
        "countries": ['china', 'japan', 'germany', 'brazil', 'india'],
        "sports": ['football', 'cricket', 'golf', 'olympics', 'tennis'],
        "health": ['virus', 'cancer', 'health', 'doctor', 'vaccine'],
        "tech": ['computer', 'internet', 'software', 'google', 'microsoft']
    }
    output_lines = []
    for group_name, words in thematic_groups.items():
        output_lines.append(f"=== Similar words in category: {group_name} ===")
        for query_word in words:
            if query_word not in word_to_vec:
                output_lines.append(f"{query_word} not in vocabulary.\n")
                continue
            query_vec = word_to_vec[query_word]
            similarities = [
                (w, cosine_similarity(query_vec, vec))
                for w, vec in word_to_vec.items()
                if w != query_word
            ]
            similarities.sort(key=lambda x: x[1], reverse=True)
            output_lines.append(f"Top 5 words similar to '{query_word}':")
            for word, sim in similarities[:5]:
                dist = np.linalg.norm(query_vec - word_to_vec[word])
                output_lines.append(
                    f"{word}: similarity={sim:.4f}, distance from '{query_word}'={dist:.4f}"
                )
            output_lines.append("") 
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in output_lines:
            f.write(line + "\n")

def visualize_selected_word_groups_glove(word_to_vec, output_path="glove_selected_groups.png"):
    thematic_groups = {
        "business": ['oil', 'energy', 'market', 'stocks', 'dollar'],
        "countries": ['china', 'japan', 'germany', 'brazil', 'india'],
        "sports": ['football', 'cricket', 'golf', 'olympics', 'tennis'],
        "health": ['virus', 'cancer', 'health', 'doctor', 'vaccine'],
        "tech": ['computer', 'internet', 'software', 'google', 'microsoft']
    }
    words, vectors, colors = [], [], []
    color_map = {"business": "red", "countries": "blue", "sports": "green", "health": "purple", "tech": "orange"}
    for category, word_list in thematic_groups.items():
        for word in word_list:
            if word in word_to_vec:
                words.append(word)
                vectors.append(word_to_vec[word])
                colors.append(color_map[category])

    if not vectors:
        print("No valid words found in vocabulary to plot.")
        return
    vectors = np.stack(vectors)
    perplexity = min(30, (len(vectors) - 1) // 3)
    perplexity = max(perplexity, 2)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    reduced = tsne.fit_transform(vectors)
    plt.figure(figsize=(12, 8))
    for i, word in enumerate(words):
        x, y = reduced[i]
        plt.scatter(x, y, color=colors[i])
        plt.annotate(word, (x, y), fontsize=10)
    plt.title("t-SNE Visualization of Selected Word Groups (GloVe)")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def visualize_3d_selected_words_glov(word_to_vec, tsne_output="glove_selected_3d_tsne.png", umap_output="glove_selected_3d_umap.png"):
    thematic_groups = {
        "business": ['oil', 'energy', 'market', 'stocks', 'dollar'],
        "countries": ['china', 'japan', 'germany', 'brazil', 'india'],
        "sports": ['football', 'cricket', 'golf', 'olympics', 'tennis'],
        "health": ['virus', 'cancer', 'health', 'doctor', 'vaccine'],
        "tech": ['computer', 'internet', 'software', 'google', 'microsoft']
    }
    words, vectors, colors = [], [], []
    color_map = {"business": "red", "countries": "blue", "sports": "green", "health": "purple", "tech": "orange"}
    for category, word_list in thematic_groups.items():
        for word in word_list:
            if word in word_to_vec:
                words.append(word)
                vectors.append(word_to_vec[word])
                colors.append(color_map[category])
    if not vectors:
        print("No valid words found in vocabulary to plot.")
        return
    vectors = np.stack(vectors)
    reducer = UMAP(n_components=3, random_state=42)
    reduced_umap = reducer.fit_transform(vectors)
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    for i, word in enumerate(words):
        ax.scatter(*reduced_umap[i], color=colors[i])
        ax.text(*reduced_umap[i], word, size=9)
    plt.title("3D UMAP of Selected Word Groups (GloVe)")
    plt.tight_layout()
    plt.savefig(umap_output)
    plt.close()
