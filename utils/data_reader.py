import numpy as np


def load_embeddings(embeddings_path):
    return np.load(embeddings_path)['glove'].astype(np.float32)