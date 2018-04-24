import numpy as np


def batches(dataset, is_train=True, batch_size=24, window_size=3, shuffle=True):
    n_samples = len(dataset['context'])
    n_buckets = n_samples // batch_size + 1
    if is_train:
        batch_indices = np.arange(n_samples)
        if shuffle:
            np.random.shuffle(batch_indices)
        for i in range(n_buckets):
            start = i * batch_size
            end = min(start + batch_size * window_size, n_samples)
            window = list(range(start, end))
            indices = np.random.choice(window, min(len(window), batch_size), replace=False)
            ret = {}
            for k, v in dataset.items():
                ret[k] = v[indices]
            yield ret
    else:
        indices = np.arange(n_samples)
        if shuffle:
            np.random.shuffle(indices)
        for i in range(0, n_buckets * batch_size, batch_size):
            ret = {}
            for k, v in dataset.items():
                ret[k] = v[indices[i:i + batch_size]]
            yield ret
