import tensorflow as tf


def pad_sequences(sequences, max_len):
    if max_len is None:
        max_len = max([len(seq) for seq in sequences])
    padded_seq = []
    masks = []
    for seq in sequences:
        l = len(seq)
        if l <= max_len:
            masks.append([1] * l + [0] * (max_len - l))
            padded_seq.append(seq + [0] * (max_len - l))
        else:
            masks.append([1] * max_len)
            padded_seq.append(seq[:max_len])
    return padded_seq, masks, max_len


def seq_length(masks):
    masks = tf.cast(masks, tf.int32)
    return tf.reduce_sum(masks, axis=1)


def lstm2logits(inputs, max_input_len):
    hidden = tf.layers.dense(inputs, 1)
    hidden = tf.reshape(hidden, shape=(-1, max_input_len))
    return hidden


def prepare_for_softmax(inputs, mask):
    new_mask = (1 - tf.cast(mask, tf.int32)) * tf.constant(-int(1e9), dtype=tf.int32)
    return tf.where(mask, inputs, tf.cast(new_mask, tf.float32))


def BiLSTM(inputs, masks, size, initial_state_fw=None, initial_state_bw=None, dropout=1.0, reuse=False):
    cell_fw = tf.nn.rnn_cell.BasicLSTMCell(size, reuse=reuse)
    cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, input_keep_prob=dropout)
    cell_bw = tf.nn.rnn_cell.BasicLSTMCell(size, reuse=reuse)
    cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, input_keep_prob=dropout)

    outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
        cell_fw, cell_bw, inputs, seq_length(masks),
        initial_state_bw=initial_state_bw,
        initial_state_fw=initial_state_fw,
        dtype=tf.float32
    )
    return tf.concat(outputs, 2), output_states
