import time
import logging
from collections import Counter

import numpy as np
import tensorflow as tf
from keras.utils import Progbar
from tensorflow.python.ops import variable_scope as vs

from evaluate import f1_score
from utils.base_models import BiLSTM, lstm2logits, prepare_for_softmax, pad_sequences
from utils.generals import batches

logging.basicConfig(level=logging.INFO)


def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn


class Encoder(object):
    def __init__(self, size):
        self.size = size

    def encode(self, inputs, masks, initial_state_fw=None, initial_state_bw=None, dropout=1.0, reuse=False):
        return BiLSTM(inputs, masks, self.size,
                      initial_state_fw, initial_state_bw,
                      dropout=dropout,
                      reuse=reuse)


def get_attention(hc, hq, c_mask, q_mask, mcl, mql, dropout=1):
    d = hc.shape[-1]
    w_cq = tf.get_variable('W_cq_ct', shape=(d, d))
    score = tf.reshape(tf.reshape(hc, shape=[-1, d]) @ w_cq, shape=(-1, mcl, d)) @ tf.transpose(hq, (0, 2, 1))
    c_mask_aug = tf.tile(tf.expand_dims(c_mask, 2), [1, 1, mql])
    q_mask_aug = tf.tile(tf.expand_dims(q_mask, 1), [1, mcl, 1])
    mask_aug = c_mask_aug & q_mask_aug
    score_prepro = prepare_for_softmax(score, mask_aug)
    alignment = tf.nn.softmax(score_prepro)
    context_aware = alignment @ hq
    concat_hidden = tf.concat([context_aware, hc], 2)
    concat_hidden = tf.cond(dropout < 1, lambda: tf.nn.dropout(concat_hidden, dropout), lambda: concat_hidden)
    w_s = tf.get_variable('W_s', shape=(d * 2, d))
    return tf.nn.tanh(
        tf.reshape(tf.reshape(concat_hidden, shape=(-1, d * 2)) @ w_s, shape=(-1, mcl, d))
    )


class Decoder(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def decode(self, inputs, mask, max_input_len, dropout):
        with vs.variable_scope('m1'):
            m1, _ = BiLSTM(inputs, mask, self.output_size, dropout=dropout)

        with vs.variable_scope('m2'):
            m2, _ = BiLSTM(m1, mask, self.output_size, dropout=dropout)

        with vs.variable_scope('start'):
            start = lstm2logits(tf.concat([inputs, m1], 2), max_input_len)
            start = prepare_for_softmax(start, mask)

        with vs.variable_scope('end'):
            end = lstm2logits(tf.concat([inputs, m2], 2), max_input_len)
            end = prepare_for_softmax(end, mask)

        return start, end


class QASystem(object):
    def __init__(self, embeddings, *args):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """
        self.FLAG = tf.app.flags.FLAGS

        # ==== set up placeholder tokens ========
        self.context_placeholder = tf.placeholder(tf.int32, shape=(None, None))
        self.context_mask_placeholder = tf.placeholder(tf.bool, shape=(None, None))
        self.context_max_len_placeholder = tf.placeholder(tf.int32)
        self.question_placeholder = tf.placeholder(tf.int32, shape=(None, None))
        self.question_mask_placeholder = tf.placeholder(tf.bool, shape=(None, None))
        self.question_max_len_placeholder = tf.placeholder(tf.int32)
        self.dropout_placeholder = tf.placeholder(tf.float32)

        self.answer_se_placeholder = tf.placeholder(tf.int32, shape=(2, None))

        self.encoder = Encoder(self.FLAG.state_size)
        self.decoder = Decoder(self.FLAG.output_size)

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.context_embeddings, self.question_embeddings = self.setup_embeddings(embeddings)
            self.start, self.end = self.setup_pred()
            self.loss = self.setup_loss()
            self.train_op = self.setup_train_op()

    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """

    def setup_pred(self):
        with vs.variable_scope("q"):
            Hq, (q_final_state_fw, q_final_state_bw) = self.encoder.encode(
                self.question_embeddings,
                self.question_mask_placeholder,
                dropout=self.dropout_placeholder
            )
        with vs.variable_scope('c'):
            Hc, _ = self.encoder.encode(
                self.context_embeddings,
                self.context_mask_placeholder,
                initial_state_fw=q_final_state_fw,
                initial_state_bw=q_final_state_bw,
                dropout=self.dropout_placeholder
            )
        with vs.variable_scope('attention'):
            attention = get_attention(
                Hc, Hq,
                self.context_mask_placeholder,
                self.question_mask_placeholder,
                self.context_max_len_placeholder,
                self.question_max_len_placeholder,
                self.dropout_placeholder
            )

        with vs.variable_scope('decode'):
            start, end = self.decoder.decode(
                attention,
                self.context_mask_placeholder,
                self.context_max_len_placeholder,
                self.dropout_placeholder
            )
        return start, end

    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            ans_s_one_hot = tf.one_hot(self.answer_se_placeholder[0], self.context_max_len_placeholder)
            ans_e_one_hot = tf.one_hot(self.answer_se_placeholder[1], self.context_max_len_placeholder)

            loss1 = tf.nn.softmax_cross_entropy_with_logits(logits=self.start, labels=ans_s_one_hot)
            loss2 = tf.nn.softmax_cross_entropy_with_logits(logits=self.end, labels=ans_e_one_hot)

        return loss1 + loss2

    def setup_train_op(self):
        variables = tf.trainable_variables()
        gradients = tf.gradients(self.loss, variables)
        gradients, _ = tf.clip_by_global_norm(gradients, self.FLAG.max_gradient_norm)
        learning_rate = self.FLAG.learning_rate
        optimizer = get_optimizer(self.FLAG.optimizer)(learning_rate=learning_rate)
        train_op = optimizer.apply_gradients(zip(gradients, variables))

        # ema = tf.train.ExponentialMovingAverage(0.999)
        # ema_op = ema.apply(variables)
        # with tf.control_dependencies([train_op]):
        #     train_op = tf.group(ema_op)
        return train_op

    def setup_embeddings(self, embeddings):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with vs.variable_scope("embeddings"):
            embeddings_var = tf.get_variable('embeddings', initializer=embeddings)
            context_embeddings = self.embedding_lookup(
                embeddings_var,
                self.context_placeholder,
                self.context_max_len_placeholder
            )
            question_embeddings = self.embedding_lookup(
                embeddings_var,
                self.question_placeholder,
                self.question_max_len_placeholder
            )
        return context_embeddings, question_embeddings

    def embedding_lookup(self, embeddings, indices, max_length):
        embeddings = tf.nn.embedding_lookup(embeddings, indices)
        embeddings = tf.reshape(embeddings, shape=(-1, max_length, self.FLAG.embedding_size))
        return embeddings

    def optimize(self, session, dataset):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        context = dataset['context']
        question = dataset['question']
        ans_s = dataset['answer_span_start']
        ans_e = dataset['answer_span_end']

        input_feed = self.create_feed_dict(context, question, ans_s, ans_e, is_train=True)

        output_feed = [self.train_op, self.loss]

        _, loss = session.run(output_feed, input_feed)
        return loss

    def create_feed_dict(self, context, question, answer_span_start_batch=None, answer_span_end_batch=None,
                         is_train=True):
        padded_context, context_mask, max_context_len = pad_sequences(context, self.FLAG.max_context_length)
        padded_question, question_mask, max_question_len = pad_sequences(question, self.FLAG.max_question_length)
        feed_dict = {
            self.context_placeholder: padded_context,
            self.context_mask_placeholder: context_mask,
            self.context_max_len_placeholder: max_context_len,
            self.question_placeholder: padded_question,
            self.question_mask_placeholder: question_mask,
            self.question_max_len_placeholder: max_question_len,
        }
        if answer_span_start_batch is not None and answer_span_end_batch is not None:
            feed_dict[self.answer_se_placeholder] = (answer_span_start_batch, answer_span_end_batch)
        if is_train:
            feed_dict[self.dropout_placeholder] = self.FLAG.dropout
        else:
            feed_dict[self.dropout_placeholder] = 1.0
        return feed_dict

    def test(self, session, valid_data):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        context = valid_data['context']
        question = valid_data['question']
        ans_s = valid_data['answer_span_start']
        ans_e = valid_data['answer_span_end']
        input_feed = self.create_feed_dict(context, question, ans_s, ans_e, is_train=False)
        output_feed = self.loss

        outputs = session.run(output_feed, input_feed)

        return outputs

    def decode(self, session, test_x):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        context = test_x['context']
        question = test_x['question']
        input_feed = self.create_feed_dict(context, question, is_train=False)

        # fill in this feed_dictionary like:
        # input_feed['test_x'] = test_x

        output_feed = [self.start, self.end]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def answer(self, session, test_x):

        yp, yp2 = self.decode(session, test_x)

        a_s = np.argmax(yp, axis=1)
        a_e = np.argmax(yp2, axis=1)

        return (a_s, a_e)

    def validate(self, sess, valid_dataset):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """
        valid_cost = 0

        for valid_data in valid_dataset:
            valid_cost += self.test(sess, valid_data)

        return valid_cost / len(valid_dataset)

    def evaluate_answer(self, session, dataset, sample=100, log=True):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """

        indices = np.arange(len(dataset['context']))
        indices = np.random.choice(indices, sample)
        sampled = {}
        for k, v in dataset.items():
            sampled[k] = v[indices]
        pred_s, pred_e = self.answer(session, sampled)
        ans_s, ans_e = sampled['answer_span_start'], sampled['answer_span_end']
        em = np.sum((pred_s == ans_s) &
                    (pred_e == ans_e)) / sample

        f1 = 0.
        for i in range(sample):
            prediction_tokens = sampled['context'][i][pred_s[i]:pred_e[i] + 1]
            ground_truth_tokens = sampled['context'][i][ans_s[i]:ans_e[i] + 1]
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 += (2 * precision * recall) / (precision + recall)

        f1 /= sample

        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))

        return f1, em

    def train(self, session, dataset, train_dir):
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious appoarch can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.

        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.

        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param train_dir: path to the directory where you should save the model checkpoint
        :return:
        """

        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum([np.prod(tf.shape(t.value()).eval()) for t in params])
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        for i in range(self.FLAG.epochs):
            self.run_epoch(session, dataset, train_dir)

    def run_epoch(self, session, dataset, train_dir):
        num_samples = len(dataset['context'])
        num_batches = num_samples // self.FLAG.batch_size + 1
        progress = Progbar(target=num_batches)
        for i, train_batch in enumerate(batches(dataset, is_train=True, batch_size=self.FLAG.batch_size)):
            loss_ = self.optimize(session, train_batch)
            progress.update(i + 1, [('loss', loss_)])
            if i % 1000 == 0:
                saver = tf.train.Saver()
                saver.save(session, train_dir + "/mymodel")
                self.evaluate_answer(session, dataset, log=True)
