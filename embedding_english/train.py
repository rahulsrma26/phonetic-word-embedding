import os
import argparse
from wsim.wsim import wsimdict as wd
from time import time
import numpy as np
from numpy.linalg import norm
import tensorflow as tf
from tensorflow.keras.constraints import UnitNorm
from random import randrange, random

D_FLAGS = wd.BIGRAM | wd.INSERT_BEG_END | wd.VOWEL_BUFF
D_PEN = 0.4  # penalty = 1/0.4 = 2.5


class Dictionary():
    def data_generator(self):
        n = int(len(self.dictionary))
        for _ in range(self.num_of_batches):
            (i1,
             i2), s = self.dictionary.random_scores(self.batch_size, D_FLAGS,
                                                    D_PEN)
            yield ((np.array(i1), np.array(i2)), np.array(s))

    def __init__(self, params):
        self.dictionary = wd(params.dictionary_path)
        self.num_of_batches = params.num_of_batches
        self.batch_size = params.batch_size

    def dataset(self):
        return tf.data.Dataset.from_generator(
            self.data_generator, ((tf.int64, tf.int64), tf.float32),
            ((tf.TensorShape([None]), tf.TensorShape(
                [None])), tf.TensorShape([None])))


class EmbeddingModel(tf.keras.Model):
    def __init__(self, dictionary, params):
        super(EmbeddingModel, self).__init__()
        self.embedding = tf.Variable(tf.random.normal(
            [len(dictionary), params.vector_size]),
                                     name='embeddings',
                                     trainable=True,
                                     dtype=tf.float32)
                                    #  constraint=UnitNorm(axis=-1))
        # self.embedding = tf.compat.v1.get_variable('embeddings', initializer=tf.random.normal(
        #     [len(dictionary), params.vector_size]),
        #                              trainable=True,
        #                              dtype=tf.float32,
        #                              constraint=tf.keras.constraints.UnitNorm)

    @tf.function
    def call(self, inputs, training=True):
        x1, x2 = inputs
        # print('input shape', x1.shape)
        X1 = tf.nn.embedding_lookup(self.embedding, x1, max_norm=1)
        X2 = tf.nn.embedding_lookup(self.embedding, x2, max_norm=1)
        # print('lookup shape', X1.shape)
        # X1 = tf.nn.l2_normalize(X1, axis=-1)
        # X2 = tf.nn.l2_normalize(X2, axis=-1)
        # print('l2_normalize shape', X1.shape)
        cos_similarity = tf.reduce_sum(tf.multiply(X1, X2), axis=-1)
        # print('cos_similarity shape', cos_similarity.shape)
        return cos_similarity


def _main(params):
    config(params)
    dictionary = Dictionary(params)
    dataset = dictionary.dataset()
    # def parse_fn(_):
    #     return EmbeddingDataset(dictionary, params.num_of_batches // 4, params.batch_size)
    # dataset = tf.data.Dataset.range(4).interleave(
    #     parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    model = EmbeddingModel(
        dictionary.dictionary,
        params) if params.load_model == '' else tf.keras.models.load_model(
            params.load_model)

    opt = tf.keras.optimizers.Adam(learning_rate=params.learning_rate)
    model.compile(optimizer=opt, loss='mean_squared_error') # loss is MSE

    if os.path.isdir(params.checkpoint_path):
        model.load_weights(params.checkpoint_path)
    else:
        os.makedirs(params.checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=params.checkpoint_path, save_weights_only=True, verbose=1)

    for i in range(params.num_of_epochs):
        start = time()
        result = model.fit(dataset, callbacks=[cp_callback])
        with open('history.csv', 'a+') as f:
            f.write(f'{result.history["loss"][0]}\n')
        print(f'Epoch: {i}, time taken: {time() - start}')
        # model.embedding = tf.nn.l2_normalize(model.embedding, axis=-1)
        embedding = model.embedding.numpy()
        # model.embedding = norm(embedding, axis=0)
        model.save(params.save_model)
        check(dictionary.dictionary, embedding, params.check_size)
        export(dictionary.dictionary, embedding, params.save_embedding)


def check(dict, embd, n):
    # actual, predicted = [], []
    (i1, i2), actual = dict.random_scores(n, D_FLAGS, D_PEN)
    v1, v2 = embd[i1], embd[i2]
    n1, n2 = norm(v1, axis=1), norm(v2, axis=1)
    predicted = np.sum(v1 * v2, axis=1) / (n1 * n2)
    print(f'{n} MSE', np.square(actual - predicted).mean())
    a = np.abs(actual - predicted)
    print(f'{n} Diff avg: {np.mean(a)}, minmax: {np.min(a)} - {np.max(a)}')
    print(f'{n} Embedding Norms min: {min(np.min(n1), np.min(n2))} max: {max(np.max(n1), np.max(n2))}')


def export(dictionary, embedding, path):
    with open(path, 'w') as fout:
        for i, v in enumerate(embedding):
            word = dictionary.get_word(i)
            vector = ' '.join([str(x) for x in v])
            fout.write(f'{word}  {vector}\n')


def config(params):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices(
                    'GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus),
                      "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def _args():
    parser = argparse.ArgumentParser(
        prog=__file__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dictionary_path', type=str)
    parser.add_argument('-bs',
                        '--batch_size',
                        type=int,
                        default=4096,
                        metavar='',
                        help='batch size')
    parser.add_argument('-vs',
                        '--vector_size',
                        type=int,
                        default=50,
                        metavar='',
                        help='embedding vector length')
    parser.add_argument('-nb',
                        '--num_of_batches',
                        type=int,
                        default=100,
                        metavar='',
                        help='number_of_batches_per_epochs')
    parser.add_argument('-ne',
                        '--num_of_epochs',
                        type=int,
                        default=1,
                        metavar='',
                        help='number_of_epochs')
    parser.add_argument('-lm',
                        '--load_model',
                        type=str,
                        default='',
                        metavar='',
                        help='path to load_model')
    parser.add_argument('-sm',
                        '--save_model',
                        type=str,
                        default='model.tf',
                        metavar='',
                        help='path to save_model')
    parser.add_argument('-se',
                        '--save_embedding',
                        type=str,
                        default='simvecs',
                        metavar='',
                        help='path to save_embedding')
    parser.add_argument('-cp',
                        '--checkpoint_path',
                        type=str,
                        default='checkpoints/',
                        metavar='',
                        help='checkpoint path')
    parser.add_argument('-cs',
                        '--check_size',
                        type=int,
                        default=100000,
                        metavar='',
                        help='number of pairs to check after each epochs')
    parser.add_argument('-lr',
                        '--learning_rate',
                        type=float,
                        default=0.001,
                        metavar='',
                        help='learning rate for Adam optimizer')
    return parser.parse_args()


if __name__ == '__main__':
    params = _args()
    _main(params)