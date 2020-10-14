"""
Embedding
==========

Loads the vitz-1973-experiment dataset and generate embedding scores for them.
"""

import os
import sys
from argparse import ArgumentParser, RawTextHelpFormatter
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

class Dictionary:
    def __init__(self, filepath, encoding="latin1"):
        self.words = list()
        self.lookup = dict()
        dictionary = list()

        print("loading...", file=sys.stderr)
        for i, line in enumerate(open(filepath, encoding=encoding)):
            line = line.strip()
            word, vec_s = line.split("  ")
            vec = [float(n) for n in vec_s.split()]
            self.lookup[word] = i
            dictionary.append(vec)
            self.words.append(word)
        print(f'Total words: {len(self.words)}', file=sys.stderr)
        self.dictionary = np.array(dictionary)
        self.norms = normalize(self.dictionary, axis=1)
        print('min Norm', np.min(self.norms))
        print('max Norm', np.max(self.norms))

    def vec(self, word):
        return self.dictionary[self.lookup[word.strip().upper()], :]

    def score(self, word1, word2):
        v1 = self.norms[self.lookup[word1.strip().upper()], :]
        v2 = self.norms[self.lookup[word2.strip().upper()], :]
        return np.sum(v1*v2)

    def word(self, vec, n=None):
        v = vec / np.linalg.norm(vec)
        dots = np.dot(self.norms, v)
        if n is None:
            return self.words[np.argmax(dots)]
        return [(self.words[x], dots[x]) for x in np.argsort(-dots)[:n]]
        # return [self.words[x] for x in np.argsort(-dots)[:n]]


def compare(word, dictionary, res_dir):
    filepath = os.path.join(res_dir, f'vitz-1973-experiment-{word}.csv')
    df = pd.read_csv(filepath)
    df['score'] = df.apply(
        lambda row: dictionary.score(row['word'], word), axis=1)
    df['actual'] = word
    return df[['actual', 'word', 'score']]

def main(args):
    dictionary = Dictionary(args.input, encoding=args.encoding)
    words = ['sit', 'plant', 'wonder', 'relation']
    df =  pd.concat([compare(w, dictionary, args.res) for w in words], ignore_index=True)
    df.to_csv(args.output, index=False)


def _get_args():
    parser = ArgumentParser(
        os.path.basename(__file__), description=__doc__, formatter_class=RawTextHelpFormatter)
    parser.add_argument("input", type=str, help='embedding file path')
    parser.add_argument("output", type=str, help='output score file path')
    parser.add_argument("-e", "--encoding", default='latin1', type=str, help='File encoding (default: latin1)')
    res_dir = os.path.join(os.path.dirname(__file__), '..', 'res')
    parser.add_argument("-r", "--res", default=res_dir, type=str, help=f'Resourse directory (default: {res_dir})')
    return parser.parse_args()

if __name__ == '__main__':
    main(_get_args())
