import os
import sys
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


def compare(word, dictionary):
    filedir = os.path.join(os.path.dirname(__file__), '..', 'res')
    filepath = os.path.join(filedir, f'vitz-1973-experiment-{word}.csv')
    df = pd.read_csv(filepath)
    df['embedding'] = df.apply(
        lambda row: dictionary.score(row['word'], word), axis=1)
    obtained = df['obtained'].to_numpy()
    actual = df['embedding'].to_numpy()
    return np.corrcoef(obtained, actual)[0, 1]

def main():
    dictionary = Dictionary('simvecs')
    rows = []
    for word in ['sit', 'plant', 'wonder', 'relation']:
        rows.append([word, compare(word, dictionary)])
    filedir = os.path.join(os.path.dirname(__file__), '..', 'res')
    filepath = os.path.join(filedir, f'embedding_score.csv')
    pd.DataFrame(rows, columns=['word', 'score']).to_csv(filepath, index=False)

if __name__ == '__main__':
    main()
