import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

class Dictionary:
    def __init__(self, filepath):
        dictionary = {}
        with open(filepath, 'r', encoding='latin1') as in_file:
            for line in in_file:
                if not line or line.startswith(';'):
                    continue
                words = line.strip().split()
                dictionary[words[0]] = len(dictionary)
        self.dictionary = dictionary

    def index(self, word):
        return self.dictionary[word.upper()]


def indexes(word, dictionary):
    filedir = os.path.join(os.path.dirname(__file__), '..', 'res')
    filepath = os.path.join(filedir, f'vitz-1973-experiment-{word}.csv')
    df = pd.read_csv(filepath)
    return [dictionary.index(w) for w in df['word'].to_list()]


def main():
    filedir = os.path.join(os.path.dirname(__file__), '..', 'data')
    filepath = os.path.join(filedir, 'cmudict-0.7b-with-vitz-nonce')
    dictionary = Dictionary(filepath)
    d = {}
    for word in ['sit', 'plant', 'wonder', 'relation']:
        d[dictionary.index(word)] = indexes(word, dictionary)
    print(d)

if __name__ == '__main__':
    main()
