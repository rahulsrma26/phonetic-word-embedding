"""
Word Similarity
===============
Enter single word to get top 20 results.
Enter two words to get the similarity.
Enter empty line to exit.
"""
import os
import io
import sys
import math
from time import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
np.set_printoptions(linewidth=np.inf)

from feature_map import FeatureMap
from dictionary import Dictionary


class Similarity:
    def __init__(self, mapping_file, dictionary_file, encoding, debug=False):
        self.phonetic = FeatureMap(mapping_file)
        print(f'Phonetic information loaded: {len(self.phonetic)}')
        self.dictionary = Dictionary(dictionary_file, encoding)
        print(f'Dictionary loaded: {len(self.dictionary)}')
        self.debug = debug

    def phone_similarity(self, ph1, ph2, vowel_weight):
        if isinstance(ph1, str):
            ph1 = [ph1]
        if isinstance(ph2, str):
            ph2 = [ph2]
        s1 = set([y for x in ph1 for y in self.phonetic.features(x)])
        s2 = set([y for x in ph2 for y in self.phonetic.features(x)])
        common = s1.intersection(s2)
        total = s1.union(s2)
        score = len(common) / len(total)
        if vowel_weight:
            same = ph1[-1] == ph2[-1] and 'vwl' in self.phonetic.features(ph1[-1])
            score = score**0.5 if same else score**2
        return score

    def get_phone_matrix(self, word1, word2, bigram, weight):
        ph1 = ['^'] + self.dictionary.phones(word1) + ['$']
        ph2 = ['^'] + self.dictionary.phones(word2) + ['$']
        a = None
        if bigram:
            a = np.array([[
                self.phone_similarity(p1, p2, weight)
                for p1 in zip(ph1[:-1], ph1[1:])
            ] for p2 in zip(ph2[:-1], ph2[1:])])
        else:
            a = np.array(
                [[self.phone_similarity(p1, p2, weight) for p1 in ph1]
                 for p2 in ph2])
        if self.debug:
            print(ph1)
            print(ph2)
            print(a)
        return a

    def word_similarity(self,
                        word1,
                        word2,
                        bigram=True,
                        vowel=True,
                        penalty=1,
                        rhyme=False):
        a = self.get_phone_matrix(word1, word2, bigram, vowel)
        d = np.zeros(a.shape)
        n2, n1 = a.shape
        for j in range(1, n1):
            a[0, j] += a[0, j - 1]
        for i in range(1, n2):
            a[i, 0] = a[i,0] + a[i - 1, 0]
            for j in range(1, n1):
                if a[i, j] < 1:
                # if max(a[i, j - 1], a[i - 1, j]) > a[i,j]:
                    if a[i, j - 1] > a[i - 1, j]:
                        # d[i, j] += d[i, j - 1]
                        a[i, j] = a[i, j] / penalty + a[i, j - 1]
                    else:
                        # d[i, j] += d[i - 1, j]
                        a[i, j] = a[i, j] / penalty + a[i - 1, j]
                else:
                    d[i, j] += d[i - 1, j - 1] + 1
                    a[i, j] += a[i - 1, j - 1]
                # if a[i, j] < 1:
                #     a[i, j] /= 2
                # a[i, j] += max(a[i - 1, j - 1], a[i, j - 1], a[i - 1, j])
                # a[i, j] += max(a[i - 1, j - 1], a[i, j - 1], a[i - 1, j])
        if self.debug:
            print(a)
            # print(d)
            pass
        score = a[-1, -1] / max(n1, n2) if bigram else (a[-1, -1] - 2) / (max(n1, n2) - 2)
        if rhyme:
            weight = max(0, d[-1, -1] - 1) / (max(n1, n2) - 1)
            weight *= weight
            # print(score, weight)
            score = score * (1 - weight) + weight
            # print(score)
        return score

    def top_similar(self, word, **params):
        debug_status = self.debug
        self.debug = False
        top = []
        for w in self.dictionary.dictionary.keys():
            top.append((self.word_similarity(word, w, **params), w))
        top.sort(key=lambda tup: tup[0], reverse=True)
        self.debug = debug_status
        return top


def _main(args):
    sim = Similarity(
        mapping_file=args.map, dictionary_file=args.dictionary, encoding=args.encoding, debug=args.debug)
    print(__doc__)
    while True:
        words = input().strip().split()
        if not words:
            break
        if len(words) == 2:
            result = sim.word_similarity(words[0].upper(), words[1].upper(), bigram=args.bigram, vowel=args.vowel, penalty=args.penalty)
            print('similarity', result)
        elif len(words) == 1:
            start_time = time()
            result = sim.top_similar(words[0].upper(), bigram=args.bigram, vowel=args.vowel, penalty=args.penalty)
            print('time taken', time() - start_time)
            print('similarity', result[:20])
        else:
            print(__doc__)


def _get_args():
    parser = ArgumentParser(
        os.path.basename(__file__), description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('map', type=str, help='phonetic feature mapping file')
    parser.add_argument('dictionary', type=str, help='dictionary file')
    parser.add_argument('-e', '--encoding', type=str, default='latin1', help='dictionary encoding')
    parser.add_argument('-d', '--debug', action='store_true', help='displays intermediate information')
    parser.add_argument('-b', '--bigram', action='store_true', help='Enables bigram istead of unigram')
    parser.add_argument('-v', '--vowel', action='store_true', help='vowel weight (only if bigram enabled)')
    parser.add_argument('-p', '--penalty', type=float, default=1, help='non-diagonal penalty')
    return parser.parse_args()


if __name__ == '__main__':
    _main(_get_args())
    # sim = WordSimilarity(os.path.join('phonetic-similarity-vectors', 'cmudict-0.7b-with-vitz-nonce'))

    # # print(phone_similarity('S', 'Z'))
    # # print(sim.word_similarity('split', 'plant'))
    # if len(sys.argv) > 1 and sys.argv[1] == '1':
    #     DEBUG = False
    #     while True:
    #         start = time()
    #         # print([x[1] for x in sim.top_similar(input().strip())[:10]])
    #         print(sim.top_similar(input().strip())[:20])
    #         print(time() - start)
    # # print(sim.word_similarity('sit', 'feet', use_bigram=False, vowel_weight=False, non_diagonal_penalty=1))  # tall feet
    # print(sim.word_similarity('sit', 'hit', use_bigram=False, vowel_weight=False, non_diagonal_penalty=1))
    # # print(sim.phone_similarity(['W', 'AH'], ['^', 'AH'], True))
    # # print(sim.word_similarity('WONDER', 'ASUNDER', use_bigram=True, vowel_weight=True, non_diagonal_penalty=1))
    # # print(sim.word_similarity('night', 'light'))
    # # print(sim.word_similarity('night', 'lite'))
    # # print(sim.word_similarity('plant', 'grant'))
    # # print(sim.word_similarity('plant', 'plint'))
    # # print(sim.word_similarity('wonder', 'tender'))
    # # print(sim.word_similarity('wonder', 'wunter'))
    # # print(sim.word_similarity('wunter', 'wonder'))