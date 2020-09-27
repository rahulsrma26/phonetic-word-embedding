"""
Load words from a dictionary file.

A file may contain words and their phoneme sequence like CMU dictionary.
Or
It can contain only words like hindi dictionary.
Languages like hindi have orthographies with a high grapheme-to-phoneme and phoneme-to-grapheme correspondence.
So in that case charaters in the word are the phoneme sequence itself.
"""

class Dictionary:
    def __init__(self, filepath, encoding):
        dictionary = None
        with open(filepath, 'r', encoding=encoding) as in_file:
            for line in in_file:
                if not line or line.startswith(';'):
                    continue
                words = line.strip().split()
                if dictionary is None:
                    dictionary = set() if len(words) == 1 else {}
                if len(words) == 1:
                    dictionary.add(words[0])
                else:
                    dictionary[words[0]] = words[1:]
        self.dictionary = dictionary

    def phones(self, word):
        return self.dictionary[word]

    def __contains__(self, word):
        return word in self.dictionary

    # def __iter__(self):
    #     return iter(self.dictionary)

    def __len__(self):
        return len(self.dictionary)
