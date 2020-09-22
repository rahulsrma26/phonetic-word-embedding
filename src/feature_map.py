"""
Read feature mapping file.
A feature mapping file is space separaed text file in the format of

phoneme feature1 feature2 ... featureN
"""

class FeatureMap:
    def __init__(self, filepath, encoding='utf-8'):
        features_set = {}
        with open(filepath, 'r', encoding=encoding) as in_file:
            for line in in_file:
                line = line.strip()
                if line and line[0] != ';':
                    word, *features = line.split(' ')
                    features_set[word] = set(features)
        self.features_set = features_set

    def features(self, phone):
        if phone[-1] in '012':
            phone = phone[:-1]
        return self.features_set[phone]

    def __len__(self):
        return len(self.features_set)
