#!/usr/bin/env python3

from collections import defaultdict, OrderedDict


class DataToken:
    def __init__(self, offset, word, lowercase, pos_tag, mwe_tag, parent_offset, strength, supersense, sentence_id):
        self.offset = offset
        self.word = word
        self.lowercase_lemma = lowercase
        self.pos_tag = pos_tag
        self.mwe_tag = mwe_tag
        self.parent_offset = None if parent_offset == '0' else int(parent_offset)
        self.strength = strength
        self.supersense = supersense
        self.sentence_id = sentence_id


class DataSentence:
    def __init__(self):
        self._tokens: OrderedDict = OrderedDict()
        self.sentence_id = None

    def __iter__(self):
        return iter(self._tokens)

    def __getitem__(self, item):
        return self._tokens[item]

    def __bool__(self):
        return bool(self._tokens)

    def append(self, token: DataToken):
        if not self.sentence_id:
            self.sentence_id = token.sentence_id
        else:
            if token.sentence_id != self.sentence_id:
                raise ValueError("sentence IDs do not match: {} (sentence) != {} (token)".format(self.sentence_id,
                                                                                                 token.sentence_id))
        self._tokens[int(token.offset)] = token

    @property
    def sentence(self):
        return ' '.join(map(lambda t: t.word, self._tokens.values()))


def read_data_file(datafile: str):
    data_sentences = defaultdict(DataSentence)
    with open(datafile) as df:
        for line in df:
            if not line.isspace():
                token = DataToken(*line.strip(' \n').split('\t'))
                data_sentences[token.sentence_id].append(token)
    return data_sentences


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('training-data', help='the datafile to train with')
    parser.add_argument('testing-data', help='the datafile to test with')
    args = parser.parse_args()

    # Read the data.
    training_sentences = read_data_file(args.training_data)
    testing_sentences = read_data_file(args.testing_data)
