#!/usr/bin/env python3

from collections import defaultdict, OrderedDict
from typing import Iterable, Set


class DataToken:
    def __init__(self, offset, word, lowercase, pos_tag, mwe_tag, parent_offset, strength, supersense, sentence_id):
        self.offset = int(offset)
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

    def __iter__(self) -> Iterable[DataToken]:
        return iter(self._tokens.values())

    def __getitem__(self, item) -> DataToken:
        return self._tokens[item]

    def __bool__(self) -> bool:
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
    def sentence(self) -> str:
        return ' '.join(map(lambda t: t.word, self._tokens.values()))


class EmissionProbabilities:
    def __init__(self, features: Set[str]):
        self.counts = {feature: 0 for feature in features}
        self.total_count = 0

    def __getitem__(self, feature: str):
        return self.get_probability(feature)

    def add_count(self, feature: str, count=1):
        self.counts[feature] += count
        self.total_count += 1

    def get_probability(self, feature: str):
        return self.counts[feature] / self.total_count

    def __str__(self):
        lines = []
        for feature in self.counts:
            lines.append(f'{round(self.get_probability(feature), 4):>8} : {feature}')
        return '\n'.join(lines)


class HMM:
    def __init__(self, features: Set[str]):
        self.features = features
        self.emissions = {}

    def get_features_for_token(self, token: DataToken) -> Iterable[str]:
        features = []
        mwe_feature = f'{token.mwe_tag}_mwe'
        features.append(mwe_feature)
        return features

    def train(self, sentences: Iterable[DataSentence]):
        for sentence in sentences:
            for token in sentence:
                for feature in self.get_features_for_token(token):
                    ep = self.emissions.get(token.lowercase_lemma)
                    if ep is None:
                        ep = EmissionProbabilities(self.features)
                    ep.add_count(feature)

    def test(self, sentences: Iterable[DataSentence]):
        ...


def generate_features(sentences: DataSentence):
    features = set()
    features.add('O_mwe')
    features.add('o_mwe')
    features.add('B_mwe')
    features.add('b_mwe')
    features.add('I_mwe')
    features.add('i_mwe')
    return features


def read_data_file(datafile: str) -> Iterable[DataSentence]:
    data_sentences = defaultdict(DataSentence)
    with open(datafile) as df:
        for line in df:
            if not line.isspace():
                # Strip extraneous whitespace from the line, then split it on the tabs. There should be exactly enough
                # fields to fill the DataToken.
                token = DataToken(*(line.strip(' \n').split('\t')))
                data_sentences[token.sentence_id].append(token)
    return data_sentences.values()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('training-data', help='the datafile to train with')
    parser.add_argument('testing-data', help='the datafile to test with')
    args = parser.parse_args()

    # Read the data.
    training_sentences = read_data_file(args.training_data)
    testing_sentences = read_data_file(args.testing_data)
