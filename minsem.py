#!/usr/bin/env python3

from collections import defaultdict, OrderedDict
from typing import Dict, Iterable, Set


class DataToken:
    def __init__(self, offset, word, lowercase, pos_tag, mwe_tag, parent_offset, strength, supersense, sentence_id):
        self.offset = int(offset)
        self.word = word
        self.lowercase_lemma = lowercase
        self.pos_tag = pos_tag
        self.mwe_tag = mwe_tag
        self.parent_offset = int(parent_offset)
        self.strength = strength
        self.supersense = supersense
        self.sentence_id = sentence_id

    def __repr__(self):
        return '\t'.join(map(lambda w: str(w) if w is not None else '', [
            self.offset, self.word, self.lowercase_lemma, self.pos_tag, self.mwe_tag, self.parent_offset, self.strength,
            self.supersense, self.sentence_id
        ]))

    def __str__(self):
        return repr(self)


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

    def get_highest_probability_feature(self) -> str:
        highest = 0
        best_feature = None
        for feature, count in self.counts.items():
            if count > highest:
                highest = count
                best_feature = feature
        return best_feature

    def __str__(self):
        lines = []
        for feature in self.counts:
            lines.append(f'{round(self.get_probability(feature), 4):>8} : {feature}')
        return '\n'.join(lines)


class HMM:
    def __init__(self, mwe_features: Set[str], ss_features: Set[str]):
        self.mwe_features = mwe_features
        self.ss_features = ss_features
        self.mwe_emissions: Dict[str, EmissionProbabilities] = {}
        self.ss_emissions: Dict[str, EmissionProbabilities] = {}

    def train(self, sentences: Iterable[DataSentence]):
        for sentence in sentences:
            for token in sentence:
                # Count emission probability for MWE.
                mwe_feature = token.mwe_tag
                mwe_ep = self.mwe_emissions.get(token.lowercase_lemma)
                if mwe_ep is None:
                    mwe_ep = EmissionProbabilities(self.mwe_features)
                    self.mwe_emissions[token.lowercase_lemma] = mwe_ep
                mwe_ep.add_count(mwe_feature)
                # Count emission probability for SS.
                ss_feature = token.supersense
                ss_ep = self.ss_emissions.get(token.lowercase_lemma)
                if ss_ep is None:
                    ss_ep = EmissionProbabilities(self.ss_features)
                    self.ss_emissions[token.lowercase_lemma] = ss_ep
                ss_ep.add_count(ss_feature)

    def test(self, sentences: Iterable[DataSentence]):
        for sentence in sentences:
            for token in sentence:
                # Find most-likely MWE tag.
                mwe_ep = self.mwe_emissions.get(token.lowercase_lemma)
                if mwe_ep is not None:
                    mwe_tag = mwe_ep.get_highest_probability_feature()
                    token.mwe_tag = mwe_tag
                # Find most-likely SS tag.
                ss_ep = self.ss_emissions.get(token.lowercase_lemma)
                if ss_ep is not None:
                    ss_tag = ss_ep.get_highest_probability_feature()
                    token.supersense = ss_tag


MWE_FEATURES = {'O', 'o', 'B', 'b', 'I', 'i'}


def generate_ss_features(sentences: Iterable[DataSentence]) -> Set[str]:
    ss_features = set()
    for sentence in sentences:
        for token in sentence:
            ss_features.add(token.supersense)
    return ss_features


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


def write_predictions(sentences: Iterable[DataSentence], outfile: str):
    with open(outfile, 'w') as of:
        for sentence in sentences:
            for token in sentence:
                of.write(repr(token))
                of.write('\n')
            of.write('\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('training_data', help='the datafile to train with')
    parser.add_argument('testing_data', help='the datafile to test with')
    parser.add_argument('predictions_output', default='minsem.pred', nargs='?',
                        help='the output file for test data predictions')
    args = parser.parse_args()

    # Read the data.
    training_sentences = read_data_file(args.training_data)
    testing_sentences = read_data_file(args.testing_data)

    # Generate Features
    user_mwe_features = MWE_FEATURES
    user_ss_features = generate_ss_features(training_sentences)

    # Build model.
    hmm = HMM(user_mwe_features, user_ss_features)
    hmm.train(training_sentences)

    # Test model.
    hmm.test(testing_sentences)

    # Output predictions.
    write_predictions(testing_sentences, args.predictions_output)
