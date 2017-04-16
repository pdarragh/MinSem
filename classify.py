#!/usr/bin/env python3

from collections import OrderedDict
from enum import Enum
from itertools import chain
from typing import Iterable, List, Mapping, Sequence, Tuple


class Label(Enum):
    O = 0
    B = 1
    I = 2


def label_from_string(s: str) -> Label:
    for label in Label:
        if s.upper().replace('-', '_') == label.name:
            return label
    else:
        raise ValueError(f"invalid label: {s}")


class DataToken:
    def __init__(self, offset, word, lowercase, pos_tag, mwe_tag, parent_offset, sentence_id=''):
        self.offset = int(offset)
        self.word = word
        self.lowercase_lemma = lowercase
        self.pos_tag = pos_tag
        self.mwe_tag = label_from_string(mwe_tag)
        self.parent_offset = 0 if not parent_offset else int(parent_offset)
        self.sentence_id = sentence_id

    def __repr__(self):
        return '\t'.join(map(lambda w: str(w) if w is not None else '', [
            self.offset, self.word, self.lowercase_lemma, self.pos_tag, self.mwe_tag, self.parent_offset,
            self.sentence_id
        ]))

    def __str__(self):
        return repr(self)


def make_phi(sentence_id='') -> DataToken:
    return DataToken(0, 'PHI', 'PHI', 'PHIPOS', 'O', '', sentence_id)


def make_omega(sentence_id='') -> DataToken:
    return DataToken(0, 'OMEGA', 'OMEGA', 'OMEGAPOS', 'O', '', sentence_id)


def make_unknown(sentence_id='') -> DataToken:
    return DataToken(0, 'UNKWORD', 'UNKWORD', 'UNKPOS', 'O', '', sentence_id)


PHI = make_phi()
OMEGA = make_omega()
UNKNOWN = make_unknown()


class DataSentence(Sequence):
    def __init__(self):
        self._tokens: List[DataToken] = list()
        self.sentence_id = None

    def __iter__(self) -> Iterable[DataToken]:
        return iter(self._tokens)

    def __getitem__(self, item: int) -> DataToken:
        # The indices are adjusted to match token offsets (i.e. the "first" element in the list has index 1).
        if item < 1:
            return make_phi(self.sentence_id)
        elif item > len(self._tokens):
            return make_omega(self.sentence_id)
        else:
            return self._tokens[item - 1]  # Adjust the offset to match the true index.

    def __bool__(self) -> bool:
        return bool(self._tokens)

    def __len__(self) -> int:
        return len(self._tokens)

    def append(self, token: DataToken):
        if not self.sentence_id:
            self.sentence_id = token.sentence_id
        else:
            if token.sentence_id != self.sentence_id:
                raise ValueError(f"sentence IDs do not match: {self.sentence_id} (sent) != {token.sentence_id} (token)")
        self._tokens.append(token)

    @property
    def sentence(self) -> str:
        return ' '.join(map(lambda t: t.word, self._tokens))


class FeatureSet(Mapping):
    def __init__(self):
        self._features = {}

    def __len__(self):
        return len(self._features)

    def __iter__(self):
        return iter(self._features)

    def __getitem__(self, feature):
        if feature not in self._features:
            raise KeyError(f"no such feature in feature set: {feature}")
        return self._features[feature]

    def add(self, feature):
        if feature in self._features:
            return self._features[feature]
        self._features[feature] = len(self) + 1
        return len(self)

    def get(self, key, default=None):
        return self._features.get(key, default)


class FeatureVector:
    def __init__(self, label: Label):
        self.label = label
        self._features = set()

    def add(self, feature: int):
        self._features.add(feature)

    def __str__(self):
        return f'{self.label.value} {" ".join((str(feature) for feature in sorted(self._features)))}'


class Classifier:
    def __init__(self, feature_words: Iterable[str], feature_poses: Iterable[str]):
        self.feature_set = FeatureSet()
        self._generate_features(feature_words, feature_poses)
        self.reset_vectors()

    def reset_vectors(self):
        self.training_vectors = list()
        self.testing_vectors = list()

    @staticmethod
    def _current_word_feature(word):
        return f'curr-word-{word}'

    @staticmethod
    def _previous_word_feature(word):
        return f'prev-word-{word}'

    @staticmethod
    def _next_word_feature(word):
        return f'next-word-{word}'

    @staticmethod
    def _previous_pos_feature(pos):
        return f'prev-pos-{pos}'

    @staticmethod
    def _next_pos_feature(pos):
        return f'next-pos-{pos}'

    def _generate_features(self, words: Iterable[str], poses: Iterable[str]):
        pseudos = [PHI, OMEGA, UNKNOWN]
        for word in chain(words, map(lambda p: p.lowercase_lemma, pseudos)):
            self._generate_features_for_word(word)
        for pos in chain(poses, map(lambda p: p.pos_tag, pseudos)):
            self._generate_features_for_pos(pos)

    def _generate_features_for_word(self, word: str):
        self.feature_set.add(self._current_word_feature(word))
        self.feature_set.add(self._previous_word_feature(word))
        self.feature_set.add(self._next_word_feature(word))

    def _generate_features_for_pos(self, pos: str):
        self.feature_set.add(self._previous_pos_feature(pos))
        self.feature_set.add(self._next_pos_feature(pos))

    def train(self, sentences: Iterable[DataSentence]):
        for sentence in sentences:
            for curr_word in sentence:
                # Create a feature vector for each word in the training set.
                vector = FeatureVector(curr_word.mwe_tag)
                # Create and add a feature for the current word.
                curr_word_feature = self._current_word_feature(curr_word.lowercase_lemma)
                vector.add(self.feature_set[curr_word_feature])
                # Create features for the previous word, next word, previous POS, and next POS.
                prev_word = sentence[curr_word.offset - 1]
                next_word = sentence[curr_word.offset + 1]
                prev_word_feature = self._previous_word_feature(prev_word.lowercase_lemma)
                next_word_feature = self._next_word_feature(next_word.lowercase_lemma)
                prev_pos_feature = self._previous_pos_feature(prev_word.pos_tag)
                next_pos_feature = self._next_pos_feature(next_word.pos_tag)
                # Add those features to the vector.
                vector.add(self.feature_set[prev_word_feature])
                vector.add(self.feature_set[next_word_feature])
                vector.add(self.feature_set[prev_pos_feature])
                vector.add(self.feature_set[next_pos_feature])
                # Add the vector to the list of training vectors.
                self.training_vectors.append(vector)

    def test(self, sentences: Iterable[DataSentence]):
        # Set up the default labels.
        default_curr_word_label = self.feature_set[self._current_word_feature(UNKNOWN.lowercase_lemma)]
        default_prev_word_label = self.feature_set[self._previous_word_feature(UNKNOWN.lowercase_lemma)]
        default_next_word_label = self.feature_set[self._next_word_feature(UNKNOWN.lowercase_lemma)]
        default_prev_pos_label = self.feature_set[self._previous_pos_feature(UNKNOWN.pos_tag)]
        default_next_pos_label = self.feature_set[self._next_pos_feature(UNKNOWN.pos_tag)]
        for sentence in sentences:
            for curr_word in sentence:
                # Create a feature vector for each word in the test set..
                vector = FeatureVector(curr_word.mwe_tag)
                # Create and add a feature for the current word.
                curr_word_feature = self._current_word_feature(curr_word.lowercase_lemma)
                curr_word_feature_label = self.feature_set.get(curr_word_feature, default=default_curr_word_label)
                vector.add(curr_word_feature_label)
                # Create features for the previous word, next word, previous POS, and next POS.
                prev_word = sentence[curr_word.offset - 1]
                next_word = sentence[curr_word.offset + 1]
                prev_word_feature = self._previous_word_feature(prev_word.lowercase_lemma)
                next_word_feature = self._next_word_feature(next_word.lowercase_lemma)
                prev_pos_feature = self._previous_pos_feature(prev_word.pos_tag)
                next_pos_feature = self._next_pos_feature(next_word.pos_tag)
                prev_word_feature_label = self.feature_set.get(prev_word_feature, default=default_prev_word_label)
                next_word_feature_label = self.feature_set.get(next_word_feature, default=default_next_word_label)
                prev_pos_feature_label = self.feature_set.get(prev_pos_feature, default=default_prev_pos_label)
                next_pos_feature_label = self.feature_set.get(next_pos_feature, default=default_next_pos_label)
                # Add those features to the vector.
                vector.add(prev_word_feature_label)
                vector.add(next_word_feature_label)
                vector.add(prev_pos_feature_label)
                vector.add(next_pos_feature_label)
                # Add the vector to the list of testing vectors.
                self.testing_vectors.append(vector)

    @staticmethod
    def _write_vectors_to_file(vectors: Iterable[FeatureVector], filename: str):
        with open(filename, 'w') as f:
            for vector in vectors:
                f.write(f'{vector}\n')

    def write_training_vectors_to_file(self, filename: str):
        self._write_vectors_to_file(self.training_vectors, filename)

    def write_testing_vectors_to_file(self, filename: str):
        self._write_vectors_to_file(self.testing_vectors, filename)

    def write_features_to_file(self, filename: str):
        with open(filename, 'w') as f:
            for key, value in self.feature_set.items():
                f.write(f'{value:<{len(str(len(self.feature_set)))}} : {key}\n')


def read_data_file(datafile: str) -> Iterable[DataSentence]:
    data_sentences = []
    with open(datafile) as df:
        current_sentence = DataSentence()
        for line in df:
            if not line or line.isspace():
                if current_sentence:
                    data_sentences.append(current_sentence)
                current_sentence = DataSentence()
            else:
                # Strip extraneous whitespace from the line, then split it on the tabs. There should be exactly enough
                # fields to fill the DataToken.
                token = DataToken(*(line.strip(' \n').split('\t')))
                current_sentence.append(token)
    if current_sentence:
        data_sentences.append(current_sentence)
    return data_sentences


def get_distinct_words_and_poses_from_sentences(sentences: Iterable[DataSentence]) -> Tuple[List[str], List[str]]:
    words = OrderedDict()
    poses = OrderedDict()
    for sentence in sentences:
        for token in sentence:
            words[token.lowercase_lemma] = 1
            poses[token.pos_tag] = 1
    words_list = list(words.keys())
    poses_list = list(poses.keys())
    return words_list, poses_list


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('training_data', help='the datafile to train with')
    parser.add_argument('testing_data', help='the datafile to test with')
    parser.add_argument('training_output', help='the output file for training data classification')
    parser.add_argument('testing_output', help='the output file for testing data classification')
    parser.add_argument('--feature_output', help='the output file for the entire feature set')
    args = parser.parse_args()

    # Read the data.
    training_sentences = read_data_file(args.training_data)
    testing_sentences = read_data_file(args.testing_data)

    distinct_words, distinct_poses = get_distinct_words_and_poses_from_sentences(training_sentences)

    classifier = Classifier(distinct_words, distinct_poses)

    classifier.train(training_sentences)
    classifier.test(testing_sentences)

    classifier.write_training_vectors_to_file(args.training_output)
    classifier.write_testing_vectors_to_file(args.testing_output)

    if args.feature_output:
        classifier.write_features_to_file(args.feature_output)
