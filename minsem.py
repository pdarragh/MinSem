#!/usr/bin/env python3

from collections import OrderedDict


class DataToken:
    def __init__(self, line: str):
        parts = line.split('\t')
        (
            self.offset,
            self.word,
            self.lowercase_lemma,
            self.pos_tag,
            self.mwe_tag,
            self.parent_offset,
            self.strength,
            self.supersense,
            self.sentence_id
        ) = list(map(lambda x: None if not x else x, parts))
        self.parent_offset = None if self.parent_offset == '0' else int(self.parent_offset)


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
    data_sentences = {}
    with open(datafile) as df:
        sentence = DataSentence()
        for line in df:
            line = line.strip(' \n')
            if not line:
                # An empty line indicates the end of a sentence.
                if sentence:
                    data_sentences[sentence.sentence_id] = sentence
                sentence = DataSentence()
            else:
                # A new token to be accumulated!
                token = DataToken(line)
                sentence.append(token)
        # Check if there is a valid sentence; this would happen if the file does not end in a newline.
        if sentence:
            data_sentences[sentence.sentence_id] = sentence
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
