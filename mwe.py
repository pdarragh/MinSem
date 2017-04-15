#!/usr/bin/env python3

"""
A machine learning classifier for identifying minimal semantic units. See the README.md for more information.
"""

from enum import Enum
from typing import Dict, List, Tuple

FeatureID = int


class Label(Enum):
    O = 0
    B = 1
    I = 2


class Frequency:
    def __init__(self):
        self._frequencies: Dict[Label, int] = {label: 0 for label in Label}

    def __getitem__(self, label: Label) -> int:
        return self._frequencies[label]

    def __setitem__(self, key: Label, value: int):
        self._frequencies[key] = value

    @property
    def total_occurrences(self) -> int:
        return sum(self._frequencies.values())


class Prediction:
    def __init__(self, line_number: int, predicted_label: Label, actual_label: Label, features: List[FeatureID]):
        self.line_number = line_number
        self.predicted_label = predicted_label
        self.actual_label = actual_label
        self.features = features


class MWE:
    def __init__(self):
        self.frequencies: Dict[FeatureID, Frequency] = {}

    def train(self, training_datafile: str):
        with open(training_datafile) as df:
            for line in df:
                label, features = self._read_line(line)
                for feature in features:
                    frequency = self.frequencies.get(feature)
                    if frequency is None:
                        frequency = Frequency()
                        self.frequencies[feature] = frequency
                    frequency[label] += 1

    def test(self, testing_datafile: str):
        pass

    @staticmethod
    def _read_line(line: str) -> Tuple[Label, List[FeatureID]]:
        # Lines are assumed to be written in the form:
        #   <label> <feature_1> <feature_2> ...
        parts = [int(part) for part in line.strip().split()]
        label = Label(parts[0])
        features = parts[1:]
        return label, features


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('training_datafile')
    parser.add_argument('testing_datafile')
    args = parser.parse_args()

    mwe = MWE()

    # Train the MWE recognizer.
    mwe.train(args.training_datafile)

    # Now test it!
    mwe.test(args.testing_datafile)
