#!/usr/bin/env python3

"""
A machine learning classifier for identifying minimal semantic units. See the README.md for more information.
"""

from classify import Label

from typing import Dict, List, Mapping, Tuple

FeatureID = int
LabelDict = Dict[Label, int]


class ZeroedLabelDict(LabelDict):
    def __new__(cls):
        return {label: 0 for label in Label}


class FrequencyCounter(Mapping):
    def __init__(self):
        self._frequencies: Dict[Label, int] = ZeroedLabelDict()

    def __str__(self):
        return f'{{{", ".join(f"{label.name}: {count}" for label, count in self._frequencies.items())}}}'

    def __repr__(self):
        return str(self)

    def __iter__(self):
        return iter(self._frequencies)

    def __len__(self) -> int:
        return len(self._frequencies)

    def __getitem__(self, label: Label) -> int:
        return self._frequencies[label]

    def __setitem__(self, key: Label, value: int):
        self._frequencies[key] = value

    @property
    def total_occurrences(self) -> int:
        return sum(self._frequencies.values())

    def probability_of_label(self, label: Label) -> float:
        if self.total_occurrences == 0:
            return 0
        return self._frequencies[label] / self.total_occurrences


class Prediction:
    def __init__(self, predicted_label: Label, actual_label: Label, features: List[FeatureID]):
        self.predicted_label = predicted_label
        self.actual_label = actual_label
        self.features = features


class Evaluation:
    def __init__(self):
        self.correct = ZeroedLabelDict()
        self.predicted = ZeroedLabelDict()
        self.actual = ZeroedLabelDict()
        self.confusion_matrix = {label: ZeroedLabelDict() for label in Label}

    def __str__(self):
        default_len = 10
        longest_label: Label = max(self.correct.keys(), key=lambda l: len(l.name))
        label_len = max(len(longest_label.name), len('label'))
        leader = (
            f'{"Label":<{label_len}}'
            '  '
            f'{"Recall":<{default_len}}'
            '  '
            f'{"Precision":<{default_len}}'
        )
        lines = [leader]
        for label in self.correct:
            recall = round(self.recall(label), default_len - 2)
            precision = round(self.precision(label), default_len - 2)
            lines.append(
                ' '
                f'{label.name:<{label_len}}'
                '  '
                f'{recall:<{default_len}}'
                '  '
                f'{precision:<{default_len}}'
            )
        recall_precision = '\n'.join(lines)
        matrix_string = self._confusion_matrix_to_string()
        return recall_precision + '\n\n' + matrix_string

    def _confusion_matrix_to_string(self) -> str:
        length = 0
        label_length = 0
        for label in Label:
            label_length = max(label_length, len(label.name))
            length = max(length, len(label.name))
            count_dict = self.confusion_matrix[label]
            for count in count_dict.values():
                length = max(length, len(str(count)))
        labels = (f'{label.name:^{length}}' for label in Label)
        leader = (' ' * label_length) + '  ' + '  '.join(labels)
        lines = [leader]
        for label in Label:
            line_parts = [f'{label.name:^{label_length}}']
            count_dict = self.confusion_matrix[label]
            for inner_label in Label:
                count = count_dict[inner_label]
                line_parts.append(f'{count:>{length}}')
            lines.append('  '.join(line_parts))
        return '\n'.join(lines)

    def process_prediction(self, prediction: Prediction):
        actual = prediction.actual_label
        predicted = prediction.predicted_label
        self.confusion_matrix[actual][predicted] += 1
        self.predicted[predicted] += 1
        self.actual[actual] += 1
        if predicted == actual:
            self.correct[predicted] += 1

    def recall(self, label: Label):
        try:
            return self.correct[label] / self.actual[label]
        except ZeroDivisionError:
            return 0

    def precision(self, label: Label):
        try:
            return self.correct[label] / self.predicted[label]
        except ZeroDivisionError:
            return 0


class MWE:
    def __init__(self, threshold_multiplier=1.0, base_label_tag=0):
        self.dtm = threshold_multiplier
        self.base_label = Label(base_label_tag)
        self.frequencies: Dict[FeatureID, FrequencyCounter] = {}
        self.predictions: List[Prediction] = []
        self.total_frequencies = FrequencyCounter()

    def train(self, training_datafile: str):
        with open(training_datafile) as df:
            for line in df:
                label, features = self._read_line(line)
                for feature in features:
                    frequency = self.frequencies.get(feature)
                    if frequency is None:
                        frequency = FrequencyCounter()
                        self.frequencies[feature] = frequency
                    frequency[label] += 1
                    self.total_frequencies[label] += 1

    def test(self, testing_datafile: str):
        with open(testing_datafile) as df:
            for line in df:
                actual_label, features = self._read_line(line)
                prediction = self._predict(actual_label, features)
                self.predictions.append(prediction)

    def _predict(self, actual_label: Label, features: List[FeatureID]) -> Prediction:
        probabilities = self._compute_probabilities(features)
        # Now that we have the accumulated the probabilities of each label for this word, identify which is most likely.
        # We compare non-BASE_LABEL labels against their distributions within the entire training set. If the
        # probability is at or above the given threshold multiplier (default: 1.0), consider it a "likely" occurrence.
        # Due to the possibility of multiple labels being above the necessary minimum, compare their respective
        # differences to find which is *most* likely.
        likely_labels: List[Tuple[Label, float]] = []  # A list of tuples mapping labels to their threshold difference.
        for label in Label:
            if label == self.base_label:
                # Skip the "base" label.
                continue
            difference = (self.total_frequencies.probability_of_label(label) * self.dtm) - probabilities[label]
            if difference >= 0:
                pair = (label, difference)
                likely_labels.append(pair)
        # Check if there were any labels above their thresholds.
        if likely_labels:
            # There was at least one likely label, so pick the label with the greatest difference from the baseline.
            likely_label = max(likely_labels, key=lambda p: p[1])[0]
        else:
            # No labels were considered likely, so just use the base label.
            likely_label = self.base_label
        return Prediction(likely_label, actual_label, features)

    def _compute_probabilities(self, features: List[FeatureID]) -> LabelDict:
        feature_count = len(features)
        probabilities = ZeroedLabelDict()
        for feature in features:
            counter = self.frequencies.get(feature)
            if counter is None:
                # This feature is unique to the test data -- it is not found in the training data at all.
                # So just use the total values as inspiration.
                counter = self.total_frequencies
            # Find the probabilities of each label for this feature.
            for label in counter:
                prob_of_label_for_feature = counter.probability_of_label(label)
                probabilities[label] += prob_of_label_for_feature / feature_count
        return probabilities

    def evaluate(self) -> Evaluation:
        evaluation = Evaluation()
        for prediction in self.predictions:
            evaluation.process_prediction(prediction)
        return evaluation

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
    parser.add_argument('--multiplier', '-m', type=float, default=1.0)
    parser.add_argument('--base_label_tag', '-b', type=int, default=0)
    args = parser.parse_args()

    mwe = MWE(args.multiplier, args.base_label_tag)

    # Train the MWE recognizer.
    mwe.train(args.training_datafile)

    # Now test it!
    mwe.test(args.testing_datafile)

    # And evaluation here.
    result = mwe.evaluate()

    # Print results.
    print(result)
