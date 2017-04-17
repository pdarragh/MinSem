#!/usr/bin/env python3

import argparse
import subprocess

from itertools import combinations
from sys import executable

parser = argparse.ArgumentParser()
parser.add_argument('classifier', help='path to the classifier module')
parser.add_argument('feature_count', help='the number of features', type=int)
parser.add_argument('mwe', help='path to the MWE module')
parser.add_argument('training_data', help='the datafile to train with')
parser.add_argument('testing_data', help='the datafile to test with')
parser.add_argument('training_output', help='the output file for training data classification')
parser.add_argument('testing_output', help='the output file for testing data classification')
parser.add_argument('--multiplier', '-m', type=float, default=1.0)
args = parser.parse_args()

for count in range(args.feature_count + 1):
    for combination in combinations(range(args.feature_count), count):
        classify = [
            executable,
            args.classifier, args.training_data, args.testing_data, args.training_output, args.testing_output
        ]
        for suppressed_feature_number in combination:
            classify.append('-s')
            classify.append(str(suppressed_feature_number))
        print(f"Classifying while suppressing features: {combination}")
        print()
        subprocess.run(classify)
        mwe = [
            executable,
            args.mwe, args.training_output, args.testing_output,
            '-m', args.multiplier
        ]
        process = subprocess.run(mwe, stdout=subprocess.PIPE)
        print(process.stdout.decode('utf-8'))
        print('-' * 80)
