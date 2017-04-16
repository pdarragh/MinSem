#!/usr/bin/env python3
"""
The relabeler reads data files and outputs new data files with different information.

The input files must contain one word per line, with sentences separated by blank lines. Each word is annotated with the
following tab-separated fields:

1. token offset within sentence
2. word
3. lowercase lemma
4. part of speech
5. multi-word expression (MWE) tag
6. offset from parent token (if inside an MWE; blank otherwise)
7. strength level
8. supersense label (if applicable)
9. sentence ID

The input data uses the following six tags for MWE labeling:

O   - not part of or inside any MWE
o   - not part of an MWE, but inside one
B   - first token of an MWE, not inside another MWE
b   - first token of an MWE occurring inside another MWE
I   - token continuing an MWE, but not inside another MWE
i   - token continuing an MWE which occurs inside another MWE

This script will output the same number of lines, with one word per line, but with the following tab-separated fields:

1. token offset within sentence
2. word
3. lowercase lemma
4. part of speech
5. MWE tag (revised)
6. offset from parent token (if applicable)
7. sentence ID

The revised MWE tags are:

O   - not part of an MWE
B   - first token of an MWE
I   - token continuing an MWE

In this annotation scheme, there are only top-level MWEs.
"""

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='data file to read')
    parser.add_argument('output_file', help='location to output revised data to')
    args = parser.parse_args()

    print(f"Relabeling input data from '{args.input_file}' to '{args.output_file}'...")
    with open(args.output_file, 'w') as outfile:
        with open(args.input_file) as infile:
            for line in infile:
                if line and not line.isspace():
                    try:
                        off, word, lowlem, pos, mwe, paroff, strength, sup_and_id = line.strip(' \n').split('\t', 7)
                        if len(sup_and_id.split('\t')) == 1:
                            supsen = sup_and_id
                            sentid = ''
                        else:
                            supsen, sentid = sup_and_id.split('\t')
                    except ValueError:
                        print(f"Error with line: {repr(line)}")
                        raise
                    if mwe.islower():
                        # Anything which occurs inside an MWE is considered part of that MWE.
                        mwe = 'I'
                    outfile.write('\t'.join([off, word, lowlem, pos, mwe, paroff, sentid]))
                outfile.write('\n')
    print("Relabeling complete.")
