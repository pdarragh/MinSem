# MinSem

A solution for [SemEval 2016 Task 10](http://dimsum16.github.io/).

This project was done for Ellen Riloff's 2017 spring semester Information Extraction course. I am doing this project on
my own.

## Purpose

MinSem's goal is to identify the usage of certain semantic classes in English sentences. For example (from the DiMSUM
webpage), given the following POS-tagged sentence:

> I`PRON` googled`VERB` restaurants`NOUN` in`ADP` the`DET` area`NOUN` and`CONJ` Fuji`PROPN` Sushi`PROPN` came`VERB`
> up`ADV` and`CONJ` reviews`NOUN` were`VERB` great`ADJ` so`ADV` I`PRON` made`VERB` a`DET` carry`VERB` out`ADP`
> order`NOUN`

the goal is to predict the following representation:

> I googled`v.communication` restaurants`GROUP` in the area`n.location` and Fuji`_`Sushi`n.group`
> came`_`up`v.communication` and reviews`n.communication` were`v.stative` great so I made`_` a carry`_`out`v.possession`
> `_`order`v.communication`

The second representation includes information about *multi-word expressions*, or MWEs, and *supersenses*. An MWE is a
group of tokens which carry a strong semantic link to one another. A supersense 

## Inputs

`minsem.py` takes two positional arguments: the path to a file containing training data, and the path to a file
containing testing data. The files must be tab-delimited and contain the following nine fields on each line:

1. token offset within the sentences
2. word literal
3. lowercase form of word literal
4. part-of-speech tag
5. multi-word expression (MWE) tag
6. offset from parent token within MWE (if applicable)
7. strength level (if applicable)
8. supersense label (if applicable)
9. sentence identification string

Fields may be left blank if unused. Blank lines will be skipped, but are not assumed to separate sentences.
