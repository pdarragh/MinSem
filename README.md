# MWE Predict

This project was done for Ellen Riloff's 2017 spring semester Information Extraction course. I am doing this project on
my own. It is based on a task given for [SemEval 2016 Task 10](http://dimsum16.github.io/).

## Purpose

This project is meant to identify multi-word expressions in English sentences.
[From Wikipedia](https://en.wikipedia.org/wiki/Multiword_expression):

> A **multiword expression (MWE)**, also called phraseme, is a lexeme made up of a sequence of two or more lexemes that
> has properties that are not predictable from the properties of the individual lexemes or their normal mode of
> combination.
> 
> For a shorter definition, MWEs can be described as "idiosyncratic interpretations that cross word boundaries (or
> spaces)". (Sag *et al.*, 2002: 2).

## Walkthrough

Following is an explanation of the workings of the modules in this project, but here is a brief overview:

* `relabel.py` — takes the original data from the DiMSUM task and relabels it
* `classify.py` — takes in the relabeled data and classifies the features for analysis
* `mwe.py` — takes in the classified data to train and then run prediction on the testing data

### Relabeling

I am using the data sets provided in the SemEval task upon which this project is based, but some relabeling was
necessary to facilitate the narrower scope of this project. The original data files are available from
[this repository](https://github.com/dimsum16/dimsum-data/tree/3af1dd34ce49783e0ab99f31e989dd6171f75433) (which has been
added as a submodule to the current repository).

`relabel.py` takes in as input a tab-delimited file, where each line contains the following nine fields:

1. token offset within the sentences
2. word literal
3. lowercase form of word literal
4. part-of-speech tag
5. multi-word expression (MWE) tag
6. offset from parent token within MWE (if applicable)
7. strength level (if applicable)
8. supersense label (if applicable)
9. sentence identification string

Fields may be left blank if unused. Blank lines will be skipped, but are not assumed to separate sentences. An example
of input data might be:

```
1	My	my	PRON	O	0		ewtb.r.001325.2
2	8	8	NUM	O	0		ewtb.r.001325.2
3	year	year	NOUN	B	0	n.person	ewtb.r.001325.2
4	old	old	ADJ	I	3			ewtb.r.001325.2
5	daughter	daughter	NOUN	O	0		n.person	ewtb.r.001325.2
6	loves	love	VERB	O	0		v.emotion	ewtb.r.001325.2
7	this	this	DET	O	0			ewtb.r.001325.2
8	place	place	NOUN	O	0		n.location	ewtb.r.001325.2
9	.	.	PUNCT	O	0			ewtb.r.001325.2
```

(This segment corresponds to the sentence labeled `ewtb.r.001325.2` in the original data: "My 8 year old daughter loves
this place.")

The output is a new tab-delimited file with only the following fields:

1. token offset within sentence
2. word
3. lowercase lemma
4. part of speech
5. MWE tag **(revised)**
6. offset from parent token (if applicable)
7. sentence ID

The key difference is that the MWE tags have been revised by the relabeler. Whereas the initial data specifies MWEs
which occur inside other MWEs, I choose not to recognize this distinction. The previous six labels which were previously
allowed for the MWE tag have been stripped down to only three: B, I, and O (representing words which occur at the
Beginning, Inside, and Outside of a multi-word expression, respectively).

An example output of relabeling the example segment is:

```
1	My	my	PRON	O	0	
2	8	8	NUM	O	0	
3	year	year	NOUN	B	0	
4	old	old	ADJ	I	3	ewtb.r.001325.2
5	daughter	daughter	NOUN	O	0	ewtb.r.001325.2
6	loves	love	VERB	O	0	ewtb.r.001325.2
7	this	this	DET	O	0	ewtb.r.001325.2
8	place	place	NOUN	O	0	ewtb.r.001325.2
9	.	.	PUNCT	O	0	ewtb.r.001325.2
```

### Feature Classification

The relabeled data files are taken from the previous process and fed into `classify.py`. The classifier creates a list
of features for each word in the input. The features and BIO labels are assigned integer values, which will be written
out.

Each word is given values to describe the following five features:

1. The current word
2. The previous word
3. The previous word's part-of-speech tag
4. The next word
5. The next word's part-of-speech tag

This classification is applied to both the training and the test data. The classified data for the example segment from
the previous section is:

```
0 1408 8331 23966 23989 24008
0 549 1409 8329 23975 23996
1 547 6039 8330 23987 23988
2 548 5982 6037 23974 23975
0 396 5980 6038 23977 23986
0 394 546 5981 23974 23991
0 395 544 4278 23975 23976
0 156 545 4276 23983 23990
0 154 4277 23970 23974 24011
```

### Prediction

Classified data files are passed to `mwe.py`. This is where the statistical model is created to predict MWE membership
for words in the test data.

Running prediction over the example segment from the previous sections yields the following output:

```
Label  Recall      Precision 
 O      0.0         0.0       
 B      0.0         0.0       
 I      0.0         0.0       

   O  B  I
O  0  3  4
B  0  0  1
I  1  0  0
```

In this short example, none of the words were correctly labeled. In fact, only one word was labeled as being "Outside"
of a multi-word expression, when in truth only two words are considered part of an MWE.
