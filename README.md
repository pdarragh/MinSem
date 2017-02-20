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
