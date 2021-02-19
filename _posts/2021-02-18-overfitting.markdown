---
toc: False
layout: post
description: Explanation from my experience in Fraud Detection competition
categories: [markdown, posts]
title: How to know if you are actually overfitting
permalink: /:title.html
image: images/fraud-detection/overfitting.png
---

## **Preventing overfitting**

There are usually two reasons why the Training score and Test score
differ:

1. Overfitting

2. Out of Domain Data 

	i.e., test and train data are from different times, or different
    clients etc...

There is a nice trick to see what is causing the difference in scores
between the training and the test data: Determining OOF (out-of-fold)
scores. The OOF score is basically a score on unseen data but within
the training data domain itself. It nicely controls for the effect of
"out of domain" data. So,

- OOF (out-of-fold) scores < Train scores ==> **Overfitting**

- Test scores < OOF scores ==> **Out-of-domain data**

**Example of only over-fitting**

From the beginning I have suffered mainly with overfitting rather than
the out-of-domain data issue. My `OOF<<Train` (indicating overfitting)
and `Test>OOF` (not indicating "out-of-domain" issue):

|              | train | OOF    | test (public) |
|--------------|-------|--------|---------------|
| baseline AUC | 0.99  | 0.9292 | 0.9384        |

**Example of both over-fitting and out of domain issue**

In another case, where I accidentally changed some values of columns
in the test dataset as NaNs, I saw the following. Here `OOF<<Train`
(indicating overfitting) and also `Test<<OOF` (indicating
"out-of-domain" issue):

|     | train  | OOF    | test (public) |
|-----|--------|--------|---------------|
| AUC | 0.9971 | 0.9426 | 0.9043        |

Looking at the important AV columns, and probing into those columns
allowed me to fix the issue.

## References

1. [fastai tabular notebook](https://github.com/fastai/fastbook/blob/master/09_tabular.ipynb)
1. [Data description](https://www.kaggle.com/c/ieee-fraud-detection/discussion/101203)
1. [Plots and much more for many features](https://www.kaggle.com/alijs1/ieee-transaction-columns-reference)

