#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy
import codecs

import TextProcessUtils
import SMOTE

RANDOMSEED = 1508


if __name__ == "__main__":
    f = codecs.open("validateData.txt", "r", "utf-8")
    g = codecs.open("validateData2.txt", "r", "utf-8")

    data1 = [line for line in f]
    data2 = [line for line in g]

    train_corpus = [TextProcessUtils.getTitle(line) for line in data1]
    train_labels = [TextProcessUtils.getLabel(line) for line in data2]

    matrix = [[0, 0], [0, 0]]

    for (x1, x2) in zip(data1, data2):
        l1 = TextProcessUtils.getLabel(x1)
        l2 = TextProcessUtils.getLabel(x2)
        if (l1 == 1):
            if (l2 == 1):
                matrix[0][0] += 1
            else:
                print(TextProcessUtils.getTitle(x1))
                matrix[0][1] += 1
        else:
            if (l2 == 1):
                print(TextProcessUtils.getTitle(x2))
                matrix[1][0] += 1
            else:
                matrix[1][1] += 1
    print("<<<<<<<<<<<<<<<<<")
    a = matrix[0][0]
    b = matrix[0][1]
    c = matrix[1][0]
    d = matrix[1][1]
    total = (a+b+c+d)
    print("%s %s\n%s %s" % (a, b, c, d))
    p0 = 1.0*(a + d)/total
    pYes = 1.0*(a+b)/total * (a+c)/total
    pNo = 1.0*(c+d)/total * (b+d)/total
    pe = pYes + pNo
    k = 1.0*(p0 - pe)/(1-pe)
    print(k)




