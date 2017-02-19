import codecs
import re
from sets import Set

import TextProcessUtils

fiStopWordsFile = "vietnamese-stopwords-dash.txt"
fiInputFile = "test.tagged"
foOutputFile = "test.tagged2"

def readStopWord(filename):
    fiStopWords = codecs.open(filename, "r", "UTF-8")
    stopWords = []
    for line in fiStopWords:
        stopWords.append(TextProcessUtils.removeEndline(line))
    return Set(stopWords)

def removeStopWord(stopWords, line):
    result = ' '.join([word for word in line.split() if word not in stopWords])
    return result

def removeStopWordInFile(stopwordFilename, filename, outputFilename):
    fi = codecs.open(filename, "r", "UTF-8")
    fo = codecs.open(outputFilename, "w", "UTF-8")

    stopWords = readStopWord(stopwordFilename)

    for line in fi:
        fo.write("%s\n" % removeStopWord(stopWords, line))

    fo.close()

if __name__ == "__main__":
    removeStopWordInFile(fiStopWordsFile, "test.tagged", "test.tagged2")
    removeStopWordInFile(fiStopWordsFile, "train.tagged", "train.tagged2")





