import re
import TextProcessUtils
import codecs
import collections


def generateBOW(wordToInt, line):
    wordList = TextProcessUtils.lineToWords(line)
    bagOfWords = {}
    for word in wordList:
        if (bagOfWords.has_key(wordToInt[word])):
            bagOfWords[wordToInt[word]] += 1
        else:
            bagOfWords[wordToInt[word]] = 1
    return bagOfWords


def bowToText(bow):
    result = ""
    orderedBOW = collections.OrderedDict(sorted(bow.items()))
    for key in orderedBOW.keys():
        result += "%s:%s " % (key, orderedBOW[key])
    return result

def generateWordsMap(lines):
    wordsMap = {}
    numb = 0
    for line in lines:
        wordList = TextProcessUtils.lineToWords(line)
        for word in wordList:
            if (not wordsMap.has_key(word)):
                numb += 1
                wordsMap[word] = numb
    return wordsMap

def genVectorFilesFromTextFiles(inFiName, ouFiName):
    fi = codecs.open(inFiName, "r", "UTF-8")
    fo = codecs.open(ouFiName, "w", "UTF-8")

    data = [line for line in fi]
    lines = [TextProcessUtils.getTitle(line) for line in data]
    labels = [TextProcessUtils.getLabel(line) for line in data]

    wordsMap = generateWordsMap(lines)

    index = 0
    dataSize = len(data)

    for line, label in zip(lines, labels):
        bow = generateBOW(wordsMap, line)
        fo.write("%s %s\n" %(label, bowToText(bow)))

    fo.close()


    print("Successfully created vector file")

if __name__ == "__main__":
    # # Unit test for bowToText
    # bow = {"123":"1", "234":"2"}
    # print(bowToText(bow))

    # # Unit test for generateWordsMap
    # lines = ["abc cba abc ccc", "cba abc def"]
    # wordsMap = generateWordsMap(lines)
    # for key in wordsMap.keys():
    #     print("%s:%s" % (key, wordsMap[key]))

    # # Unit test for generateBOW
    # print(bowToText(generateBOW(wordsMap, "cba ccc cba abc abc def abc")))

    genVectorFilesFromTextFiles("test.tagged", "test.txt")
    genVectorFilesFromTextFiles("train.tagged", "train.txt")






