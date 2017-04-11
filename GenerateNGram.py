import TextProcessUtils

import codecs

def generateNGram(n, line):
    words = TextProcessUtils.lineToWords(line)
    nGramWords = []
    # for iGram in xrange(1, n+1):
    #     for i in xrange(0, len(words) - iGram):
    #         result = ""
    #         for j in xrange(i, i + iGram - 1):
    #             result = result + words[i] + "_"
    #         result = result + words[i+iGram-1]
    #         nGramWords.append(result)
    # return nGramWords
    for i in xrange(0, len(words)):
        for iGram in xrange(1, n+1):
            result = ""
            if (i+iGram-1 >= len(words)):
                continue
            for j in xrange(i, i + iGram - 1):
                result = result + words[j] + "_"
            result = result + words[i+iGram-1]
            nGramWords.append(result)
    return nGramWords

def nGramWordsToLine(nGramWords):
    result = ""
    for word in nGramWords:
        result = result + word + " "
    return result

def generateNGramFromFile(n, fiName, foName):
    fi = codecs.open(fiName, "r", "UTF-8")
    fo = codecs.open(foName, "w", "UTF-8")

    for line in fi:
        sentence = TextProcessUtils.getTitle(line)
        label = TextProcessUtils.getLabel(line)
        fo.write("%s %s\n" % (nGramWordsToLine(generateNGram(n, sentence)), label))

    fo.close()

    print("Done created nGram file")

if __name__ == "__main__":
    generateNGramFromFile(3, "test.tagged", "nGram_data.txt")


