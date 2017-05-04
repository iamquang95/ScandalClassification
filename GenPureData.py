#!/usr/bin/python

import sys

import codecs

import TextProcessUtils

startDataLine = int(sys.argv[1])
endDataLine = int(sys.argv[2])
foUnTokData = sys.argv[3] # "UnTokData.txt"
fiFullTextFilename = "FullData.txt"

def genUnTokData():
    print startDataLine
    print endDataLine
    print(foUnTokData)
    fiFullText = codecs.open(fiFullTextFilename, "r", 'UTF-8')
    data = fiFullText.read().replace(u'\u2028','').replace(u'\u2029','').splitlines()
    print(len(data))
    fiFullText.close()
    fo = codecs.open(foUnTokData, "w+", "UTF-8")

    index = 1
    lineType = 0
    title = ""
    summary = ""

    for (i, line) in enumerate(data):
        lineType = i%3
        if (lineType == 0):
            if (index >= endDataLine):
                break
            title = TextProcessUtils.removeEndline(line)
            index += 1
        elif (lineType == 1):
            summary = TextProcessUtils.getSummary(line)
            if (index > startDataLine):
                fo.write("%s. %s\n" % (title, summary))
        elif (lineType == 2):
            if (line[0] != '-'):
                print("vkl %s %s" % (i, line))
                break

    fo.close()

if __name__ == "__main__":
    genUnTokData()
