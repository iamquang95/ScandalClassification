import codecs

import TextProcessUtils

startDataLine = 5000
endDataLine = 20000
foUnTokData = "UnTokData.txt"
fiFullTextFilename = "FullData.txt"

def genUnTokData():
    fiFullText = codecs.open(fiFullTextFilename, "r", "UTF-8")
    fo = codecs.open(foUnTokData, "w", "UTF-8")

    index = 1
    lineType = 0
    title = ""
    summary = ""

    for line in fiFullText:
        if (lineType == 0):
            if (index >= endDataLine):
                break
            title = TextProcessUtils.removeEndline(line)
            index += 1
        elif (lineType == 1):
            summary = TextProcessUtils.getSummary(line)
            if (index > startDataLine):
                fo.write("%s. %s\n" % (title, summary))
        lineType = (lineType + 1) % 3

    fo.close()

if __name__ == "__main__":
    genUnTokData()
