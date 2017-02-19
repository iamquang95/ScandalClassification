import codecs

if __name__ == "__main__":
    # Config
    inputFilename = "test2.txt"
    startLine = 4806


    outputFilename = "data2.txt"
    writerMode = "a"

    ###################################

    inputFile = codecs.open(inputFilename, "r", "UTF-8")

    with codecs.open(outputFilename, writerMode, "UTF-8") as outputFile:
        curLine = 1
        for line in inputFile:
            if (curLine >= startLine):
                print("%d %s" % (curLine, line[:len(line)-1]))
                label = raw_input("0/1/ESC = ")
                if (label == "ESC"):
                    break
                # HACK because there are 2 classes
                if (label != "1"):
                    label = 0
                outputFile.write("%s %s\n" % (line[:len(line)-1], label))
            curLine += 1

