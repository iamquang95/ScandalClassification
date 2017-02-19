import codecs

if __name__ == "__main__":

    inputFilename = "data2.txt"

    f = codecs.open(inputFilename, "r", "UTF-8")

    number1Labels = 0
    numberSamples = 0

    for line in f:
        leng = len(line)
        numberSamples += 1
        label = line[leng-2:leng-1]
        if (label == "1"):
            number1Labels += 1
            print line


    print("%d / %d" % (number1Labels, numberSamples))
    print(100.0 * number1Labels/numberSamples)
