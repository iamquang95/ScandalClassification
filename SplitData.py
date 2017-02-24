import codecs
import random

if __name__ == "__main__":
	f = codecs.open("data_to_split.txt", "r", "utf-8")

	trainFile = codecs.open("train.tagged", "w+", "utf-8")
	testFile = codecs.open("test.tagged", "w+", "utf-8")

	cntTestData = 0
	maxTestData = 1000
	randomRate = 4
	justAdd = 0

	for line in f:
	    if (random.randint(0, 9) < randomRate) and (cntTestData < maxTestData) and (justAdd > 5):
	        cntTestData += 1
	        testFile.write(line)
	        justAdd = 0
	    else:
	        trainFile.write(line)
	        justAdd += 1

	trainFile.close()
	testFile.close()


