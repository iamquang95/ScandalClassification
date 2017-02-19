import codecs

import TextProcessUtils


fiLabeledFilename = "data2.txt"
fiFullTextFilename = "FullData.txt"
foContenFilename = "content.txt"
foFilename = "output.txt"

def appendLabelToEndLine(content, label):
    return "%s %s" % (content, label)

# O(len)
def genContentFile():
    fiLabeled = codecs.open(fiLabeledFilename, "r", "UTF-8")
    fiFullText = codecs.open(fiFullTextFilename, "r", "UTF-8")
    
    foContent = codecs.open(foContenFilename, "w", "UTF-8")

    titles = []
    for line in fiLabeled:
        titles.append(TextProcessUtils.getTitle(line))

    lineType = 0
    index = 0
    labeledSize = len(titles)
    matchedTitle = False

    for line in fiFullText:
        if (lineType == 0):
            if (index >= labeledSize):
                break
            title = TextProcessUtils.removeEndline(line)
            matchedTitle = (title == titles[index])
            index += 1
        elif (lineType == 1):
            if (matchedTitle):
                foContent.write("%s\n" % TextProcessUtils.getSummary(line))
            else:
                foContent.write("x\n")
        lineType = (lineType + 1) % 3

    foContent.close()




def mergeTitleContentLabels():
   fiLabeled = codecs.open(fiLabeledFilename, "r", "UTF-8")
   fiContent = codecs.open(foContenFilename, "r", "UTF-8")

   fo = codecs.open(foFilename, "w", "UTF-8")

   titles = []
   labels = []
   contents = []

   for line in fiLabeled:
       titles.append(TextProcessUtils.getTitle(line))
       labels.append(TextProcessUtils.getLabel(line))

   for line in fiContent:
       contents.append(line)

   for (title, content, label) in zip(titles, contents, labels):
       fo.write("%s. %s %s\n" % (title, TextProcessUtils.removeEndline(content), label))

   fo.close()

if __name__ == "__main__":
  genContentFile()
  mergeTitleContentLabels()
