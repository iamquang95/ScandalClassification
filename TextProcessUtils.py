import re
import string
import codecs

from sets import Set

def getSummary(content):
    res = re.match(r'(.*?\.)[A-Z]', content, re.UNICODE)
    if (res is None or len(res.group(1)) > 1000):
        return ""
    else: 
        return res.group(1)

def getLabel(content):
    leng = len(content)
    return content[leng-2:leng-1]

def getTitle(content):
    leng = len(content)
    return content[:leng-3]

def removeEndline(content):
    return content.strip()

    
def lineToWords(line):
    regex = re.compile(
        r'[%s\s]+' % re.escape(
            string.punctuation.replace("_", "") # Dont use _ to split
            )
        )
    return regex.split(line)

def getDictionary(fileName):
    return Set([])
    f = codecs.open(fileName, "r", "utf-8")
    dict = [removeEndline(line) for line in f]
    f.close()
    return Set(dict)

def countWordInDict(dict, line):
    count = 0
    for word in lineToWords(line):
        if (word in dict):
            count += 1
    return count
