import re
import string

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
    leng = len(content)
    return content[:leng-1]

    
def lineToWords(line):
    regex = re.compile(
        r'[%s\s]+' % re.escape(
            string.punctuation.replace("_", "") # Dont use _ to split
            )
        )
    return regex.split(line)
