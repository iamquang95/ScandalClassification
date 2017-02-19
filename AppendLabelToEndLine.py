import codecs

if __name__ == "__main__":

    f = codecs.open("test.txt", 'r', 'utf-8')

    g = codecs.open("data.txt", 'w+', 'utf-8')

    for line in f:
        s = "%s %s\n" % (line[2:len(line) - 1], line[:2])
        print(s)
        g.write(s)

    g.close()
