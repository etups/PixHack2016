import os
import langid
import glob

filter_lang = ['ja', 'ko']

def main():

    filter_dict = dict()
    for filename in glob.glob('./*.txt'):
        f = open(filename, 'r')
        for line in f.readlines():
            for word in line.split():
                lang, conf = langid.classify(word)
                if lang in filter_lang:
                    if word in filter_dict:
                        filter_dict[word] += 1
                    else:
                        filter_dict[word] = 1

        f.close()

    f = open("filter.txt", "w+")
    for k, v in filter_dict.items():
        f.write("%s %d\n" % (k, v))

if __name__ == '__main__':
    main()
