import os
import xml.etree.ElementTree as ET
from itertools import product
import re
import ipdb
import gensim

# settings collected here
dataDirName = '../Data'
cleanedDir = '../CleanData'
modelDir = '../Models'
sentences = []
windows = [2,3,5] 
sizes = [100,200,500]
model = [0,1]
min_count = 5
worker = 3

# Idea: Preprocess the files and write them out to /tmp/ with one sentence per line
def preprocess():
    for filename in os.listdir(dataDirName):
        print(os.path.join(dataDirName, filename))
        with open(os.path.join(dataDirName, filename)) as file:
            with open(os.path.join(cleanedDir, filename),mode='w+') as outf:
                sentences = []
                # clean the data by removing xml-sensitive characters
                data = file.read().replace('&mdash;', '-')
                data = data.replace('&deg;', 'deg')
                data = re.sub('(&)(?!\\S{1,10};)', '&amp;', data)
                tree = ET.ElementTree(ET.fromstring(data))
                # generate a list of word lists as required by gensim
                for d in tree.getroot().findall('doc'):
                    try:
                        for t in d.find('text'):
                            s = ET.tostring(t, encoding='unicode', method='text')
                            sentences += [re.findall('[a-zA-Z0-9]+', w) for s in s.splitlines() for w in re.findall('[^.!?:]+', s) if not bool(re.match('^\\s*$', s))]
                    except TypeError:
                        print('Text missing in ' + filename) # We do not care since there is no content we could embed
                print(len(sentences))
                #ipdb.set_trace()
                # write the data to files with one sentence per line to reduce
                # RAM requirements, see below class MySentence
                for sen in sentences[:-1]:
                    if len(sen)==0:
                        continue
                    outf.write(' '.join(sen))
                    outf.write('\n')
                if len(sentences[-1])>1:
                    outf.write(' '.join(sentences[-1]))
                del sentences
            #ipdb.set_trace()

# following idea is based on https://rare-technologies.com/word2vec-tutorial/ to reduce RAM requirements
class MySentences():
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()

# train by applying all parameter combinations to the data
def train():
    for n in (product(windows, sizes, model)):
        sentences = MySentences(cleanedDir) # a memory-friendly iterator
        model = gensim.models.word2vec.Word2Vec(sentences, size=n[1], window=n[0],
                                                min_count=min_count,
                                                workers=worker, sg=n[2])
        model.save(os.path.join(modelDir, '{0}-{1}-{2}'.format(*n)))
    #    print(os.path.join(modelDir, '{0}-{1}-{2}'.format(*n)))
        print('{0}-{1}-{2}'.format(*n))
    #    print(n[0])
    #print(next(iter(sentences)))

# calculate all metrics
