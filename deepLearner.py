import os
import xml.etree.ElementTree as ET
from itertools import product
import re
import ipdb
import gensim
from math import log
from sys import argv

# settings collected here
dataDirName = '../Data'
cleanedDir = '../CleanData'
modelDir = '../Models'
sentences = []
windows = [5] 
sizes = [100,300]
models = [0,1]
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
    for n in (product(windows, sizes, models)):
        sentences = MySentences(cleanedDir) # a memory-friendly iterator
        model = gensim.models.word2vec.Word2Vec(sentences, size=n[1], window=n[0],
                                                min_count=min_count,
                                                workers=worker, sg=n[2])
        model.save(os.path.join(modelDir, '{0}-{1}-{2}'.format(*n)))
    #    print(os.path.join(modelDir, '{0}-{1}-{2}'.format(*n)))
        print('{0}-{1}-{2}'.format(*n))
    #    print(n[0])
    #print(next(iter(sentences)))

def trainAssignment():
    sentences = MySentences('/home/swingert/workspace/newModels') # a memory-friendly iterator
    model = gensim.models.word2vec.Word2Vec(sentences)
    similar, _ = zip(*model.most_similar(positive=argv[1:], topn = 5))
    print('%s' % ' '.join(similar))

#trainAssignment()

# calculate all metrics
def metrics(word):
    print(word)
    for m in models:
        basemodel = gensim.models.Word2Vec.load(os.path.join(modelDir,
                                                             '5-500-' + str(m)))
        baselist, _ = zip(*basemodel.most_similar(positive=word)) 
        for n in (product(windows, sizes)):
            print('{0} window, {1} layers, '.format(*n) + ('CBOW' if m == 0
                                                           else 'skip-gram'))
            othermodel = gensim.models.Word2Vec.load(os.path.join(modelDir,
                                                             '{0}-{1}-'.format(*n)
                                                             + str(m)))
            otherlist, _ = zip(*othermodel.most_similar(positive=word))
            # precision@5
            prec = set(baselist[0:5]).intersection(set(otherlist[0:5]))
            print('precision@5=\t'+str(len(prec)/5.0))
            # recall
            recall = set(baselist[0:10]).intersection(set(otherlist[0:10]))
            print('recall=\t\t'+str(len(recall)/10.0))
            # ndcg@10
            sum=0
            for i in range(0,9):
                if(otherlist[i] in baselist[0:5]):
                    sum+=(2^2 - 1)/log(i+1+1)
                if(otherlist[i] in baselist[5:10]):
                    sum+=(2^1 - 1)/log(i+1+1)
            print('NDCG@10=\t' + str(sum))

            # MAP
            prec=j=0
            for i in range(0,9):
                if(otherlist[i] in baselist[0:(i+1)]):
                    j+=1
                    prec += len(set(baselist[0:(i+1)]).intersection(set(otherlist[0:(i+1)])))/float(i+1)
            print('MAP=\t\t' + str(prec/float(j) if j!= 0 else 0))

metrics('earthquake')
metrics('diabetes')
