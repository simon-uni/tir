import os
import xml.etree.ElementTree as ET
import re
import ipdb

dataDirName = '../Data'
sentences = []

# Idea: Preprocess the files and write them out to /tmp/ with one sentence per line
for filename in os.listdir(dataDirName):
    print(os.path.join(dataDirName, filename))
    with open(os.path.join(dataDirName, filename)) as file:
        with open(os.path.join('/tmp', filename),mode='w+') as outf:
            sentences = []
            data = file.read().replace('&mdash;', '-')
            data = data.replace('&deg;', 'deg')
            data = re.sub('(&)(?!\\S{1,10};)', '&amp;', data)
            tree = ET.ElementTree(ET.fromstring(data))
            for d in tree.getroot().findall('doc'):
                try:
                    for t in d.find('text'):
                        s = ET.tostring(t, encoding='unicode', method='text')
                        sentences += [re.findall('[a-zA-Z0-9]+', w) for s in s.splitlines() for w in re.findall('[^.!?:]+', s) if not bool(re.match('^\\s*$', s))]
                except TypeError:
                    print('Text missing in ' + filename) # We do not care since there is no content we could embed
            print(len(sentences))
            #ipdb.set_trace()
            for sen in sentences:
                if len(sen)==0:
                    continue
                for word in sen:
                    outf.write(word.lower() + ' ')
                outf.write('\n')
            del sentences
        #ipdb.set_trace()
