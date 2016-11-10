import nltk
import nltk.chunk
import codecs
from collections import Counter

def load_sentences(path):
    sentences = []
    sentence = []
    for line in codecs.open(path, 'r', 'utf8'):
        line = line.rstrip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split()
            assert len(word) >= 2
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences

#def conll_tag_chunks(chunk_sents):
#    tag_sents = [nltk.chunk.tree2conlltags(tree) for tree in chunk_sents]
#    return [[(t, c) for (w, t, c, k) in chunk_tags] for chunk_tags in tag_sents]

train='/home/ayushi/github/NLP/dataset/CoNLL-2003/eng _lesser.testa'

allData=load_sentences(train)
#alltaggedWords=list(conll2000.tagged_words(train))
#allchunckedWords=list(conll2000.chunked_words(train))
#
##freq of each word
#freq_word=nltk.FreqDist(word for(word, tag) in alltaggedWords)
##freq of each tag
#freq=nltk.FreqDist(tag for (word,tag) in alltaggedWords)

#print("------------")

#all unique tags in given dataset
#alltags=[a for a in freq]
#all unique words in given dataset
#allWords=[a for a in freq_word]

globalData=[]
for i in allData:
    for j in i:
        globalData.append(j)

globalData=[tuple(l) for l in globalData]

freq_word=nltk.FreqDist(word for (word,tag,chunck,ne) in globalData)
freq_tag=nltk.FreqDist(tag for (word,tag,chunck,ne) in globalData)
freq_chunck=nltk.FreqDist(chunck for (word,tag,chunck,ne) in globalData)
freq_ne=nltk.FreqDist(ne for (word,tag,chunck,ne) in globalData)        

allbigrams=list(nltk.bigrams(globalData))

#all unique words,tags,chuncks,ne
allWords=[a for a in freq_word]
allTags=[a for a in freq_tag]
allChunck=[a for a in freq_chunck]
allNe=[a for a in freq_ne]

dict1={}
#P(NEi|NEi-1)
for i in allNe:
    ne_type=[b[3] for (a,b) in allbigrams if a[3]==i]
    dict1[i]=ne_type
for i in dict1 :
    dict1[i]=dict(Counter(dict1[i]))
    totalCount=sum(dict1[i].values());
    for k in dict1[i]:
        dict1[i][k]/=float(totalCount)
    print (i, ':', dict1[i])
print('-----------------')

#P(W|NE)
dict2={}
for i,x,y,j in globalData: # i:word j:tag
    dict2[j]=[a for (a,x,y,b) in globalData if b==j]
for i in dict2 :
    dict2[i]=dict(Counter(dict2[i]))
    totalCount=sum(dict2[i].values());
    for k in dict2[i]:
        dict2[i][k]/=float(totalCount)
    print(i,':',dict2[i])

#for i in allData:
#    #print(i)
#    temp = [tuple(l) for l in i]
#    count=nltk.FreqDist(ne for(word,tag,chunk,ne) in temp)
#    print(count)
#    print("----------")

#
#for sents in conll2000.chunked_sents(train):
#    print (sents)    
#    print("------")

#print(conll2000.chunked_sents(train))
#myarr=[]
#myarr=conll_tag_chunks(conll2000.chunked_sents(train))
#chunk_sents=conll2000.chunked_sents(train)
#tag_sents = [nltk.chunk.tree2conlltags(tree) for tree in chunk_sents]
#alltaggedWords=list(conll2000.tagged_words(train))
#allchunckedWords=list(conll2000.chunked_words(train))
#allNEwords=list(conll2000.iob_words(train))


 

#allbigrams=list(nltk.bigrams(alltaggedWords))
#freq_word=nltk.FreqDist(word for(word, tag) in alltaggedWords)
#freq=nltk.FreqDist(tag for (word,tag) in alltaggedWords)

#print("-------------")
#alltags=[a for a in freq]
#allWords=[a for a in freq_word]
#print("------------")



