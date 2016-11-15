import nltk
import nltk.chunk
import codecs
import testing
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

train='/home/ayushi/github/NLP/dataset/CoNLL-2003/eng _lesser.train'

allData=load_sentences(train)
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

#all unique words,tag   s,chuncks,ne
allWords=[a for a in freq_word]
allTags=[a for a in freq_tag]
allChunck=[a for a in freq_chunck]
allNe=[a for a in freq_ne]
#P(NE|Sentence/context)=P(context|NE)*P(NE)
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
print('-----------------')
#P(NE|POS,W)
#P(POS|W)=P(W|POS)
#P(fet|NE)

#Probability of word being starting sentence
sentencestart={}
for i in allData:
    if i[0][0] in sentencestart.keys():
        sentencestart[i[0][0]]+=1
    else:
        sentencestart[i[0][0]]=1
totalCount=len(allWords)
for i in sentencestart:
    sentencestart[i]/=float(totalCount)
print('-----------------')
print("-----testing-------")
test='/home/ayushi/github/NLP/dataset/CoNLL-2003/eng _lesser.testb '
allTestData=load_sentences(test)

count=0
for i in allTestData:
    sent=[]    
    for j in i:
        j[0]=j[0].encode('utf8')
        sent+=[j[0]]
    print(sent)
    param={}
    param['states'] = tuple(dict1.keys()) #named-entities
    param['observations'] = tuple(sent) #word
    param['start_probability'] = sentencestart #tag
    param['transition_probability'] = dict1
    param['emission_probability'] = dict2
    obj= testing.Viterbi(param)   
    obj.viterbi()    
    count+=1
obj=testing.Viterbi(param)
#obj.efficiency()        



