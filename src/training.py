import nltk
import nltk.chunk
import codecs
import viterbi
from collections import Counter
from sklearn.metrics import confusion_matrix

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

train='../dataset/CoNLL-2003/eng.train'

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
    #print (i, ':', dict1[i])
   
print('--------aa---------')
#P(W|NE)
dict2={}
for i in allNe:
    dict2[i]=[a for (a,x,y,j) in globalData if j==i]
for i in dict2:
    dict2[i]=dict(Counter(dict2[i]))
    totalCount=sum(dict2[i].values());
    for k in dict2[i]:
        dict2[i][k]/=float(totalCount)
        
print('-------bb----------')
#P(POSi|POSi-1)
dict3={}
for i in allTags:
    dict3[i]=[b[1] for (a,b) in allbigrams if a[1]==i]   
for i in dict3 :
    dict3[i]=dict(Counter(dict3[i]))
    totalCount=sum(dict3[i].values());
    for k in dict3[i]:
        dict3[i][k]/=float(totalCount)
print('--------cc---------')
#P(POS|NE)
dict5={}
for i in allNe:
    dict5[i]=[x for (a,x,y,j) in globalData if j==i]
for i in dict5:
    dict5[i]=dict(Counter(dict5[i]))
    totalCount=sum(dict5[i].values());
    for k in dict5[i]:
        dict5[i][k]/=float(totalCount)
print('--------aa---------')
#P(Chunk|NE)
dict6={}
for i in allNe:
    dict6[i]=[y for (a,x,y,j) in globalData if j==i]
for i in dict6:
    dict6[i]=dict(Counter(dict6[i]))
    totalCount=sum(dict6[i].values());
    for k in dict6[i]:
        dict6[i][k]/=float(totalCount)

print('-----------------')
#P(POS,W|NE)
dict4={}
for i in allNe:
    dict4[i]=[(a,x) for (a,x,y,j) in globalData if j==i]
for i in dict4:
    dict4[i]=dict(Counter(dict4[i]))
    totalCount=sum(dict4[i].values());
    for k in dict4[i]:
        dict4[i][k]/=float(totalCount)

#P(POS|W)=P(W|POS)
#P(fet|NE)

#Probability of Named entity being starting starting NE
sentencestart=dict(zip(allNe,[1 for x in range(0,len(allNe))]))
for i in allData:
    if i[0][3] in sentencestart.keys():
        sentencestart[i[0][3]]+=1
    else:
        sentencestart[i[0][3]]=1
totalCount=sum(sentencestart.values())
for i in sentencestart:
    sentencestart[i]/=float(totalCount)
print('-----------------')
print("-----testing-------")
test='../dataset/CoNLL-2003/eng.testb'
allTestData=load_sentences(test)

count=0
predicted=[]
tags=[]
for i in allTestData:
    sent=[]  
    for j in i:
        j[0]=j[0].encode('utf8')
        sent+=[j[0]]
    #print(sent)
    tags=nltk.pos_tag(sent)
    pos=[b for (a,b) in tags]
    #chunk=nltk.chunk_sents(sent)
    param={}


#    param['states'] = tuple(allNe) #named-entities
#    param['observations'] = tuple(sent) #word
#    param['start_probability'] = sentencestart #tag
#    param['transition_probability'] = dict1
#    param['emission_probability'] = dict2

#    param['states'] = tuple(allNe) #named-entities
#    param['observations'] = tuple(tags) #word
#    param['start_probability'] = sentencestart #tag
#    param['transition_probability'] = dict1
#    param['emission_probability'] = dict4


    param['states'] = tuple(allNe) #named-entities
    param['observations'] = tuple(pos) #word
    param['start_probability'] = sentencestart #tag
    param['transition_probability'] = dict1
    param['emission_probability'] = dict5
    
    obj= viterbi.Viterbi(param)   
    predicted=predicted+[obj.viterbi()[1]]    
    count+=1
    
#obj.efficiency()        
actual=[]
for i in allTestData:
    line=[]
    for j in i:
        line=line+[j[3]]
    actual=actual+[line]    
x=0;
correct=0;total=0
for i in actual:
    y=0;
    for j in i:
        if(j==predicted[x][y]):
            correct+=1;
        total+=1
        y+=1
    x+=1
accuracy=((correct+0.0)/total)*100;
predicted_label=[item for sublist in predicted for item in sublist]
actual_label=[item for sublist in actual for item in sublist]
cn=confusion_matrix(actual_label,predicted_label,labels=allNe);
print(cn)
print (accuracy)