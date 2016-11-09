#!/usr/bin/python
import nltk
import testing
import sys
from nltk.corpus import brown
from nltk import word_tokenize
from collections import Counter
input_file="/home/ayushi/nltk_data/corpora/brown"

#train=['ca01','ca02']#,'ca03','ca04','ca05','ca06']#,'cb01','cb02','cb03','cb04','cb05','cb06']
train=['ca01','ca02','cb01','cb02','ca03','ca04','cb03','cb04']
test=['ay']
alltaggedWords=list(brown.tagged_words(train))
allbigrams=list(nltk.bigrams(alltaggedWords))
freq_word=nltk.FreqDist(word for(word, tag) in alltaggedWords)
freq=nltk.FreqDist(tag for (word,tag) in alltaggedWords)
print("\n------------")
alltags=[a for a in freq]
#for i in freq:
#    alltags.append(i)

allWords=[a for a in freq_word]
#for i in freq_word :
#    allWords.append(i)
print("------------")
#print (alltags)
#print(allWords)
#print(allbigrams)

dict1={}
#P(ti|ti-1)
for i in alltags :
    tag_type=[b[1] for (a,b) in allbigrams if a[1]==i]
    dict1[i]=tag_type
for i in dict1 :
    dict1[i]=dict(Counter(dict1[i]))
    totalCount=sum(dict1[i].values());
    for k in dict1[i]:
        dict1[i][k]/=float(totalCount)
#    print (i, ':', dict1[i])
print("------------")
    #P(w|t)
dict2={}
for i,j in alltaggedWords: # i:word j:tag
    dict2[j]=[a for (a, b) in alltaggedWords if b==j]
for i in dict2 :
    dict2[i]=dict(Counter(dict2[i]))
    totalCount=sum(dict2[i].values());
    for k in dict2[i]:
        dict2[i][k]/=float(totalCount)
print("------------")

#import collections
import os
#sentencestart = collections.defaultdict()
sentencestart={}
for filename in os.listdir(input_file):
    fname=input_file+'/'+filename
    file = open(fname, "r")
    with file as f:
        for line in f:
            temp=[nltk.tag.str2tuple(t) for t in line.split()]
            for i,j in temp:
                #print (j)
                if j in sentencestart.keys():
                    sentencestart[j]+=1
                else:
                    sentencestart[j]=1
                break
    totalCount=sum(sentencestart.values());
    #print (totalCount)    
    for k in sentencestart:
        sentencestart[k]/=float(totalCount)
print("----end-------")



print("-----testing-------")
count=0
for i in brown.sents(test):
    sent=[]    
    for j in i:
        j=j.encode('utf8')
        sent+=[j]
    param={}
    param['states'] = tuple(dict1.keys()) #tag
    param['observations'] = tuple(sent) #word
    param['start_probability'] = sentencestart #tag
    
    param['transition_probability'] = dict1
    param['emission_probability'] = dict2
    obj= testing.Viterbi(param)   
    obj.viterbi()    
    count+=1
    #if(count>10):
    #    break 
obj=testing.Viterbi(param)
obj.efficiency()    

##--------------------------
#input = raw_input("Enter your sentence: ")
#input_tokens=word_tokenize(input)
#
#param={}
#
#param['states'] = tuple(dict1.keys()) #tag
##param['observations'] = tuple(input_tokens) #word
#param['start_probability'] = sentencestart #tag
#param['observations'] = tuple(input_tokens)
#param['transition_probability'] = dict1
#param['emission_probability'] = dict2
#param['test_data']=test
#obj= testing.Viterbi(param)   
#obj.viterbi()