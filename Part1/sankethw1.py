#Libraries
import numpy as np
import nltk  #Natural Language Processing toolkit
from xml.dom import minidom  #Needed to parse tag structured files
nltk.download('punkt') #tokenizer
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer #Open Source Porterstemmer implementation
from nltk.corpus import stopwords
import re
import os 
import collections
import sys
import time


#Function for tokenization
def document_tokenize(docu, flag): #flag = 1 returns count, 2 returns token list, 3 returns the entire posting list
    document = minidom.parse(docu)  
    text_tag = document.getElementsByTagName('TEXT') # parse text tag under docno
    text = text_tag[0].firstChild.data  #used to read tree structured data
    text = text.replace("\'s"," ")
    text = re.sub('[^a-zA-Z]+', ' ', text)
    text_tokenize = nltk.word_tokenize(text)  #tokenizer
    wfreq = [] 
    for w in text_tokenize:                     #count occurences
        wfreq.append(text_tokenize.count(w))  
    posting_list=list(zip(text_tokenize,wfreq))  
    count = 0
    for i in range(0,len(wfreq)):
        count = count+wfreq[i]
    if(flag==1):
        return count #return count of words in a document
    if(flag==2):
        return text_tokenize  #return list of tokens in a document
    if(flag == 3):
	return posting_list #2 dimensional representation of posting list


#Listing dataset files : /home/sanketkulkarni/Desktop/Cranfield
import os
documents = os.listdir(sys.argv[1])
s = len(documents)  #Total number of documents in the dataset
print("Total number of documents detected: ")
print(s)


#go to directory where dataset is present 
get_ipython().magic('cd '+sys.argv[1])
t1 = time.time()
#tokenize the Cranfield dataset
tokenized_words = []
for i in range(0,s):
    tokenized_words = tokenized_words + document_tokenize(documents[i], 2)  

#FUNCTIONS: these will be called by tokens as well as stemmer

#COMPUTE UNIQUE WORDS IN THE COLLECTION
def unique_words(tokenized_w):
    final_list = collections.Counter(tokenized_w)
    l = len(final_list)
    print("Number of unique words in the collection: ")
    print(l)
    return final_list


# WORDS OCCURING ONLY ONCE IN THE ENTIRE COLLECTION
def only_once(final_list):
    c = 0
    for w,freq in final_list.items():
    	if(freq == 1):
    	    c=c+1
    print("Words that are occuring only once in the collection: ")
    print(c)


# TOP 30 WORDS IN THE COLLECTION
def top30(tokenzied_words):
    #t30 = []
    t30 = collections.Counter(tokenized_words).most_common(30)
    print("Top 30 words and their frequencies are")
    print(t30)


#TOTAL TOKENS IN THE COLLECTION
print("Total tokens in Cranfield collection = ")
total = len(tokenized_words)
print(total)

final_list = unique_words(tokenized_words)
#print(final_list)

only_once(final_list)

top30(tokenized_words)

# In [102]:AVERAGE NO OF TOKENS PER DOCUMENT
avg = total/s
print("Average tokens per document:")
print(avg)

t2 = time.time()
t3 = t2-t1
print("Seconds required for tokenizing Cranfield collection:")
print(t3)

#STEMMER
def stemming(docu):
    t = document_tokenize(docu,2)
    ps = PorterStemmer()
    for i in range(0, len(t)):
        t[i] = t[i].lower()  #same case for alphabets
    t = [ps.stem(word) for word in t if not word in set(stopwords.words('english'))]
    return t


#STEMMING 
t4 = time.time()
print("Now gathering information for stemming...")
print("\n")
stem = []
for i in range(0,s):
	stem = stem + stemming(documents[i])
#print(stem)
print("Number of Stems after removing stopwords= ")
print(len(stem))
final_stem = []
final_stem = unique_words(stem)
#print(final_stem)
only_once(final_stem)

topthrity = collections.Counter(stem).most_common(30)
print("The most frequently 30 occuring stems are:")
print(topthrity)
t5 = time.time()
t6 = t5-t4
print("Seconds required for stemming operation for Cranfield collection:")
print(t6)




