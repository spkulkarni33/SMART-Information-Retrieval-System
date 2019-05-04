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
from nltk.stem import WordNetLemmatizer
import struct
from struct import calcsize
nltk.download('wordnet')

def read(docu, i):
	ftable = []
	#print(i)
	document = minidom.parse(docu)
        doc_tag = document.getElementsByTagName('DOCNO')
        d = doc_tag[0].firstChild.data
        #print(d)
	text_tag = document.getElementsByTagName('TEXT') # parse text tag under docno
	text = text_tag[0].firstChild.data  #used to read tree structured data
	#print(text)
	text = text.replace("\'s"," ")
	text = re.sub('[^a-zA-Z]+', ' ', text)
	text_tokenize = nltk.word_tokenize(text)
	ftable = ftable + text_tokenize
	#print len(ftable)
	do = int(d)
	#print(do)
	lemma_dict(ftable, do)
	stem_dict(ftable, do)
	
lemma_table = dict()
stem_table = dict()

def lemma_dict(ft, i):
	doclen = len(ft)
	largest_doc_len_in_c.append((doclen, i))
	for j in range(0, len(ft)):
        	ft[j] = ft[j].lower()
	nft = []
	for w in ft:			#remove stopwords
		if w in set(stopwords.words('english')):
			continue
		else:
			nft.append(w)
	word_net_lemmatizer = WordNetLemmatizer()
	#maxtf = most_common(nft)
	term = max(set(nft), key=nft.count)		#obtain frequency of the maximum occuring lemma
	maxtf = nft.count(term)
	maxtf_in_c.append((maxtf, i))
	for w in nft:
		lemma = word_net_lemmatizer.lemmatize(w)
		if lemma not in lemma_table:
			lemma_table[lemma] = [1, 1, [{"docno":i,"tf":1,"max_frequency":maxtf,"doclen":doclen}]]
		elif any(d["docno"] == i for d in lemma_table[lemma][2]):
			lemma_table[lemma][1] += 1		#if the lemma exists in the table with docno as well
			(item for item in lemma_table[lemma][2] if item["docno"] == i).next()["tf"] += 1
		else:
			lemma_table[lemma][0] += 1;
			lemma_table[lemma][1] += 1;
			lemma_table[lemma][2].append({"docno":i, "tf":1, "max_frequency":maxtf, "doclen":doclen})
	#print(lemma_table)

def uncompress_lemma_binary():
	binary = open("Index_Version1.uncompressed", "wb")
	buff1 = ''
	buff2 = ''
	buff3 = ''
	lemma_pointer = 0
	global num_inverted_uncompressed_v1
	num_inverted_uncompressed_v1 = 0
	posting_pointer = 0
	for lemma_key, lemma_value in lemma_table.iteritems():
		term = lemma_key
		termb = term.encode('UTF-8')
		buff2 += termb			#encode lemma term in binary
		postingb = ''
		for item in lemma_value[2]:
			postingb += struct.pack(">4i", item['docno'], item['tf'], item['max_frequency'], item['doclen']) 	
		buff3 += postingb
		df = lemma_value[0]		#The document frequency
		termf = lemma_value[1]		#Term Frequency
		buff1 = struct.pack(">4i", df, termf, lemma_pointer, posting_pointer)
		lemma_pointer += len(termb)
		posting_pointer += len(postingb)
		num_inverted_uncompressed_v1 += 1
	binary.write(buff1)
	binary.write(buff2) 		#term
	binary.write(buff3) 		#postings

def bitstringbytes(v):
        m = int(v, 2)
        b = bytearray()
        while m:
                b.append(m & 0xff)
                m >>= 8
        return bytes(b[::-1])

def gammacods(i, flag):
        bina = bin(i)[2:]                       #binary equivalent
        offset = bina[1:]			#discard
        loffset = len(offset)
        value = '1'*loffset+'0'
        v = value + offset
	if (flag == 2):				#for delta codes 
		return v
	else:
        	return bitstringbytes(v)

def compress_lemma_binary():
	binaryc = open("Index_Version1.compressed", "wb")
	posting_pointer = 0
	global num_inverted_compressed_v1
	buff1 = ''
	buff2 = ''
	buff3 = ''
	posting_b = ''
	lemma_pointer = 0
	blocksize = 0
	num_inverted_compressed_v1 = 0
	prevart = 0
	blockcount = 0
	num_inverted_compressed_v1 = 0
	for lemma_key, lemma_values in sorted(lemma_table.iteritems()):
		blockcount = blockcount + 1
		num_inverted_compressed_v1 += 1
		term = lemma_key
		termb = term.encode('UTF-8')
		ntermb = struct.pack(">i", len(termb)) + termb		#1 byte for storing the term length
		buff2 += ntermb
		postingb = ''
		#j = 0		
		for item in lemma_values[2]:		#building docno, tf, maxtf, doclen			
			prevart = lemma_values[2][0]['docno']			#Extract the first docno from the posting list
			#print(prevart)
			#j+=1;
			if item['docno'] == prevart:
				#print("Previous doc: "+pervart)
				#print(prevart)
				postingb += gammacods(item['docno'], 1) + struct.pack(">3i", item['tf'], item['max_frequency'],item['doclen'])
			else:
				gap = item['docno'] - prevart
				prevart = item['docno']
				#print("Previous doc: ")
				#print(prevart)
				#print("Gap is : ")
				#print(gap)
				#print(gap)
				postingb += gammacods(gap, 1) + struct.pack(">3i", item['tf'], item['max_frequency'],item['doclen'])
		buff3 += postingb
		df = lemma_values[0]
		cf = lemma_values[1]
		if blockcount == 8:
			buff1+=struct.pack(">4i",df, cf, posting_pointer,lemma_pointer)
			blockcount = 0
		else:
			buff1 += struct.pack(">3i", df, cf, posting_pointer)
		lemma_pointer += len(ntermb)
		posting_pointer += len(postingb)
		
	binaryc.write(buff1)
	binaryc.write(buff2)
	binaryc.write(buff3)

def stem_dict(ft, i):
	doclen = len(ft)
	for j in range(0, len(ft)):
        	ft[j] = ft[j].lower()
	nft = []
	for w in ft:			#remove stopwords
		if w in set(stopwords.words('english')):
			continue
		else:
			nft.append(w)	
	
	term= max(set(nft), key=nft.count)		#obtain frequency of the maximum occuring stem
	maxtf = nft.count(term)
	ps = PorterStemmer()
	for item in nft:
		it = ps.stem(item)
		if it not in stem_table:
			stem_table[it] = [1, 1, [{"docno":i,"tf":1,"max_frequency":maxtf,"doclen":doclen}]]
		elif any(d["docno"]== i for d in stem_table[it][2]):
			stem_table[it][1]+=1
			(items for items in stem_table[it][2] if items["docno"]== i).next()["tf"]+=1
		else:		#doc_no not in posting list 
			stem_table[it][0]+=1
			stem_table[it][1]+=1
			stem_table[it][2].append({"docno":i,"tf":1,"max_frequency":maxtf,"doclen":doclen})	
		
def uncompressed_stem_binary():
	binary = open("Index_Version2.uncompressed", "wb")
	global num_inverted_uncompressed_v2
	buff1 = ''
	buff2 = ''		#for the stem term
	buff3 = ''		#for the posting list
	#sample = ''
	posting_pointer = 0
	stem_pointer = 0
	num_inverted_uncompressed_v2 = 0
	for stem_key, stem_value in stem_table.iteritems():
		term = stem_key
		termb = term.encode("UTF-8")
		num_inverted_uncompressed_v2 += 1 
		buff2 = buff2 + termb
		postingb = ''
		for item in stem_value[2]:

			postingb += struct.pack(">4i", item['docno'], item['tf'], item['max_frequency'], item['doclen'])
			#print(postingb)
		buff3 = buff3 + postingb
		df = stem_value[0]
		cf = stem_value[1]
		buff1 = struct.pack(">4i", df, cf, posting_pointer, stem_pointer)
		stem_pointer += len(termb)
		posting_pointer += len(postingb)
	map_tablesize = len(buff1)
	stemsize = len(buff2)
	binary.write(buff1)
	#term
	binary.write(buff2)
	#posting list
	binary.write(buff3)

def deltacods(i):
        bina = bin(i)[2:]                       #binary equivalent
        offset = bina[1:]
        lbin = len(bina)
        value = gammacods(lbin, 2)
        v = value + offset
        return bitstringbytes(v)	

def frontcode(terms):
	pref = os.path.commonprefix(terms)
	newtermb = struct.pack(">b", len(terms[0])) + pref.encode("UTF-8") + '*'.encode("UTF-8") + terms[0][(len(pref) - len(terms[0])):]
	for item in terms[1:]:
		extralength = len(pref) - len(pref)
		newtermb+=struct.pack(">b",extralength)+'$'.encode('UTF-8')+item[-extralength:].encode('UTF-8')
	return newtermb


def compress_stem_binary():
	binary=open("Index_Version2.compressed","wb")
	global num_inverted_compressed_v2
	buff1=''
	buff2=''
	stem_pointer=0
	buff3=''				#posting list
	posting_pointer=0
	blockcount=8
	termlist=[]
	num_inverted_compressed_v2 = 0
	for stem_key, stem_values in sorted(stem_table.iteritems()):
		term=stem_key
		num_inverted_compressed_v2 +=1		
		postingb=''
		for item in stem_values[2]:
			first_article_no = stem_values[2][0]['docno']		#obtain first article
			if item['docno'] == first_article_no:
				postingb += deltacods(item['docno']) + struct.pack(">3i", item['tf'], item['max_frequency'],item['doclen'])
			else:
				gap=item['docno']-first_article_no
				first_article_no = item['docno']
				postingb += deltacods(gap) + struct.pack(">3i", item['tf'], item['max_frequency'],item['doclen'])
				
		buff3+=postingb
		df=stem_values[0]
		cf=stem_values[1]
		if blockcount==8:						#blocksize = 8
			buff1+=struct.pack(">4i",df, cf, posting_pointer, stem_pointer)

			blockcount = 1;
		else:
			buff1 += struct.pack(">3i",df, cf, posting_pointer)
			blockcount += 1
		
		if blockcount == 8:
			front_coding_termb = frontcode(termlist)				#front coding 
			buff2 += front_coding_termb
			stem_pointer += len(front_coding_termb)
			termlist = []
			termlist.append(term)
		else:
			termlist.append(term)
		
		posting_pointer += len(postingb)
	map_tablesize = len(buff1)
	stemsize = len(buff2)
	binary.write(buff1)
	#term
	binary.write(buff2)
	#posting list
	binary.write(buff3)


def query(term):
	#ps1 = PorterStemmer()
	text = ps1.stem(term)
	dfreq=stem_table[text][0]
	tfreq=stem_table[text][1]
	postingb = ''
	for item in stem_table[text][2]:
		postingb += struct.pack(">4i", item['docno'], item['tf'], item['max_frequency'], item['doclen'])
	p = len(postingb)
	print term, 'df:',dfreq,'tf:',tfreq,'Length of the posting list in bytes: ',p

##########################################################################################################################
documents = os.listdir(sys.argv[1])
s = len(documents)  #Total number of documents in the dataset
print("Total number of documents detected: ")
print(s)

ps1 = PorterStemmer()
#go to directory where dataset is present 
#get_ipython().magic('cd /home/sanketkulkarni/Desktop/SanketIR/Cranfield')
get_ipython().magic('cd '+sys.argv[1])
#largest max_tf_docid
maxtf_in_c=[]
#Largest doc_len
largest_doc_len_in_c=[]			
t11 = time.time()
	
for i in range(0,s):#documents:
	read(documents[i], i)
t21 = time.time()
print("Time taken to read the files and build the dictionary:")
t31 = t21-t11
print(t31)

get_ipython().magic('cd')
t1 = time.time()
#version 1: uncompressed lemmatized index
uncompress_lemma_binary()
t2 = time.time()
t3 = t2-t1
print("Time taken to create Index_Version1.uncompressed:")
print(t3)

t4 = time.time()
#version 2: compressed lemmatized index
compress_lemma_binary()
t5 = time.time()
t6 = t5-t4
print("Time taken to create Index_Version1.compressed:")
print(t6)

t7 = time.time()
#version 3: uncompressed stem index
uncompressed_stem_binary()
t8 = time.time()
print("Time taken to create Index_Version2.uncompressed:")
t9 = t8 - t7
print(t9)


t30 = time.time()
#version 4: compressed stem index
compress_stem_binary()
t31 = time.time()
print("Time taken to create Index_Version2.compressed:")
t32 = t31 - t30
print(t32)


#Sizes of the 4 versions of the indices
cwd = os.getcwd()
st = os.stat(cwd+"/Index_Version1.uncompressed")
sv1u =  st.st_size
print("Size of Index_Version1.uncompressed is:")
print sv1u,'bytes'

temp2 = os.stat(cwd+"/Index_Version1.compressed")
sv1c =  temp2.st_size
print("Size of Index_Version1.compressed is:")
print sv1c,'bytes'

temp3 = os.stat(cwd+"/Index_Version2.uncompressed")
sv2u =  temp3.st_size
print("Size of Index_Version2.uncompressed is:")
print sv2u,'bytes'

temp4 = os.stat(cwd+"/Index_Version2.compressed")
sv2c =  temp4.st_size
print("Size of Index_Version2.compressed is:")
print sv2c,'bytes'

"""#document frequency and term frequency for "Reynolds"
text = ps1.stem("Reynolds")
dfreq=stem_table[text][0]
tfreq=stem_table[text][1]
print text, 'df:',dfreq,'tf:',tfreq

#document frequency and term frequency for "NASA"
text = ps1.stem("NASA")
dfreq=stem_table[text][0]
tfreq=stem_table[text][1]
print text, 'df:',dfreq,'tf:',tfreq

#document frequency and term frequency for "Prandtl"
text = ps1.stem("Prandtl")
dfreq=stem_table[text][0]
tfreq=stem_table[text][1]
print text, 'df:',dfreq,'tf:',tfreq

#document frequency and term frequency for "flow"
text = ps1.stem("flow")
dfreq=stem_table[text][0]
tfreq=stem_table[text][1]
print text, 'df:',dfreq,'tf:',tfreq"""

#Number of inverted lists in each version of the index:

print ("The number of Inverted Lists in Index_Version1.uncompressed: ") 
print(num_inverted_uncompressed_v1) 
print(" The number of inverted lists in Index_Version1.compressed: ")
print(num_inverted_compressed_v1)
print ("The number of Inverted Lists in Index_Version2.uncompressed:")
print(num_inverted_uncompressed_v2)
print("The number of inverted lists in Index_Version2.compressed: ")
print(num_inverted_compressed_v2)

#df and tf for the terms
query("Reynolds")
query("NASA")
query("Prandtl")
query("flow")
query("pressure")
query("boundary")
query("shock")


#First 3 posting lists for the term NASA

print '\"NASA\"', 'df: ',stem_table[ps1.stem('nasa')][0]
print 'First item in Posting list:', 'term_frequency: ', stem_table[ps1.stem('nasa')][2][0]['tf'], 'Document length: ',stem_table[ps1.stem('nasa')][2][0]['doclen'], 'Max_tf: ',stem_table[ps1.stem('nasa')][2][0]['max_frequency']
print 'Second item in Posting list:', 'term_frequency: ', stem_table[ps1.stem('nasa')][2][1]['tf'], 'Document length: ',stem_table[ps1.stem('nasa')][2][1]['doclen'], 'Max_tf: ',stem_table[ps1.stem('nasa')][2][1]['max_frequency']
print 'Third item in Posting list:', 'term_frequency: ', stem_table[ps1.stem('nasa')][2][2]['tf'], 'Document length: ',stem_table[ps1.stem('nasa')][2][2]['doclen'], 'Max_tf: ',stem_table[ps1.stem('nasa')][2][2]['max_frequency']



res = max(largest_doc_len_in_c,key=lambda x:x[0])
large = res[1]
print("Document having the largest document length in the collection is Document number: ")
print(large)

res1 = max(maxtf_in_c,key=lambda x:x[0])
lc = res[1]
print("The document having highest tf is :")
print(lc)


