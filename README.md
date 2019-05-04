# SMART-Information-Retrieval-System

## Part 1: Tokenisation, Lemmatisation and Stemming
## Part 2: Single Pass In Memory Indexing. Two versions:
  1) Indexing the lemmas. Compression techniques using blocked compression and gamma encoding.
  2) Indexing the stems. Compression techniques using front coding and delta encoding for document gaps.
## Part 3: Document Retrieval using Custom weighting techniques:
  W1 = (0.4 + 0.6 * log (tf + 0.5) / log (maxtf + 1.0)) * (log (collectionsize / df)/ log (collectionsize))
  W2 = (0.4 + 0.6 * (tf / (tf + 0.5 + 1.5 *(doclen / avgdoclen))) * log (collectionsize / df)/log (collectionsize))
  
where:
tf: the frequency of the term in the document,
maxtf: the frequency of the most frequent indexed term in the document,
df: the number of documents containing the term,
doclen: the length of the document, in words,
discounting stop-words, - you may use the same stopword list as in the previous homework;
avgdoclen: the average document length in the collection, considering the doclen of each document, and
collectionsize: the number of documents in the collection.
