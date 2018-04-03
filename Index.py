# this program is used to process texts and index the terms
from __future__ import division
import math 
import os
import pickle  # serialization the same object will not serialized again
from sets import Set
import read_documents
from nltk.stem import SnowballStemmer
from glob import glob
import operator
import time

docfile = "documents.txt"
queryfile = "queries.txt"
stopfile = "stop_list.txt"
databasefile = "database.dat"
queryDatabasefile="queryDatabase.dat"
resultfile = "Result.txt"
MAX_RESULTS_PER_QUERY=10


docCollection = read_documents.ReadDocuments(docfile)
queryCollection = read_documents.ReadDocuments(queryfile)

totalDocCount=0;
wordVsIDF={}
"""
    Extract only alpha numeric characters from the input string
"""
def ExtractAlphanumeric(InputString):
    from string import ascii_letters, digits
    return "".join([ch for ch in InputString if ch in (ascii_letters + digits)])
    

"""
    This part processes the documents and then index the terms in the below form
    {word -> {doc 1: word_freq_count, doc 2: word_freq_count, doc 3: word_freq_count}
"""
def indexBuilder(databasefile,docfile,stopfile):
    docCollection = read_documents.ReadDocuments(docfile)
    stopset = set(line.strip() for line in open(stopfile))
    # dictionary to keep the words
    database = {}
    # unwanted delimiter characters adjoining words.
    delimiter_chars=",.;:!?"
    #Use Stemmer to convert words like from running to run
    snowballStemmer=SnowballStemmer("english")
    global totalDocCount
    totalDocCount=0
    for doc in docCollection: # This check every document
        cnt = 0   # word count
        totalDocCount += 1
        for line in doc.lines:
            # Split the line into words delimited by whitespace.
            words = line.split()
            for word in words:
                # Remove unwanted delimiter characters adjoining words.
                ##print("word before:",word)
                word = word.strip(delimiter_chars)
                ##print("word after:",word)
                #Stemming the word
                word = snowballStemmer.stem(word)
                ##print"word after stem:",word
                cnt += 1
                cur = ExtractAlphanumeric(word.lower())
                if (cur and (cur not in stopset)):
                    if not database.has_key(cur):
                        #print("cur:",cur)
                        database[cur] = {} # This create a new entry in database
                    entry = database[cur]
                    if entry.has_key(doc.docid):
                        #print("doc.docid:",doc.docid)
                        database[cur][doc.docid].append(cnt)
                        #print("doc.docid##:",database[cur][doc.docid])
                    else:
                        database[cur][doc.docid] = [cnt]
                        #print("database[cur][doc.docid] ",database[cur][doc.docid])
                    #print("doc.docid:",database[cur])        
    fileObject = open(databasefile,'wb')    # wb = write and binary mode                 
    pickle.dump(database, fileObject)
    #print("fileObject#",fileObject)
    #print("database#",database)
    fileObject.close()
    #print ("Saved ", databasefile)
    
    return database

"""
    Load the index file if file is exist else rebuild the index
"""
def indexFileLoad(databasefile,docfile,stopfile):
    if os.path.isfile(databasefile): # This check if the index file is computed before
        fileObject = open(databasefile, 'r')  # r = read mode
        #print ("Loaded ", databasefile)
        database = pickle.load(fileObject)  
        #print"database ",database
        fileObject.close()
        docCollection = read_documents.ReadDocuments(docfile)
        global totalDocCount 
        totalDocCount = 0
        for doc in docCollection:
            totalDocCount += 1
        #print "D...",totalDocCount
    else: # the first time , not computed.
        database =indexBuilder(databasefile,docfile, stopfile)
    return database
"""
    Compute term frequency
"""       
def getTermFreq(term, docid, database):
    if database.has_key(term):
        entry = database[term]
        if entry.has_key(docid):
            return len(entry[docid])
    return 0
"""
    Compute document frequency
"""      
def getDocFreq(term, database):
    if database.has_key(term):
        return len(database[term].keys())
    return 0
"""
    Load the database file 
"""              
def fileLoadToDatabase(fileName):
    fileObject = open(fileName, 'r')  # r = read mode
    #print ("Loaded ", fileName)
    database = pickle.load(fileObject)  
    #print"database ",database
    fileObject.close()
    return database            

"""
    Function to build tf_IDF matrix like below example
         term1     term2     term2
    doc1 tf_IDF    tf_IDF    tf_IDF
    doc2 0         0         0.584 
    doc3 1.584     1.584     0.584  
""" 
def tf_IDF_MatrixBuilder(database,databasefile):
    newDatabase={}
    docIDVsTDIDFlenth={}
    global wordVsIDF
    for word in database.keys():
        entry=database[word]
        #print "entry", len(entry)
        #print "entry", entry
        newDatabase[word]={}
        #print word
        wordFrequencyInWholeDoc=0
        for docID in entry.keys():
            wordFrequencyInWholeDoc += len(entry[docID])
        for docID in entry.keys():
            wordFrequencyInDoc = len(entry[docID])
            #print"D",totalDocCount 
            #print"wordFrequencyInWholeDoc",wordFrequencyInWholeDoc
            #print"wordFrequencyInDoc",wordFrequencyInDoc
            idf= math.log(totalDocCount/wordFrequencyInWholeDoc,2)
            #print "idf",idf
            wordVsIDF[word]=idf
            tf_idf = wordFrequencyInDoc * idf;
            #print "tf_idf#",tf_idf
            if docIDVsTDIDFlenth.has_key(docID):
                #print "####",docIDVsTDIDFlenth[docID]
                #print "###Current",math.pow(tf_idf,2)
                docIDVsTDIDFlenth[docID]=docIDVsTDIDFlenth[docID]+math.pow(tf_idf,2)
                #print docID,docIDVsTDIDFlenth[docID]
            else:
                docIDVsTDIDFlenth[docID]=math.pow(tf_idf,2)
            newDatabase[word][docID]=tf_idf;
    fileObject = open(databasefile,'wb')    # wb = write and binary mode                 
    pickle.dump(newDatabase, fileObject)
    #print("newDatabase",newDatabase)
    #print "docIDVsTDIDFlenth",docIDVsTDIDFlenth
    for docID in docIDVsTDIDFlenth.keys():
        #print docID,math.sqrt(docIDVsTDIDFlenth.get(docID))
        docIDVsTDIDFlenth[docID]=math.sqrt(docIDVsTDIDFlenth.get(docID))
    
    #print "docIDVsTDIDFlenth___",docIDVsTDIDFlenth
    fileObject.close()
    database=newDatabase
    #print ("Saved ", databasefile)
    return docIDVsTDIDFlenth

"""
    Function to build tf_IDF matrix for queries like below example
         term1     term2     term2
    doc1 tf_IDF    tf_IDF    tf_IDF
    doc2 0         0         0.584 
    doc3 1.584     1.584     0.584  
""" 
def tf_IDF_MatrixBuilderForQueries(database,databasefile,):
    docIDVsTDIDFlenth={}
    if True: #not os.path.isfile(databasefile):
        newDatabase={}
        maxWordFrequencyInDoc=1;
        for word in database.keys():
            entry = database[word]
            for docID in entry.keys():
                wordFrequencyInDoc = len(entry[docID])
                if(maxWordFrequencyInDoc <=wordFrequencyInDoc):
                    maxWordFrequencyInDoc=wordFrequencyInDoc
        #print "wordVsIDF",wordVsIDF
        #print "databaseQQQ",database
        #print "maxWordFrequencyInDoc ",maxWordFrequencyInDoc
        for word in database.keys():
            #print word
            if not wordVsIDF.has_key(word):
                continue
            entry=database[word]
            #print "entryQQQQ", entry
            for docID in entry.keys():
                #print "entryQQQQ", len(entry[docID])
                queryentry=entry[docID]
                wordFrequencyInDoc = len(queryentry)
                #print"wordFrequencyInDoc",wordFrequencyInDoc
                idf= wordVsIDF[word]
                #print "idf",idf
                tf_idf = (wordFrequencyInDoc/maxWordFrequencyInDoc) * idf;
                #print "tf_idf#",tf_idf
                if docIDVsTDIDFlenth.has_key(docID):
                    #print "####",docIDVsTDIDFlenth[docID]
                    #print "###Current",math.pow(tf_idf,2)
                    docIDVsTDIDFlenth[docID]=docIDVsTDIDFlenth[docID]+math.pow(tf_idf,2)
                    #print docID,docIDVsTDIDFlenth[docID]
                else:
                    #print "###Else Current",math.pow(tf_idf,2)
                    docIDVsTDIDFlenth[docID]=math.pow(tf_idf,2)
                
                if newDatabase.has_key(docID):
                     newDatabase[docID][word]=tf_idf
                else:
                    newDatabase[docID]={}
                    newDatabase[docID][word]=tf_idf
        fileObject = open(databasefile,'wb')    # wb = write and binary mode                 
        pickle.dump(newDatabase, fileObject)
        fileObject.close()
        #print("newDatabase",newDatabase)
        #print "docIDVsTDIDFlenth",docIDVsTDIDFlenth     
        for docID in docIDVsTDIDFlenth.keys():
            #print docID,math.sqrt(docIDVsTDIDFlenth.get(docID))
            docIDVsTDIDFlenth[docID]=math.sqrt(docIDVsTDIDFlenth.get(docID))
        database=newDatabase
        #print ("Saved ", databasefile)
    else:
        fileObject = open(databasefile, 'r')  # r = read mode
        #print ("Loaded ", databasefile)
        database = pickle.load(fileObject)  
        #print"database ",database
        fileObject.close()
        for word in database.keys():
            entry = database[word]
            for docID in entry.keys():
                if docIDVsTDIDFlenth.has_key(docID):
                    #print "####",docIDVsTDIDFlenth[docID]
                    #print "###entry[docID]",entry[docID][0]
                    #print "###Current",math.pow(entry[docID][0],2)
                    docIDVsTDIDFlenth[docID]=docIDVsTDIDFlenth[docID]+math.pow(entry[docID][0],2)
                    #print docID,docIDVsTDIDFlenth[docID]
                else:
                    #print "entry[docID] ",entry[docID][0]
                    docIDVsTDIDFlenth[docID]=math.pow(entry[docID][0],2)
        for docID in docIDVsTDIDFlenth.keys():
            #print docID,math.sqrt(docIDVsTDIDFlenth.get(docID))
            docIDVsTDIDFlenth[docID]=math.sqrt(docIDVsTDIDFlenth.get(docID))    
    return docIDVsTDIDFlenth

"""
    Function to build similarity factor between query terms and document terms. Actual comparition was build over this function 
""" 
def similarityFactorBuilder(database,docIDVsTDIDFlenth,querydatabase,querydocIDVsTDIDFlenth):
    #print "database ",database
    #print "querydatabase ",querydatabase
    queryIdvsOrderedSearchedDocIDs={}
    for docId in querydatabase.keys():
        docIDvsScoreValue={}
        for word in database.keys():
            queryEntry = querydatabase[docId]
            for queryWord in queryEntry.keys():
                if queryWord == word:
                    #print "word:",word
                    entry = database[word]
                    queryEntry = queryEntry[word]
                    #print "queryEntry",queryEntry
                    #print "Entry:",entry
                    for docID in entry.keys():
                        #print"entry[docID] :",entry[docID]
                        if docIDvsScoreValue.has_key(docID):
                            val = entry[docID]*queryEntry
                            #print "val",val
                            #print"docIDvsScoreValue[docID].append(entry[docID]) ",docIDvsScoreValue[docID] 
                            docIDvsScoreValue[docID]= docIDvsScoreValue[docID]+val
                        else:
                            #print "queryEntry.get(0)",queryEntry
                            #print "[entry[docID]]",entry[docID]
                            docIDvsScoreValue[docID]=entry[docID]*queryEntry
        #print"docIDvsScoreValue",docIDvsScoreValue
        queryIdvsOrderedSearchedDocIDs[docId]=orderDocBasedOnScore(docIDvsScoreValue, docIDVsTDIDFlenth,querydocIDVsTDIDFlenth[docId])
    #print"queryIdvsOrderedSearchedDocIDs",queryIdvsOrderedSearchedDocIDs
    return queryIdvsOrderedSearchedDocIDs
"""
    Write the results into file for each query. Max 10 results returns per query doc
""" 
def resultPrinter(queryIdvsOrderedSearchedDocIDs):
    theFile = open("Result.txt", "wb")
    for queryDocID in queryIdvsOrderedSearchedDocIDs.keys():
        maxReturnResults =MAX_RESULTS_PER_QUERY
        resultsDoc = queryIdvsOrderedSearchedDocIDs[queryDocID]
        #print "resultsDoc",resultsDoc
        for docID in resultsDoc:
            #print "docID",docID[0]
            if(maxReturnResults >0):
                maxReturnResults = maxReturnResults-1
                theFile.write( str(queryDocID)+"  "+str(docID[0])+ "\n");
            else:
                continue
    theFile.close()

"""
    Order the docs based on the TF_IDF algorithm
"""             
def orderDocBasedOnScore(docIDvsScoreValue,docIDVsTDIDFlenth,queryLenth):
    docVsScore={}
    #print "queryLenth",queryLenth
    for docID in docIDvsScoreValue.keys():
        divider = docIDVsTDIDFlenth[docID]*queryLenth
        val = docIDvsScoreValue[docID]/divider
        if val:
            docVsScore[docID]=docIDvsScoreValue[docID]/divider
    #print "unsorted :",docVsScore
    docVsScore = sorted(docVsScore.items(), key=operator.itemgetter(1),reverse=True)
    #print "sorted :",docVsScore
    return docVsScore    
    
print "Document Retrieval Process started...\nProgram will load the documents, queries and stop_list words from the files \ndocuments.txt, queries.txt and stop_list.txt respectively"
startTime = time.time()
database = indexFileLoad(databasefile,docfile,stopfile)
#print"database",    database
endTime = time.time()
print "Documents Index building:",endTime-startTime
startTime = time.time()
docIDVsTDIDFlenth = tf_IDF_MatrixBuilder(database,databasefile);
endTime = time.time()
print "Documents TF_IDF processing:",endTime-startTime
startTime = time.time()
querydatabase = indexFileLoad(queryDatabasefile,queryfile,stopfile)
endTime = time.time()
print "Query Index building:",endTime-startTime
startTime = time.time()
querydocIDVsTDIDFlenth = tf_IDF_MatrixBuilderForQueries(querydatabase,queryDatabasefile);
endTime = time.time()
print "Query TF_IDF processing:",endTime-startTime
querydatabase = fileLoadToDatabase(queryDatabasefile)
database = fileLoadToDatabase(databasefile)
startTime = time.time()
queryIdvsOrderedSearchedDocIDs = similarityFactorBuilder(database, docIDVsTDIDFlenth, querydatabase, querydocIDVsTDIDFlenth)
endTime = time.time()
print "Process Query with Documents and Produce results:",endTime-startTime
resultPrinter(queryIdvsOrderedSearchedDocIDs)
print "Document Retrieval process completed and the results are stored in \"Result.txt\""
#print"docIDVsTDIDFlenth",    docIDVsTDIDFlenth
#print"querydocIDVsTDIDFlenth",    querydocIDVsTDIDFlenth
#print"queryIdvsOrderedSearchedDocIDs",    queryIdvsOrderedSearchedDocIDs
os.remove(queryDatabasefile)
os.remove(databasefile)

