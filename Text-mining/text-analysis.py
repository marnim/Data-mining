import os
import operator
import nltk
from math import log10,sqrt
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from collections import Counter

speeches = {}
dfs = {}
idfs = {}
tfs = {}
new_tfs = {}
postinglists = {}
sortedpostinglists = {}

stemmer = PorterStemmer()
corpusroot = './dataset'
#corpusroot = './Sample_files'

def get_dfs(token):
    if token not in idfs:
        return -1
    else:
        return idfs[token]

def get_tfidf(filename, token):
    if filename not in speeches:
        return None
    if token not in speeches[filename]:
        return 0
    else:
        return speeches[filename][token]

def calc_tfidf(withidf):
    for filename,tf_vector in speeches.items():
        #if filename == "2012-10-03.txt":
            # for token, tf_value in sorted(tf_vector.items()):
            #     print (token, tf_value)
            #print ("filename, tf_vector", filename, sorted(tf_vector), len(tf_vector))
        veclen = 0.0
        # tf_vector = sorted(tf_vector)
        # print (sorted(tf_vector))
        for token, tf_value in tf_vector.items():
            #print (token, tf_value)
            if withidf :
                tf_value = (1 + log10(tf_value))*idfs[token]
            else:
                tf_value = (1 + log10(tf_value))
            tf_vector[token] = tf_value
            veclen += pow(tf_value, 2)
        if filename == "2012-10-03.txt":
            #print ('veclen', filename, sqrt(veclen), veclen)
            print (len(tf_vector))
            print (tf_vector)
            #print (tf_vector['health'])

        if veclen > 0 :

            # print ('q_Vec', q_vector[token])
            for token, tf_value in tf_vector.items():
                tf_vector[token] = tf_vector[token]/sqrt(veclen)
                #if filename == '2012-10-03.txt':
                    #print (token)
                    #print('token, tf value, Veclor length ', token, tf_value, tf_vector[token], sqrt(veclen))
    return speeches

def calc_dfs(tf_vector):
    for token in set(tf_vector):
        if token not in dfs:
            dfs[token] = 1
        else:
            dfs[token] = dfs[token] + 1
    return dfs

def calc_idfs():
    for filename, tf_vector in speeches.items():
        dfs = calc_dfs(tf_vector)
    N = len(speeches)
    for token, df in dfs.items():
        idfs[token] = log10(N / df)
    return idfs

def calc_qtfidf(q_vector):
    veclen = 0.0
    for token, tf_value in q_vector.items():
        tf_value = (1 + log10(tf_value))
        q_vector[token] = tf_value
        veclen += pow(tf_value, 2)
    if veclen > 0:
        # print ('Veclor length ', sqrt(veclen))
        # print ('q_Vec', q_vector[token])
        for token in q_vector:
            q_vector[token] /= sqrt(veclen)
    #print ("QUERY", q_vector)
    return q_vector

def readfiles(corpusroot):
    for filename in os.listdir(corpusroot):
        file = open(os.path.join(corpusroot, filename), "r", encoding='UTF-8')
        doc = file.read()
        file.close()
        doc = doc.lower()
        tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
        tokens = tokenizer.tokenize(doc)
        stop_words = sorted(stopwords.words('english'))
        stemmed_tokens = []
        for token in tokens:
            if not token in stop_words:
                stemmed_tokens.append(stemmer.stem(token))
        tf_vector = Counter(stemmed_tokens)
        speeches[filename] = tf_vector
    print (speeches["2012-10-03.txt"])
    return speeches

def topklist(token, k):
    if token in sortedpostinglists:
        return sortedpostinglists[token][:k]
    else:
        return None

def query(text):
    q_doc = text.lower()
    tokenizer = RegexpTokenizer(r'[a-z]+')
    tokens = tokenizer.tokenize(q_doc)
    stop_words = sorted(stopwords.words('english'))
    query_tokens = []
    for token in tokens:
        if not token in stop_words:
            query_tokens.append(stemmer.stem(token))
    q_vector = Counter(query_tokens)
    print (q_vector)
    q_tfidf = calc_qtfidf(q_vector)
    print(q_tfidf)


    scores = {}
    seeninall = {}
    for token in q_tfidf:
        toplist = topklist(token, 15)
        #print(toplist,'\n')
        #print('--------------------token-------------------', token)
        if toplist is None:
            continue
        for item in toplist:
            #print('item---------------------------',item)
            for filename in item[0:1]:
                #print ('method1',filename)
                if filename not in scores:
                    scores[filename] = 0
                    seeninall[filename] = True
                    #print('score', scores)
            #print('score', scores)
            #print('seeninall', seeninall)

        # for filename in [item[0] for item in toplist]:
        #     print("method2",filename)
        #     if filename not in scores:
        #         scores[filename] = 0
        #         seeninall[filename] = True
        # print (scores)
        # print(seeninall)
    if not scores:
        return (None, 0)
    for token in q_tfidf:
        toplist = topklist(token, 15)
        if toplist is None:
            continue
        # print('\n','-------------------token------------------', token)
        # print(toplist,'\n')
        for filename in scores:
            if filename in [item[0] for item in toplist]:
                #print('before', scores[filename])
                scores[filename] += dict(toplist)[filename] * q_tfidf[token]
                #print('------------Multiplying with own value',scores[filename])
            else:
                scores[filename] += toplist[-1][1] * q_tfidf[token]
                seeninall[filename] = False
                #print('Multiplying with 10th value', scores[filename])
                #print(seeninall)
            #print('filename:', filename, scores[filename], '\n\n\n')
    filename, score = max(scores.items(), key=operator.itemgetter(1))
    #print (filename)
    if seeninall[filename]:
       return (filename, score)
    else:
       return ("fetch more", 0)


def postings_list():
    for filename, tfidf_vector in speeches.items():
        for token, tfidf_score in tfidf_vector.items():
            if token not in postinglists:
                postinglists[token] = {}
            postinglist = postinglists[token]
            postinglist[filename] = tfidf_score

    for token in idfs:
        postinglist = postinglists[token]
        sortedpostinglists[token] = sorted(postinglist.items(), key=operator.itemgetter(1), reverse=True)
    #print(sortedpostinglists)

#get all the tokens from the files and their corresponding frequency in each file
speeches = readfiles(corpusroot)

#get the idf score of the tokens
idfs = calc_idfs()

#get the tfidf scores of the tokens
tfidf_scores = calc_tfidf(True)
# with open('file.txt', 'w') as f:
#     print(tfidf_scores, file=f)
#posting list of all the tokens
postings_list()

# print( query("data mining"))
# print("%.12f" % get_dfs('data'))
# print("%.12f" % get_tfidf('test1.txt','data'))

# RESULTS
# machine learning = ('test2.txt', 0.7643868159920214)
# data mining = ('test1.txt', 0.46656262497581397)
# 0.176091259056
# 0.329909595969
# print("%.12f" % get_tfidf("2012-10-03.txt","health"))
# print(query("particular constitutional amendment"))
# print("%.12f" % get_tfidf('2012-10-16.txt', 'hispan'))
# print("%.12f" % get_dfs('hispan'))
#
print("(%s, %.12f)" % query("terror attacks Aniket"))
# #(2004-09-30.txt, 0.021958318634)
print("(%s, %.12f)" % query("terror attacks Ani8ket"))
# print("(%s, %.12f)" % query("some more terror attack"))#(2004-09-30.txt, 0.026893338131)
# print("%.12f" % get_tfidf('1960-10-13.txt','identifi'))#0.013381177224
# print("%.12f" % get_tfidf('1976-10-22.txt','institut'))#0.006927937352
# print("%.12f" % get_dfs('andropov'))#1.477121254720
# print("%.12f" % get_dfs('identifi'))#0.330993219041
# print("%.12f" % get_tfidf("2012-10-03.txt","health"))