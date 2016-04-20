from __future__ import division
import os
import sys
from collections import defaultdict
import math
import time
from collections import Counter

pathNeg = "IMDB/neg/"
pathPos = "IMDB/pos/"
 
BEGIN = 0
END = 9
 
def readDataSet(classification):
    lst = list()
    for file in os.listdir (classification):
        valueName = int(file.split(".txt")[0])
        if valueName >= BEGIN and valueName <= END:
             lst.append(valueName)
    return lst
 
def splitGroups(lst, lenGroups):
    return [lst[i:i + lenGroups] for i in range(0, len(lst), lenGroups)]
 
def timing(f):
	def wrap(*args):
    		time1 = time.time()
    		ret = f(*args)
    		time2 = time.time()
    		print '%s function took %0.3f ms' % (f.func_name, (time2-time1)*1000.0)
    		return ret
    	return wrap

# Examples is a set of text documents along with their
# target values. Classes is the set of all possible
# target values. This function learns the probability
# terms P(Wk|Vj), describing the probability that a
# randomly draw words from a document in class Vj
# will be the English word Wk. It also learns the 
# class prior probabilities P(Vj).

@timing
def learnNaiveBayesText(examplesPos, examplesNeg, classes):
 
	# 1. Collect all words, punctuation, and other tokens
	# that occur in Examples
	
	t0 = time.time()
	
	print "parsing positive examples..."
	wordsPos, lenPos = getWords(examplesPos, pathPos)
	print "parsing negative examples..."
	wordsNeg, lenNeg = getWords(examplesNeg, pathNeg)
 	
 	t1 = time.time()
	print "getWords time %0.3f ms" % ((t1-t0)*1000.0)
 	

	print lenPos
	print lenNeg
	
	t0 = time.time()
	
	print "creating vocabulary..."
	vocabulary =  wordsPos + wordsNeg
	lenVocabulary =  len(vocabulary)
	print "vocabulary size" , lenVocabulary
	
	t1 = time.time()
	print "vocabulary time %0.3f ms" % ((t1-t0)*1000.0)


	probAposterioriDict = defaultdict(dict)
	prioriPClass = dict()
	# 2. Calculate the required P(Vj) and P(Wk|Vj) probability terms
	examplesLen = len(examplesPos) + len(examplesNeg)
	for i in classes:
		if i == 'pos':
			docs = len(examplesPos)
			text = wordsPos
			n = lenPos
			#print "Pos docs text e n: ", docs, n
		else:
			docs = len(examplesNeg)
			text = wordsNeg
			n = lenNeg
			#print "Neg docs text e n: ", docs, n
			
		prioriPClass[i] = docs/examplesLen
		#print docs, examplesLen
		#print "priori class", prioriPClass
		
		
		tempoTotal = 0;
		for j in vocabulary:
			#print j
			#exit();
			t0 = time.time()
			nK = text[j]
			#print j, nK
			t1 = time.time()
			tempoTotal += t1-t0 

			#print "NK: %r %r vezes" % (j, nK)
			if i == 'pos':
				probAposterioriDict [j] ['probPos'] = (nK + 1) / (n + lenVocabulary)
				
			else:
				probAposterioriDict [j] ['probNeg'] = (nK + 1) / (n + lenVocabulary)
		print "count dict time %0.3f ms" % (tempoTotal*1000.0)
		



	#for k in probAposterioriDict:
			#print k, probAposterioriDict[k]

	return probAposterioriDict, prioriPClass

def classifyNaiveBayesText(doc, vocabulary, prioriP):
	# Return the estimated target value for the document Doc. 

	probPos = 1
	probNeg = 1
	#t0 = time.time()
	#print doc
	for i in doc:
		if vocabulary.get(i, 0) != 0:
			#print probPos, probPos *0.0000000005
			probPos =  probPos + math.log(vocabulary[i]['probPos'])
			probNeg =  probNeg + math.log(vocabulary[i]['probNeg'])
	
	probPos = probPos + math.log(prioriP['pos'])
	probNeg = probNeg+ math.log(prioriP['neg'])
	
	#print "Pos", probPos
	#print "Neg", probNeg
	
	#t1 = time.time()
	#print "vocabulary time %0.3f ms" % ((t1-t0)*1000.0)
	
	if probPos >= probNeg:
		return 'p'
	else:
		return 'n'
	
def getWords(lst, path):
	words = list()
	wordsAll = list()
	dictio = Counter()
	
	for i in lst:
		arq = open(path + str(i) + ".txt", "r")
		words = arq.read().split(" ")
		wordsAll += words
		for j in words:
			j = j.lower().replace(",", "").replace("!", ""). replace("?", "").replace("(", "").replace(")", "").replace(".", "").replace(":","")
			try:
				dictio[j] +=1
			except KeyError:
				dictio[j] = 1
	#y = 0
	#for x in words:
		#words[y] = words[y].lower()
		#words[y] = words[y].replace(",", "").replace("!", ""). replace("?", "").replace("(", "").replace(")", "").replace(".", "").replace(":","")
		#y += 1

	print "\n"	
	
	return dictio,  len(wordsAll)
 

def getWordsFile(file, path):
	words = list()
	dictio = dict()
	arq = open (path + str(file) + ".txt", "r")
	words = words + arq.read().split(" ")
	
	for j in words:
			j = j.lower().replace(",", "").replace("!", ""). replace("?", "").replace("(", "").replace(")", "").replace(".", "").replace(":","")
			try:
				dictio[j] +=1
			except KeyError:
				dictio[j] = 1
	
	return dictio
 

 # ------------------------------ MAIN  ------------------------------
 
 
lenGroups = int ((END - BEGIN + 1)/10)

lstPos =  splitGroups(sorted(readDataSet(pathPos)), lenGroups)
for i in lstPos:
	i.append('p')

lstNeg =  splitGroups(sorted(readDataSet(pathNeg)), lenGroups)
for i in lstNeg:
	i.append('n')
 
fullDataSet = lstPos + lstNeg
 
#print fullDataSet
 
truePositive = 0
falsePositive = 0
 
trueNegative = 0
falseNegative = 0
 
# Cross-Validation 10 times
for i in range(0,10):
	trainningGroupPos = list()
	trainningGroupNeg = list()
	classificationGroup = list()
	
	trainningGroup = fullDataSet[0:i] + fullDataSet[i+1:]
	classificationGroup = fullDataSet[i]
	
	print classificationGroup
	
	#print i, trainningGroup
	
	print i
	
	for j in trainningGroup:
		if j[-1] == 'n':
				for anElement in j:
					if anElement != 'n': 
						trainningGroupNeg.append(anElement)
	
		if j[-1] == 'p':
			for anElement in j:
				if anElement != 'p': 
					trainningGroupPos.append(anElement)
	#print classificationGroup
	#print trainningGroupPos
	#print trainningGroupNeg

	probAposterioriDict, prioriPClass = learnNaiveBayesText(trainningGroupPos, trainningGroupNeg, ['pos', 'neg'])
	
	correctClass = classificationGroup[-1]
	path = ""
	
	if correctClass == 'p':
			path = pathPos
	else:	
			path = pathNeg
	
	for k in range (0, len(classificationGroup) - 1):
		doc = getWordsFile(k,path)
		classPredicted = classifyNaiveBayesText(doc, probAposterioriDict, prioriPClass)
		if correctClass == 'p':
			if classPredicted == 'p':
				truePositive += 1
			else:
				falseNegative +=1
		else:
			if classPredicted == 'p':
				falsePositive += 1
			else:
				trueNegative += 1
		

	
print truePositive, falseNegative
print falsePositive, trueNegative
