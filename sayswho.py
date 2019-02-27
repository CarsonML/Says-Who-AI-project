from nltk.stem import *
import numpy as np
from random import *
import string

#from sklearn.svm import SVC 
#look int different librarys w/ same commands
from sklearn.cross_validation import train_test_split
referencex = {
	0:[],
	1:[],
	2:[],
	3:[],
	4:[],
	5:[],
	6:[],
	7:[],
	8:[],
	9:[]
}
referencey = {
	0:[],
	1:[],
	2:[],
	3:[],
	4:[],
	5:[],
	6:[],
	7:[],
	8:[],
	9:[]
}
#function that lowercases and removes punctuation
def removepunctuation(strig):
	exclude = set(string.punctuation)
	strig = ''.join(ch for ch in strig if ch not in exclude)
	strig = strig.lower

filename = "lyrics.txt"
file = open(filename, "r")
content= file.read()
file.close()
content = content.split("\n")
shuffle(content) #reorder data so you don't end up with same grops

#Code to remove duplicates
fulllist = []
for item in content:
	if item not in fulllist:
		fulllist.append(item)
content = fulllist

for x in range(len(content)):
		content[x] = content[x].split("\t")

#get rid of all data shorter than 2 or more than 100
actual = []
for item in content:
	if len(item[1]) >= 2 and len(item[1]) <= 100:
		actual.append(item)
	else:
		print(item)
content = actual

#get rid of all data that starts with 2 non letters or numbers
actual2 = []
for x in range(len(content)):
	if content[x][1][0].isdigit() == False and content[x][1][0].isalpha() == False and content[x][1][1].isdigit() == False and content[x][1][1].isalpha() == False:
		print("oopsie")
	else:
		actual2.append(content[x])
content = actual2

#remove punctuation
for x in range(len(content)):
	removepunctuation(content[x])

#split into x and y data
ylist = []
xlist = []
for x in range(len(content)):
	content[x].reverse()
	ylist.append(content[x][1])
	del content[x][1]
	content[x] = content[x][0].split(" ")
	stemmer = PorterStemmer()
	for y in range(len(content[x])):
		content[x][y]= stemmer.stem(content[x][y])
	replacement = ""
	for item in content[x]:
		replacement += item
		replacement += " "
	content[x] = replacement
	xlist.append(content[x])

X = xlist
Y = ylist
finalX = []
finalY = []
finalXtest = []
finalYtest = []
def gropudata(X,Y): #create training and testing
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.16)
	print("inside the function X")
	print (X_train)
	print("inside the function Y")
	print (Y_train)
	for item in X_train:
		finalX.append(item)
	for item in Y_train:
		finalY.append(item)
	for item in X_test:
		finalXtest.append(item)
	for item in Y_test:
		finalYtest.append(item)


	groupsize = int(len(Y_train)/10)

	for x in range(0, 10):
		print (x)
		for y in range(0, groupsize):
			a = randint(0,(len(Y_train)-1))
			referencex[x].append(X_train[a])
			referencey[x].append(Y_train[a])
			del X_train[a]
			del Y_train[a]

def processy(words): #make everything taylorswift a 0 and beatles into a 1
	for x in range(len(words)):
		if words[x] == "taylor_swift":
			words[x] = 0
		else:
			words[x] = 1
def makevector(words): #create vectorizer on given data
	from sklearn.feature_extraction.text import CountVectorizer
	vectorizer = CountVectorizer()
	vectorizer.fit(words)
	return vectorizer

def vectorize(vectorizer, words): #use given vectorizer to vectorize words
	from sklearn.feature_extraction.text import CountVectorizer
	X = vectorizer.transform(words)
	return (X.toarray()).tolist()


#split data into 10, make 10 bayes models, return the model(this part isn't used) and the average score
def createbayes():
	from sklearn.naive_bayes import GaussianNB
	bayes = {}
	print("making bayes")
	for x in range(0,10):
		megaX = []
		megaY = []
		smalltestx = []
		smalltesty = []
		for y in range(10):
			if y != x:
				for item in referencex[y]:
					megaX.append(item)
				for item in referencey[y]:
					megaY.append(item)
			else:
				for item in referencex[y]:
					smalltestx.append(item)
				for item in referencey[y]:
					smalltesty.append(item)
		vectorizer = makevector(megaX)
		megaX = vectorize(vectorizer, megaX)
		smalltestx = vectorize(vectorizer, smalltestx)
		processy(megaY)
		processy(smalltesty)
		megaX = np.array(megaX)
		megaY = np.array(megaY)
		smalltestx = np.array(smalltestx)
		smalltesty = np.array(smalltesty)

		print("making model on ")
		print(megaX, megaY)
		model = GaussianNB()
		model.fit(megaX, megaY)
		print ("done with model")
		score = model.score(smalltestx, smalltesty)
		bayes[x] = [model, score]
	scores = []
	for x in range(len(bayes)):
		scores.append(bayes[x][1])
	print(scores)
	higest = scores.index(max(scores))
	bayesmodel = bayes[higest][0]
	return [bayesmodel, (sum(scores)/len(scores))]

#split data into 10, make 10 svm models, return the model(this part isn't used) and the average score
def createsvm():
	from sklearn.svm import SVC
	print("making svm")
	svm = {}
	for x in range(0,10):
		megaX = []
		megaY = []
		smalltestx = []
		smalltesty = []
		for y in range(10):
			if y != x:
				for item in referencex[y]:
					megaX.append(item)
				for item in referencey[y]:
					megaY.append(item)
			else:
				for item in referencex[y]:
					smalltestx.append(item)
				for item in referencey[y]:
					smalltesty.append(item)
		vectorizer = makevector(megaX)
		megaX = vectorize(vectorizer, megaX)
		smalltestx = vectorize(vectorizer, smalltestx)
		processy(megaY)
		processy(smalltesty)
		megaX = np.array(megaX)
		megaY = np.array(megaY)
		smalltestx = np.array(smalltestx)
		smalltesty = np.array(smalltesty)

		print("making model on ")
		print(megaX, megaY)
		model = SVC(gamma='auto')
		model.fit(megaX, megaY)
		print ("done with model")
		score = model.score(smalltestx, smalltesty)
		svm[x] = [model, score]
	scores = []
	for x in range(len(svm)):
		scores.append(svm[x][1])
	print(scores)
	higest = scores.index(max(scores))
	svmmodel = svm[higest][0]
	return [svmmodel, (sum(scores)/len(scores))]


gropudata(X,Y)
print("this is the final X test")
print (finalX)
print("this is the final Y test")
print (finalY)
bayes = createbayes()
svm = createsvm()

if bayes[1] > svm[1]: #make new Bayes model if it works better
	from sklearn.naive_bayes import GaussianNB
	finalmodel = GaussianNB()
	vectorizer = makevector(finalX)
	finalX = vectorize(vectorizer, finalX)
	finalXtest = vectorize(vectorizer, finalXtest)
	processy(finalY)
	processy(finalYtest)
	finalX = np.array(finalX)
	finalY = np.array(finalY)
	finalXtest = np.array(finalXtest)
	finalYtest = np.array(finalYtest)
	finalmodel.fit(finalX, finalY)
	score = finalmodel.score(finalXtest, finalYtest)
	print("bayes model was used, accuracy was:")
	print(score*100)
else: #make svm model if it works better
	from sklearn.svm import SVC
	finalmodel = SVC(gamma='auto')
	vectorizer = makevector(finalX)
	finalX = vectorize(vectorizer, finalX)
	finalXtest = vectorize(vectorizer, finalXtest)
	processy(finalY)
	processy(finalYtest)
	finalX = np.array(finalX)
	finalY = np.array(finalY)
	finalXtest = np.array(finalXtest)
	finalYtest = np.array(finalYtest)
	finalmodel.fit(finalX, finalY)
	finalmodel.fit(finalX, finalY)
	score = finalmodel.score(finalXtest, finalYtest)
	print("SVM model was used, accuracy was:")
	print(score*100)

