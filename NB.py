import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
import xml.etree.ElementTree as ET 
import re

tree_pos=ET.parse("C:\\Users\\notebook\\Desktop\\Python\\GitFiles\\pos_small_electronics.xml")
root_pos=tree_pos.getroot()
posfeats=[]
for child in root_pos:
	for r1 in child:
		if(r1.tag=='review_text'):
			pos_dict = {}
			for w in re.findall(r"[\w']+",r1.text):	
				pos_dict[w]=True
			tup_pos=(pos_dict,'pos')
			posfeats.append(tup_pos)
 
tree_neg=ET.parse("C:\\Users\\notebook\\Desktop\\Python\\GitFiles\\neg_small_electronics.xml")
root_neg=tree_neg.getroot()
negfeats=[]
for child in root_neg:
	for r1 in child:
		if(r1.tag=='review_text'):
			neg_dict = {}
			for w in re.findall(r"[\w']+",r1.text):
				neg_dict[w]=True
			tup_neg=(neg_dict,'neg')
			negfeats.append(tup_neg)
				
negcutoff = len(negfeats)*3/4
poscutoff = len(posfeats)*3/4

 
for i in range(0,2):
	print negfeats[i]
	print '\n------------------------------------------------\n'

trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]

print 'train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats))
 
classifier = NaiveBayesClassifier.train(trainfeats)
print 'accuracy:', nltk.classify.util.accuracy(classifier, testfeats)
classifier.show_most_informative_features()
