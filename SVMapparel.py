import nltk.classify.svm
import xml.etree.ElementTree as ET
import re
from sklearn.svm import LinearSVC
from nltk.classify.scikitlearn import SklearnClassifier

tree_pos=ET.parse("C:\\Users\\notebook\\Desktop\\Python\\pos.xml")
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
 
tree_neg=ET.parse("C:\\Users\\notebook\\Desktop\\Python\\neg.xml")
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

cls_set = ['pos', 'neg']
 
for i in range(0,2):
	print negfeats[i]
	print '\n------------------------------------------------\n'

trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]

print 'train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats))

classif = SklearnClassifier(LinearSVC())
classif.train(trainfeats)
print classif.labels()
test_skl = []
t_test_skl = []
for d in testfeats:
 test_skl.append(d[0])
 t_test_skl.append(d[1])
 
print(set(t_test_skl))

result = []
for item in test_skl:
	p = classif.classify(item)
	result.append(p)
	
print len(result)
print len(t_test_skl)

score = 0.0
for i in range(0,len(result)):
	if result[i] == t_test_skl[i]:
		score = score + 1.0

print score/len(result)

from sklearn.metrics import classification_report
# getting a full report
print classification_report(t_test_skl, result, labels=list(set(t_test_skl)),target_names=cls_set)
