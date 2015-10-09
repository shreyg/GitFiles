# ************************************************************************
# Tokenization-Bag of words
# Feature extraction = WSD(pywsd library using maximum similarity method)
#*************************************************************************

import codecs
import nltk
import re
import xml.etree.ElementTree as ET 
from pywsd import disambiguate
from pywsd.similarity import max_similarity as maxsim


class wsd2:
    def __init__(self):
        self.db = {}
        self.parse_src_file()

    def parse_src_file(self):
        lines = codecs.open("SentiWordNet_3.0.0_20130122.txt", "r", "utf8").read().splitlines()
        lines = filter((lambda x : not re.search(r"^\s*#", x)), lines)
        #min=1000
        #max=0
        for i, line in enumerate(lines):
            fields = re.split(r"\t+", line)
            fields = map(unicode.strip, fields)
            try:            
                pos, offset, pos_score, neg_score, synset_terms, gloss = fields
            except:
                sys.stderr.write("Line %s formatted incorrectly: %s\n" % (i, line))
            if pos and offset:
                offset = int(offset)
                self.db[(pos, offset)] = (float(pos_score), float(neg_score))
                #try:
                  #  if float(pos_score) == 0.0:
                 #       continue
                #    ratio = (float)(float(pos_score)/float(neg_score))
               #     print ratio
              #      if min > ratio:
             #           min=ratio
            #        if max < ratio:
           #             max=ratio
          #      except:
         #           continue
        #print min,max       
        #for i in self.db.iteritems():
           # print i

    def disambiguateWordSenses2(self,sentence):           #disambiguation without simple_lesk
        synsets = disambiguate(sentence)
        print synsets
        #print synsets
        lst=[]
        for word in synsets:
            if word[1]:
                pos=word[1].pos()
                offset=word[1].offset()
                print "$$$$$$$$$$$$$$$$"
                print word[1], pos,offset
                pos_score=0.0
                neg_score=0.0
                if(pos,offset) in self.db:
                    pos_score,neg_score = self.db[(pos,offset)]
                    #print word[0],pos_score,neg_score
                obj = 1.0-(pos_score+neg_score)
            else:
                pos = None
                obj=1.0
                pos_score=0.0
                neg_score=0.0
            lst.append((word[0],obj,pos,pos_score,neg_score))
        return lst
        
    def calculate_score(self,filename):             #calculates scores using disambiguateWordSenses2
        tree_pos = ET.parse(filename)
        score=0.0
        len=0.0
        root_pos = tree_pos.getroot()
        for child in root_pos:
            for r1 in child:
                if(r1.tag=='review_text'):
                    len += 1.0
                    print "======================================================================"
                    print r1.text
                    prev=None
                    prev_pos_score=0.0
                    prev_neg_score=0.0
                    final_pos=0.0
                    final_neg=0.0
                    newlst=[]
                    lst=self.disambiguateWordSenses2(r1.text)
                    #print "======================================================================"
                    #print lst
                    #print "======================================================================"
                    for item in lst:
                        if item[1] < 0.9 and item[2]!= 'n':
                            newlst.append((prev,prev_pos_score,prev_neg_score,item[0],item[3],item[4]))
                        prev=item[0]
                        prev_pos_score=item[3]
                        prev_neg_score=item[4]
                    print newlst
                    for x in newlst:
                        final_pos += x[1] + x[4]
                        final_neg += x[2] + x[5]
                    print "\n"
                    print "---------------------------"
                    print final_pos,final_neg
                    if final_pos > final_neg:
                        score +=1.0
        return score,len
       
          
if __name__ == '__main__':
    ws = wsd2()
    final_pos,pos_len=ws.calculate_score("C:\\Users\\notebook\\Desktop\\Python\\dummy_runs.xml")
    #final_neg,neg_len=ws.calculate_score("C:\\Users\\notebook\\Desktop\\Python\\neg.xml")
    #final_neg = neg_len-final_neg
    #accuracy = float((final_pos+final_neg)/(pos_len+neg_len))
    #print "****************************************************************************"
    print final_pos, pos_len
    #print "****************************************************************************"
    #print final_neg,neg_len
    #print "****************************************************************************"
    #print accuracy