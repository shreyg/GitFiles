#!/usr/bin/env python
# -*- coding: utf-8 -*-


# ************************************************************************
# Tokenization-Chris Potts tokenizer
# Feature extraction = WSD(pywsd library using simple_lesk method)
#Additions:
#1. Intensifiers, Diminishers, Negators
#2. Negators in sentiwordnet lexicon -> imcorporate their scores
#3. negators not in sentiwordnet lexicon -> negate the signs 
#4  convert 's' pos to 'a' for matching in sentiwordnet lexicon
#5. using a 2-gram approach
#6. Ignoring text in braces
#7. Ignoring stopwords and negator words from the evaluation
#8. Ignoring words having high objectivity value(i.e. >=0.9)
#*************************************************************************

import re
import htmlentitydefs
import nltk
import codecs
import string
import time
from nltk.corpus import wordnet as wn
#from pywsd.lesk import simple_lesk
#from nltk.tag import StanfordPOSTagger
from pywsd.lesk import cosine_lesk
import xml.etree.ElementTree as ET 
#st = StanfordPOSTagger("models\\english-bidirectional-distsim.tagger",path_to_jar="stanford-postagger.jar")
#nltk.internals.config_java(options='-xmx2G')
#from nltk.wsd import lesk
stopwords=["a","across","am","an","and","any","are","as","at","be","been","being","but","by","can","could","did","do","does","each","for","from","had","has","have","in","into","is","isn't","it","it'd","it'll","it's","its","of","on","or","that","that's","thats","the","there","there's","theres","these","this","those","to","under","until","up","were","will","with","would"]
intensifiers=["least",-3,"less",-1.5, "barely",-1.5,"hardly",-1.5,"almost",-1.5,"only",-0.5, "little",-0.5, "bit",-0.5, "slightly",-0.5, "marginally",-0.5, "relatively",-0.3, "mildly",-0.3, "moderately",-0.3, "somewhat",-0.3, "partially",-0.3, "sorta",-0.3, "kinda",-0.3, "fairly",-0.2, "pretty",-0.1, "rather",-0.05, "immediately",0.05, "quite",0.1,"perfectly",0.1,"consistently",0.1, "really",0.15,"clearly",0.15,"obviously",0.15,"certainly",0.15,"completely",0.15,"definitely",0.15,"absolutely",0.25,"highly",0.25,"very",0.25,"truly",0.25,"especially",0.25,"particularly",0.25,"significantly",0.25,"noticeably",0.25,"distinctively",0.25,"frequently",0.25,"awfully",0.25,"totally",0.25,"largely",0.25,"fully",0.25,"damn",0.25,"intensively",0.25,"downright",0.25,"entirely",0.3,"strongly",0.3,"remarkably",0.3,"majorly",0.3,"amazingly",0.3,"strikingly",0.3,"stunningly",0.3,"quintessentially",0.3,"unusually",0.3,"dramatically",0.3,"intensely",0.3,"extremely",0.35,"so",0.35,"incredibly",0.35,"terribly",0.35,"hugely",0.35, "immensely",0.35,"such",0.35,"unbelievably",0.4,"insanely",0.4,"outrageously",0.4,"radically",0.4, "exceptionally",0.4,"exceedingly",0.4 ,"way",0.4,"vastly",0.4,"deeply",0.4,"super",0.4,"profoundly",0.4,"universally",0.4,"abundantly",0.4,"infinitely",0.4,"enormously",0.4,"thoroughly",0.4,"passionately",0.4,"tremendously",0.4,"ridiculously",0.4,"obscenely",0.4, "extraordinarily", 0.5,"spectacularly",0.5, "phenomenally",0.5,"monumentally",0.5, "mind-bogglingly",0.5, "utterly",0.5, "more",-0.5, "most",1, "total",0.5,"monumental", 0.5, "great", 0.5,"huge",0.5, "tremendous",0.5, "complete",0.5, "absolute",0.5,"resounding",0.5, "massive", 0.5, "incredible", 0.5, "utter", 0.3, "clear", 0.3, "clearer", 0.2,"clearest", 0.5, "big", 0.3,"bigger",0.2,"biggest",0.5,"obvious",0.3,"serious", 0.3, "deep", 0.3, "deeper", 0.2,"deepest", 0.5,"considerable",0.3,"important",0.3,"extra",0.3,"major",0.3,"crucial",0.3,"high",0.3,"higher",0.2,"highest",0.5,"real",0.2,"true",0.2,"pure", 0.2, "definite", 0.2,"much",0.3,"small", -0.3, "smaller", -0.2,"smallest", -0.5, "minor",-0.3 ,"moderate", -0.3,"mild",-0.3,"slight",-0.5,"slightest", -0.9, "insignificant", -0.5,"inconsequential", -0.5, "low",-2,"lower",-1.5, "lowest", -3, "few",-2, "fewer",-1.5,"fewest",-3,"lot",0.3,"few",-0.3,"lots", 0.3]
negators=["no","not","never","nowhere","nobody","none","nothing","isn’t","couldn’t","wouldn’t","shouldn’t","ain’t","doesn't","didn't","wasn't"]
negators=[i.decode('UTF-8') if isinstance(i,basestring) else i for i in negators]
#universal_to_wn_mapper={"VERB":u'v',"NOUN":u'n',"ADJ":u'a',"ADV":u'r'}
start_time=time.time()
######################################################################
# The following strings are components in the regular expression
# that is used for tokenizing. It's important that phone_number
# appears first in the final regex (since it can contain whitespace).
# It also could matter that tags comes after emoticons, due to the
# possibility of having text like
#
#     <:| and some text >:)
#
# Most importantly, the final element should always be last, since it
# does a last ditch whitespace-based tokenization of whatever is left.

# This particular element is used in a couple ways, so we define it
# with a name:
emoticon_string = r"""
    (?:
      [<>]?
      [:;=8]                     # eyes
      [\-o\*\']?                 # optional nose
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth      
      |
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
      [\-o\*\']?                 # optional nose
      [:;=8]                     # eyes
      [<>]?
    )"""

# The components of the tokenizer:
regex_strings = (
    # Phone numbers:
    r"""
    (?:
      (?:            # (international)
        \+?[01]
        [\-\s.]*
      )?            
      (?:            # (area code)
        [\(]?
        \d{3}
        [\-\s.\)]*
      )?    
      \d{3}          # exchange
      [\-\s.]*   
      \d{4}          # base
    )"""
    ,
    # Emoticons:
    emoticon_string
    ,    
    # HTML tags:
     r"""<[^>]+>"""
    ,
    # Twitter username:
    r"""(?:@[\w_]+)"""
    ,
    # Twitter hashtags:
    r"""(?:\#+[\w_]+[\w\'_\-]*[\w_]+)"""
    ,
    # Remaining word types:
    r"""
    (?:[a-z][a-z'\-_]+[a-z])       # Words with apostrophes or dashes.
    |
    (?:[+\-]?\d+[,/.:-]\d+[+\-]?)  # Numbers, including fractions, decimals.
    |
    (?:[\w_]+)                     # Words without apostrophes or dashes.
    |
    (?:\.(?:\s*\.){1,})            # Ellipsis dots. 
    |
    (?:\S)                         # Everything else that isn't whitespace.
    """
    )

######################################################################
# This is the core tokenizing regex:
    
word_re = re.compile(r"""(%s)""" % "|".join(regex_strings), re.VERBOSE | re.I | re.UNICODE)

# The emoticon string gets its own regex so that we can preserve case for them as needed:
emoticon_re = re.compile(regex_strings[1], re.VERBOSE | re.I | re.UNICODE)

# These are for regularizing HTML entities to Unicode:
html_entity_digit_re = re.compile(r"&#\d+;")
html_entity_alpha_re = re.compile(r"&\w+;")
amp = "&amp;"

######################################################################

class Tokenizer:
    def __init__(self, preserve_case=False):
        self.preserve_case = preserve_case
        self.db = {}
        self.parse_src_file()
        #for (k,v) in self.db.iteritems():
        #    print (k,v)

    def tokenize(self, s):
        """
        Argument: s -- any string or unicode object
        Value: a tokenize list of strings; conatenating this list returns the original string if preserve_case=False
        """        
        # Try to ensure unicode:
        try:
            s = unicode(s)
        except UnicodeDecodeError:
            s = str(s).encode('string_escape')
            s = unicode(s)
        # Fix HTML character entitites:
        s = self.__html2unicode(s)
        # Tokenize:
        words = word_re.findall(s)
        # Possible alter the case, but avoid changing emoticons like :D into :d:
        if not self.preserve_case:            
            words = map((lambda x : x if emoticon_re.search(x) else x.lower()), words)
        return words

    def __html2unicode(self, s):
        """
        Internal metod that seeks to replace all the HTML entities in
        s with their corresponding unicode characters.
        """
        # First the digits:
        ents = set(html_entity_digit_re.findall(s))
        if len(ents) > 0:
            for ent in ents:
                entnum = ent[2:-1]
                try:
                    entnum = int(entnum)
                    s = s.replace(ent, unichr(entnum))	
                except:
                    pass
        # Now the alpha versions:
        ents = set(html_entity_alpha_re.findall(s))
        ents = filter((lambda x : x != amp), ents)
        for ent in ents:
            entname = ent[1:-1]
            try:            
                s = s.replace(ent,   unichr(htmlentitydefs.name2codepoint[entname]))
            except:
                pass                    
            s = s.replace(amp, " and ")
        return s
		        
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
                                       
    def disambiguateWordSenses3(self,sentence,word):        #disambiguation with simple_lesk
        #result=simple_lesk(sentence,word)
        result=cosine_lesk(sentence,word)         #result is a list of synsets of word
        #print result_list
        if result:
            pos = result.pos()
            if (pos == u's'):
                pos = u'a'
            offset = result.offset()
            pos_score=0.0
            neg_score=0.0
            if (pos, offset) in self.db:
         #       print word,pos,offset
                pos_score, neg_score = self.db[(pos, offset)]
            obj = 1.0-(pos_score+neg_score)
            #print "%%%%%%%%%%"
            #print pos_score,neg_score, obj
        else:
            obj=1.0
            pos=None
            pos_score=0.0
            neg_score=0.0
        return obj,pos,pos_score,neg_score
        
    def calculate_score2(self,filename):                      #calculates scores with disambiguateWordSenses3
        tree_pos = ET.parse(filename)
        length=0.0
        score=0.0
        root_pos = tree_pos.getroot()
        for child in root_pos:
            for r1 in child:
                if(r1.tag=='review_text'):
                    #if len(r1.text) > 150:
                    #    continue
                    length += 1.0
                    print "======================================================================"
                    print r1.text
                    prev=None
                    prev_pos_score=0.0
                    prev_neg_score=0.0
                    final_pos=0.0
                    final_neg=0.0
                    lst=[]
                    newlst=[]
                    brace_flag=0
                    #prev_overall=None
                    #curr_overall=None            
                    #r1.text=''.join(ch for ch in r1.text if ch not in string.punctuation)
                    tokenized = tok.tokenize(r1.text)
                    #stanfordPOS_lst=st.tag(tokenized)
                    for s in tokenized:
                        #if s == '(' or brace_flag == 1:
                        #    if s == '(':
                        #        brace_flag=1
                        #    elif s == ')':
                        #        brace_flag=0
                        #    continue
                        #if s in string.punctuation:
                        #    continue
                        obj,pos,pos_score,neg_score = tok.disambiguateWordSenses3(r1.text, s)
                        lst.append((s,obj,pos,pos_score,neg_score))
                    #lst=self.disambiguateWordSenses(r1.text)
                    print "======================================================================"
                    print lst
                    print "======================================================================"
                    for item in lst:
                        if item[1] < 0.9 and (item[0] not in stopwords) :
                            #if prev_pos_score > prev_neg_score:
                            #    prev_overall="pos"
                            #else:
                            #    prev_overall="neg"
                            #if item[3] > item[4]:
                            #    curr_overall="pos"
                            #else:
                            #    curr_overall="neg"
                            newlst.append((prev,prev_pos_score,prev_neg_score,item[0],item[3],item[4]))
                        prev=item[0]
                        prev_pos_score=item[3]
                        prev_neg_score=item[4]
                    print newlst
                    for x in newlst:
                        if x[0] in intensifiers:
                            val = intensifiers[intensifiers.index(x[0])+1]
                            if x[4] > x[5]:  #x[3] is positive
                                final_pos += x[4]*(1+val)
                                final_neg +=x[5]
                            elif x[4] < x[5]: #x[3] is negative
                                final_pos += x[4]
                                final_neg += x[5]*(1+val)
                        elif x[0] in negators:
                            if x[1] == 0 and x[2] == 0:
                                if x[4] > x[5]:   #x[3] is positive
                                    final_pos += x[4]*-1
                                elif x[4] < x[5]:  #x[0] is negative
                                    final_neg += x[5]*-1
                            else:
                                if x[4] > x[5]: #x[3] is positive
                                    final_pos += x[4]-x[2]
                                    final_neg += x[5]+x[2]
                                elif x[4] < x[5]: #x[3] is negative
                                    final_pos += x[4]+x[1]
                                    final_neg += x[5]-x[1]
                        else:
                            final_pos += x[1]+x[4]
                            final_neg += x[2]+x[5]
                        #if x[3] == "pos" and x[7] == "pos":
                        #   final_pos += x[1] + x[5]
                        #    final_neg += x[2] + x[6]
                        #elif x[3] == "neg" and x[7] == "neg":
                        #    final_pos += x[1] + x[5]
                        #    final_neg += x[2] + x[6]
                        #elif x[3] == "neg" and x[7] == "pos":
                        #    if x[2] > x[5]:
                        #        final_pos += x[1] - x[5]
                        #        final_neg += x[2] - x[6]
                        #    else:
                        #        final_pos += x[5] - x[1]
                        #        final_neg += x[6] - x[2]
                        #elif x[3] == "pos" and x[7] == "neg":
                        #    if x[1] > x[6]:
                        #        final_pos += x[1] - x[5]
                        #        final_neg += x[2] - x[6]
                        #    else:
                        #        final_pos += x[5] - x[1]
                        #        final_neg += x[6] - x[2]
                    print "\n"
                    print "---------------------------"
                    print final_pos,final_neg
                    #final_neg = final_neg*1.1
                    if final_pos > final_neg:
                        score +=1.0
        return score,length
###############################################################################

if __name__ == '__main__':
    tok = Tokenizer(preserve_case=False)
    #samples = (
    #    u"RT @ #happyfuncoding: this is a typical Twitter tweet :-)",
    #   u"HTML entities &amp; other Web oddities can be an &aacute;cute <em class='grumpy'>pain</em> >:(",
    #    u"It's perhaps noteworthy that phone numbers like +1 (800) 123-4567, (800) 123-4567, and 123-4567 are treated as words despite their whitespace."
    #    )
    
    final_pos,pos_len=tok.calculate_score2("C:\\Users\\notebook\\Desktop\\Python\\dummy_runs.xml")
    final_neg,neg_len=tok.calculate_score2("C:\\Users\\notebook\\Desktop\\Python\\GitFiles\\neg_small_software.xml")
    final_neg = neg_len-final_neg
    accuracy = float((final_pos+final_neg)/(pos_len+neg_len))
    print "****************************************************************************"
    print final_pos, pos_len
    print "****************************************************************************"
    print final_neg,neg_len
    print "****************************************************************************"
    print accuracy
    print ("---- %s seconds-------------" % (time.time()-start_time))