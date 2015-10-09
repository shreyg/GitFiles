#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ************************************************************************
# Tokenization-Chris Potts tokenizer
# Feature extraction = WSD(simple implementation)
#*************************************************************************
import re
import htmlentitydefs
import nltk
import codecs
from nltk.corpus import wordnet as wn
import xml.etree.ElementTree as ET 


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

    def tokenize_random_tweet(self):
        """
        If the twitter library is installed and a twitter connection
        can be established, then tokenize a random tweet.
        """
        try:
            import twitter
        except ImportError:
            print "Apologies. The random tweet functionality requires the Python twitter library: http://code.google.com/p/python-twitter/"
        from random import shuffle
        api = twitter.Api()
        tweets = api.GetPublicTimeline()
        if tweets:
            for tweet in tweets:
                if tweet.user.lang == 'en':            
                    return self.tokenize(tweet.text)
        else:
            raise Exception("Apologies. I couldn't get Twitter to give me a public English-language tweet. Perhaps try again")

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
		
    def disambiguateWordSenses(self,sentence,word):
        wordsynsets = wn.synsets(word)
        bestScore = 0.0
        result = None
        for synset in wordsynsets:
            for w in nltk.word_tokenize(sentence):
                score = 0.0
                for wsynset in wn.synsets(w):
                    sim = wn.path_similarity(wsynset, synset)
                    if(sim == None):
                        continue
                    else:
                        score += sim
                    if (score > bestScore):
                        bestScore = score
                        result = synset
        if result:
            pos = result.pos()
            offset = result.offset()
            pos_score=0.0
            neg_score=0.0
            if (pos, offset) in self.db:
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
                                
    def calculate_score(self,filename):
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
                    lst=[]
                    newlst=[]
                    tokenized = tok.tokenize(r1.text)
                    for s in tokenized:
                        obj,pos,pos_score,neg_score = tok.disambiguateWordSenses(r1.text, s)
                        lst.append((s,obj,pos,pos_score,neg_score))
                    #lst=self.disambiguateWordSenses(r1.text)
                    print "======================================================================"
                    print lst
                    print "======================================================================"
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
###############################################################################

if __name__ == '__main__':
    tok = Tokenizer(preserve_case=False)
    #samples = (
    #    u"RT @ #happyfuncoding: this is a typical Twitter tweet :-)",
    #   u"HTML entities &amp; other Web oddities can be an &aacute;cute <em class='grumpy'>pain</em> >:(",
    #    u"It's perhaps noteworthy that phone numbers like +1 (800) 123-4567, (800) 123-4567, and 123-4567 are treated as words despite their whitespace."
    #    )
    final_pos,pos_len=tok.calculate_score("C:\\Users\\notebook\\Desktop\\Python\\pos.xml")
    final_neg,neg_len=tok.calculate_score("C:\\Users\\notebook\\Desktop\\Python\\neg.xml")
    final_neg = neg_len-final_neg
    accuracy = float((final_pos+final_neg)/(pos_len+neg_len))
    print "****************************************************************************"
    print final_pos, pos_len
    print "****************************************************************************"
    print final_neg,neg_len
    print "****************************************************************************"
    print accuracy
    