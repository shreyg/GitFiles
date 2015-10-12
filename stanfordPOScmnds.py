from nltk.tag import StanfordPOSTagger
st = StanfordPOSTagger("models\\english-bidirectional-distsim.tagger",path_to_jar="stanford-postagger.jar")
#print(st.tag([u'paid', u'double', u'what', u'i', u'found', u'at', u'another', u'site', u'outside', u'of', u'amazon', u'/', u'this', u'store', u'-', u'never', u'again', u'!']))
print(st.tag("These sunglasses are all right . They were a little crooked but is still cool..".split()))
print(st.tag([u'These',u'sunglasses',u'are',u'all',u'right',u'They',u'were',u'a',u'little',u'crooked',u'but',u'is',u'still',u'cool']))
print(st.tag([u'These',u'sunglasses',u'are',u'all',u'right']))