# NLP course

# Text preprocessing

# A vector is a representation of text numerically
# Vectorising - transforming text into numbers

# Vectorising in Python can be done using the SciKit-Learn CountVectorizer

vectorizer = CountVectorizer()      # Step 1: Initialize a count vectorizer object

vectorizer.fit(list_of_documents_train)     # Step: 2 Fit the count vectorizer object to your training corpus

Xtrain = vectorizer.transform(list_of_documents_train)

# All in one step

Xtrain = vectorizer.fit_transform(list_of_documents_train)      # Alternatively fit transform give you back the matrix of counts and fit the model at the same time
Xtest = vectorizer.transform(list_of_documents_test)        # Step 3: Transform the test corpus

# Other methods or creating vectors yourself using Python is through Numpy or Scipy (encouraged to try this exercise)
# Scipy is preferred to Numpy since because it contains sparse matrices

# Normalization

# Count vectors will have size disparities if documents in the corpus varies in size (more words = higher counts)
# This is bad since we want similar documents to be "close" to each other in the vector space
# Therefore we use vector normalization
# CountVectorizer does not have normalisation, but TF-IDF in SciKit-Learn does

# Tokenization - processing a string of text into a list of tokens
# A token can be a word or a character

# Basic tokenization in Python can be done using the string fuction split()

my_string = "My name is Siri"
my_string.split()

>>> ['My', 'name', 'is', 'Siri']        # By default the split function split using whitespaces

# There are more advanced ways of tokenization in Python 
# that includes additional features using the CountVectorizer class

CountVectorizer(analyzer = 'word')      # Word-based
CountVectorizer(analyzer = 'char')      # Character-based

# There is also subword-based tokenization which is somewhere in between words and character tokenization
# For example tokenizing walk, walks, walking and walked as walk

# Important text features to consider when tokenizing: Punctuations, lowercase, stopwords

# Punctuations can sometimes add information to a text especially in text analysis such as sentiment analysis
# It can therefore be chosen to keep punctuation in tokenization for e.g. "I hate cats?" >>> ['I', 'hate', 'cats', '?']
# Whether to include or excllude puncuations depends on the experiment
# SciKit_Learn CountVectorizer ignores puncuations

# Transform corpus into lowercase in Python, two options:

my_string.lower()       # With built in Python function

CountVectorizer(lowercase = True)       # With SKLearn library

# Remove accents in the corpus. Words with and without accents usually has the same meaning.

CountVectorizer(strip_accents = True)

# Removing stopwords
# Stop words take up space (high dimensionality) and do not add value to our dataset

# In SKLearn

CountVectorizer(stop_words = 'english')     # Remove english stopwords
CountVectorizer(stop_words = user_defined_list)     # Remove stopwords from a user defined list of stopwords
CountVectorizer(stop_words = None)     # The default in set to none - if you don't specify any stopwords they will be kept

# Create lists of stopwords using NLTK

import nltk

nltk.download('stopwords')

from nltk import stopwords

# Includes stopwords from many languages for example

stopwords.words('english')
stopwords.words('german')

# Stemming - converts related words into their 'root word' or stem

# Example of a stemmer in Python is PorterStemmer from the nltk library

from nltk.stem import PorterStemmer

porter = PorterStemmer()

porter.stem("walking")      # pass in token you want to stem

>>> 'walk'

# Lemmatization is more sophisticated and uses actual rules of a language, returns the true root
# whereas, stemming returns words that are not necessarily a real word

# For example: Stemming will take the word 'better' and return 'better', whereas lemmatization will return the word 'good'

# Lemmatization is available in several Python libraries

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download("wordnet")        # Only need to be done once

lemmatizer = WordNetLemmatizer()

lemmatizer.lemmatize("mice")        # call function lemmatize
>>> 'mouse'

lemmatizer.lemmatize("going")
>>> 'going'

lemmatizer.lemmatize("going", pos = wordnet.VERB)       # Lemmatize take the argument pos - Part of speech
>>> 'go'

# Part of speech tagging help establishing the meaning of a word
# We should do part of speech tagging before we run the lemmatizer

# If using the wordnet lemmatizer these POS tags needs to be properly mapped using a function

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return word_net.NOUN

nltk.download('averaged_percenption_tagger')

sentence = "Donald Trump has a devoted following".split()

words_and_tags = nltk.pos_tag(sentence)

>>> [('Donald', 'NNP'), ('Trump', 'NNP'), ('has', 'VBZ'), ('a', 'DT'), 
    ('devoted', 'VBN'), ('following', 'NN')]

for word, tag in words_and_tags:        # Run the lemmatizer on each token
    lemma = lemmatizer.lemmatize(word, pos = get_wordnet_pos(tag))
    print(lemma, end = " ")

>>> Donald Trump have a devote following

# Improving the CountVectoizer using TF-IDF (Term frequencz inverse document frequency)

# The term frequency is the count of a word in a document, its what we get from the CountVectorizer
# The inverse document frquency 

