import nltk.classify.util
from nltk.classify import NaiveBayesClassifier 
from nltk.corpus import movie_reviews

# What do the moview reviews look like?
# The following command prints the last 100 characters of 
# each file in movie_reviews followed by ...
  
for fileid in movie_reviews.fileids():
  print(fileid, movie_reviews.raw(fileid)[:100], '...')
  
# You can see from the files that there is a folder with 
# negative reviews and a folder with positive reviews.
# If you know a movie review fileid you can also look at 
# the entire review.
# Use the raw function to see the entire review 
print(movie_reviews.raw('neg/cv000_29416.txt'))

# Note there are two categories 
movie_reviews.categories()
# Divide the reviews into negative and positive reviews 
negids = movie_reviews.fileids('neg')
posids = movie_reviews.fileids('pos')

# A function: given a set of words, create a python 
# dictionary (associative array) where each word is 
# a key, and each key gets the value 'True'.
def word_feats(words):
  return dict([(word, True) for word in words])

# Extract the words as features that are classified 
# as negative or positive.
# Store them in a python dictionary with the other 
# dictionary (associative array) with 'neg' and 'pos' 
# as the values and the {word: True} dictionary as 
# the keys.

negfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'neg') 
    for f in negids]
posfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'pos') 
    for f in posids]

# You can inspect the dictionaries, but it's very 
# slow to print (this may take some time!):
print(negfeats)

# Try this to see the end of the file.
print(negfeats[:1])

# Because these are dictionaries, you can check how 
# many entries there are with len():
len(negfeats)

# Let's use 3/4ths for training and 1/4th for testing
negcutoff = len(negfeats)*3/4
poscutoff = len(posfeats)*3/4


# Put both the positive and negative examples together 
# for the training and testing data:
trainfeats = negfeats[:int(negcutoff)] + posfeats[:int(poscutoff)]
testfeats = negfeats[int(negcutoff):] + posfeats[int(poscutoff):]

print('train on %d instances, test on %d instances' %
     (len(trainfeats), len(testfeats)))

classifier = NaiveBayesClassifier.train(trainfeats) 
print('accuracy:', nltk.classify.util.accuracy(classifier, testfeats)) 
classifier.show_most_informative_features()

# Place your movie review in place of 'This is a great movie'
my_review_text = 'This is a great movie.' 
my_review = my_review_text.split()

my_review_feats = dict([(word, True) for word in my_review]) 
print("Your review is " + classifier.classify(my_review_feats))

