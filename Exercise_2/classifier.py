import pandas as pd
from spacy.tokenizer import Tokenizer
import en_core_web_sm
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.stem.snowball import SnowballStemmer
import numpy as np



class Classifier:
    """The Classifier"""
    
    def __init__(self):
        """initialization of the classifier class which have several attributes"""
        self.clf = None #classifier is None at first
        self.voc = None #voc is None at first
        
        # The three paths for the used lexicons
        neglex = "../resources/negative-words.txt" 
        poslex = "../resources/positive-words.txt"
        other_lexicon = "../resources/lexicon.txt"
        
        # Extracting data contains into the lexicons
        
        # With n negative words and p positive words in this lexicon
        self.n, self.p = extract_lexicon(neglex, poslex)
        
        # Dictionary of words associated with a sentiment mark
        # ( Mark between -4 and 4, the lower, the more negative the associated
        # word is)
        self.dico_marks = extract_other_lexicon(other_lexicon)
        
        # The window size use to extract the interest zone of the sentence
        self.window_size = 6


    def train(self, trainfile):
        """Trains the classifier model on the training set stored in file trainfile"""
        train = pd.read_csv(trainfile, sep="\t", header=None)
        
        # Cleaning of sentences, extraction of interest zone
        df_train = build_dataset(train, self.window_size)
        
        # Building features based on lexicons
        new_feature_train = build_lexicon_feature1(df_train, self.n, self.p)
        new_feature_train2 = build_lexicon_feature2(df_train, self.dico_marks)
                                                    
        # Building the last features (BoW) and the vocabulary by encoding sentences with TFIDF                                             )
        x_train, y_train, voc = build_tfidf_train(df_train)
        
        # Concatenate all features
        x_train = np.concatenate([x_train, new_feature_train, new_feature_train2], axis=1)
        
        # Building the classifier using a logistic regression
        clf = train_LR(x_train,y_train)
        
        # We store the classifier and the training vocabulary in the attributes the class
        self.clf = clf
        self.voc = voc


    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
        dataset = pd.read_csv(datafile, sep="\t", header=None)
        
        # Cleaning of sentences, extraction of interest zone
        df = build_dataset(dataset, self.window_size)
        
        # Building features based on lexicons
        new_feature = build_lexicon_feature1(df, self.n, self.p)
        new_feature2 = build_lexicon_feature2(df, self.dico_marks)
        
        # Building the last features (BoW) by encoding sentences with TFIDF                                 
        # We encode the test/prediction set using the training vocabulary
        x = build_tfidf_test(df, self.voc)
        
        # Concatenate all features
        x = np.concatenate([x, new_feature, new_feature2], axis=1)
        
        # Predicting label using previously trained classifier
        y_pred = self.clf.predict(x)
        
        # Getting the string labels
        labels = []
        for i in range(len(y_pred)):
            if y_pred[i] == 1:
                labels.append('positive')
            elif y_pred[i] == 0:
                labels.append('neutral')
            else:
                labels.append('negative')
        return labels


        

def list_into_str(sentence):
    """ A tool function to transform a list of word strings 
    into a string sentence """
    sent = str(sentence[0])
    for i in range(1, len(sentence)):
        sent += " " + str(sentence[i])
    return sent



def build_dataset(dataset, window_size):
    """ Building a first dataset by cleaning the sentences and extracting the part 
    of interest of each sentence, which is of size window size on the left and on the
    right of the word on which the sentiment is expressed in the sentence """
    
    # Tokenization and cleaning of sentences and words on which sentiment is expressed
    # Removal of punctuation, getting rid of accents, of useless spaces, 
    # of non-alphanumeric characters. All numbers being considered as not bringing information,
    # they are all transform to 0
    nlp = en_core_web_sm.load()
    tokenizer = Tokenizer(nlp.vocab, rules={})
    sentences = list(dataset[4])
    words = list(dataset[2])
    
    words_of_interest = []
    for word in words:
        word1 = word.replace(u'\u2026',' ')
        word1 = word1.replace(u'\u00a0',' ')
        word1 = re.sub("[^a-zA-Z0-9 ]", ' ', word1)
        word1 = re.sub("[0-9_]", '0', word1)
        word1 = word1.split()
        word1 = list_into_str(word1)
        word_l = tokenizer(word1.lower())
        words_of_interest.append(word_l)
    
    tokenized_sentences = []
    for sent in sentences:
        sent1 = sent.replace(u'\u2026',' ')
        sent1 = sent1.replace(u'\u00a0',' ')
        sent1 = sent1.replace('...',' ')
        sent1 = sent1.replace('/',' ')
        sent1 = re.sub("[^a-zA-Z0-9 ]", ' ', sent1)
        sent1 = re.sub("[0-9_]", '0', sent1)
        sent1 = sent1.split()
        sent1 = list_into_str(sent1)
        sent_l = tokenizer(sent1.lower())
        tokenized_sentences.append(sent_l)    
    
    # Extract the part of sentence which is interesting by extracting window_size 
    # words surrounding the word of interest
    new_sentences = []
    for i in range(len(words_of_interest)):
        this_word = False
        for j in range(len(tokenized_sentences[i])):
            if len(words_of_interest[i]) == 1:
                if str(tokenized_sentences[i][j]) == str(words_of_interest[i]) and not this_word:
                    new_sent = tokenized_sentences[i][max(0, j-window_size):min(len(tokenized_sentences[i]), j+window_size+1)]
                    new_sentences.append(list_into_str(new_sent))
                    this_word = True
            else:
                if str(tokenized_sentences[i][j]) == str(words_of_interest[i][0]) and not this_word:
                    new_sent = tokenized_sentences[i][max(0, j-window_size):min(len(tokenized_sentences[i]), j+len(words_of_interest[i])+window_size+1)]
                    new_sentences.append(list_into_str(new_sent))
                    this_word = True


    # Returning the build dataset
    df = pd.concat([dataset[0], pd.DataFrame(new_sentences)], axis=1)
    df.columns = ["label","sentence"]
    return df



def label_one_hot_encoding(label_list):
    """ One_hot encoding of the labels so that they can be fed to the classifier """
    y = np.zeros(len(label_list))
    for i in range(len(label_list)):
        if label_list[i] == "positive":
            y[i] = 1
        elif label_list[i] == "neutral":
            y[i] = 0
        elif label_list[i] == "negative":
            y[i] = -1
    return y



def build_tfidf_train(dataset):
    """ Stem the preprocessed sentences using SnowballStemmer.
    Encode the preprocessed sentences using the TFIDF represnentation.
    Extracting the vocabulary of the training set 
    Return the training set based on TFIDF and the training vocabulary """
    
    # Stemming
    list_to_stem = list(dataset["sentence"])
    sent_stem = []
    stemmer = SnowballStemmer("english")
    for sent in list_to_stem:
        sent1 = nltk.word_tokenize(sent)
        pos = nltk.pos_tag(sent1)
        sent2 = ""
        for couple in pos:
            sent2 += " " + stemmer.stem(couple[0])
        sent_stem.append(sent2)
        
    # TFIDF representation, removing stopwords, keeping only unigrams
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 1))
    x = vectorizer.fit_transform(sent_stem).todense()
    
    # Getting vocabulary
    voc = list(vectorizer.vocabulary_.keys())
    
    # Getting one-hot encoded labels
    y = label_one_hot_encoding(list(dataset["label"]))
    
    # Return the training set and the associated labels
    return x,y,voc



def build_tfidf_test(dataset, voc):
    """ Stem the preprocessed sentences using SnowballStemmer.
    Encode the preprocessed sentences using the TFIDF represnentation and the 
    training vocabulary voc.
    Return the training set based on TFIDF (only x)"""
    
    # Stemming
    list_to_stem = list(dataset["sentence"])
    sent_stem = []
    stemmer = SnowballStemmer("english")
    for sent in list_to_stem:
        sent1 = nltk.word_tokenize(sent)
        pos = nltk.pos_tag(sent1)
        sent2 = ""
        for couple in pos:
            sent2 += " " + stemmer.stem(couple[0])
        sent_stem.append(sent2)
    
    # TFIDF representation, removing stopwords, keeping only unigrams, using training vocabulary
    vectorizer = TfidfVectorizer(vocabulary=voc, stop_words="english", ngram_range=(1, 1))
    x = vectorizer.fit_transform(sent_stem).todense()

    return x
    


def extract_lexicon(negpath, pospath):
    """ Extract data from the first used lexicon: a list of negative words and a
    list of negative words """
    
    # Negative words
    with open(negpath) as file:
        neg_words = list(file)[31:]
    neg = []
    for word in neg_words:
        neg.append(word[:-1])
        
    # Positive words
    with open(pospath) as file:
        pos_words = list(file)[31:]
    pos = []
    for word in pos_words:
        pos.append(word[:-1])
    return neg, pos



def build_lexicon_feature1(dataset, n, p):
    """ Building 9 features based on the first used lexicon, 
    3 features for each category of words.
    A word is considered positive if it is in p, negative if it is in n and 
    neutral otherwise. 
    The three built features for each categories are: a binary feature stating 
    if there are positive/negative/neutral words in the sentence; the number of 
    words of a categary in the sentence; the frequency of a category in the 
    sentence """
    
    new_feature = np.zeros((len(dataset), 9))
    list_sent = list(dataset["sentence"])
    sentences = []
    for sent in list_sent:
        sent1 = sent.split()
        sentences.append(sent1)
    for i in range(len(sentences)):
        for j in range(len(sentences[i])):
            if str(sentences[i][j]) in n: #negative
                new_feature[i,0] = 1 #binary feature
                new_feature[i,1] += 1 #count of words
                new_feature[i,2] += 1 # frequency
            elif str(sentences[i][j]) in p: #positive
                new_feature[i,3] = 1 #binary feature
                new_feature[i,4] += 1 #count of words
                new_feature[i,5] += 1 # frequency
            else: #neutral
                new_feature[i,6] = 1 #binary feature
                new_feature[i,7] += 1 #count of words
                new_feature[i,8] += 1 # frequency
                
    for i in range(len(sentences)):
        # effectively computing the frequency by dividing the count of words for
        # each category by the total number of words in the sentence
        new_feature[i,2] = new_feature[i,4]/len(sentences[i])
        new_feature[i,5] = new_feature[i,4]/len(sentences[i])
        new_feature[i,8] = new_feature[i,4]/len(sentences[i])
    return new_feature


        
def extract_other_lexicon(path):
    """ Extract data from the second used lexicon: a dictionary of words mapped
    to the associated marks. Marks are between -4 and 4, 
    the lower, the more negative a word is """
    words = []
    marks = []
    with open(path) as f:
        index = 0
        for line in f:
            # Do not take expression which are problematic
            if index not in [5, 1258, 2863, 5905]:
                l = line.split()
                words.append(l[0])
                marks.append(l[1])
            index+=1
    return dict(zip(words,marks))



def build_lexicon_feature2(dataset, dico_marks):
    """ Building two features based on the second used lexicon, simply
    by extracting the sentiment mark of each word in our sentences, if it exists.
    The first feature is the sum of marks for a sentence.
    The second feature is the average of marks for a sentence."""
    
    new_feature = np.zeros((len(dataset),2))
    list_sent = list(dataset["sentence"])
    sentences = []
    for sent in list_sent:
        sent1 = sent.split()
        sentences.append(sent1)
    for i in range(len(sentences)):
        for j in range(len(sentences[i])):
                # We get the mark associated witha word if this word is in our lexicon
                new_feature[i,0] += float(dico_marks.get(str(sentences[i][j]), 0))  #Sum of marks
                new_feature[i,1] += float(dico_marks.get(str(sentences[i][j]), 0))  #Average of marks
    for i in range(len(sentences)):
        # Effectively computing the average of marks
        new_feature[i,1] = new_feature[i,1]/len(sentences[i])
    return new_feature
                

  
def train_LR(x_train,y_train):
    """ Train the classifier, a logistic regression """
    # Parameters have been tuned on the dev set
    clf = LogisticRegression(solver='lbfgs', multi_class='ovr', C=0.025)
    clf.fit(x_train,y_train)
    
    # Return the train classifier
    return clf




    

