from __future__ import division
import argparse
import pandas as pd

# useful stuff
import numpy as np
from scipy.special import expit
from spacy.tokenizer import Tokenizer
import en_core_web_sm
import re



__authors__ = ['assael_jeremi', 'auriau_vincent', 'rouanet-labe_etienne']
__emails__  = ['jeremi.assael@supelec.fr', 'vincent.auriau@supelec.fr', 'etienne.rouanet-labe@student.ecp.fr']


def text2sentences(path):
    """ Function to tokenize the different sentences of a file"""
    sentences = []
    nlp = en_core_web_sm.load()
    tokenizer = Tokenizer(nlp.vocab, rules={}) #We are using the tokenizer of Spacy
    with open(path, encoding="utf-8") as f:
        for l in f:
            sent = tokenizer(l.lower())
            new_sent = [str(word) for word in sent]
            sentences.append(new_sent)
    return sentences

def loadPairs(path):
    """ Loading the pairs of word which will be used to test our model"""
    data = pd.read_csv(path,delimiter='\t')
    pairs = zip(data['word1'],data['word2'],data['similarity'])
    return pairs


class SkipGram:
    def __init__(self,sentences=None, nEmbed=100, negativeRate=5, winSize = 5, minCount = 5):
        self.sentences = sentences #Sentences on which the model is trained
        self.nEmbed = nEmbed #Size of a word embedding
        self.negativeRate = negativeRate #Number of negative samples for one word in the context
        self.winSize = winSize #The window size to get the context
        self.minCount = minCount #Minimum of time a word has to appear in the sentences to be considered
        self.words = None #Dictionnary of considered words that are in our sentences, with their counts as values
        self.vocab = None #List of the considered words in our sentences
        self.word2id = None #A dictionnary mapping each word to an idea, including the '<UNK>' word
        self.targets_emb = None #The matrix of targets embeddings, each column corresponding to a word, ranked by id
        self.contexts_emb = None #The matrix of contexts embeddings, each line corresponding to a word, ranked by id


    def processing_sentences(self):
        # we start by building the dictionnary words (attribute self.words) 
        #from the tokenized sentences
        words = {}
        for i in range(len(self.sentences)):
            for word in self.sentences[i]:
                words[word] = words.get(word,0) + 1
        
        # we remove from these words ponctuation, contractions (like 't),
        # every word which is not composed of alpha-numeric characters,
        # and words which appears less than minCount
        for word in list(words.keys()):
            if words[word] < self.minCount:
                words.pop(word, None)
            elif re.match("[^a-zA-Z0-9]", word):
                #also deleting contractions
                words.pop(word, None)
            elif re.match("\d", word):
                #removing numbers
                words.pop(word, None)        
                
        # we then get the total vocabulary list (attribute self.vocab) simply 
        # by taking the keys of the dictionary words        
        vocab = list(words.keys())
        
        # we build the word2id attribute by mapping each word of vocab to an id
        word2id = {k: v for (k,v) in zip(list(words.keys()), list(range(len(words.keys()))))}
        # we create a unknown token <UNK> which will be used to map unkown tokens of our corpus
        word2id['<UNK>'] = len(words.keys())
        
        # we then assign the built structures to the right attributes
        self.words = words
        self.vocab = vocab
        self.word2id = word2id
        
        
    
    def train(self, stepsize=0.1, epochs=7):
        """ 
        The training method, allowing to build word2vec from a set of sentences.
        -stepsize: learning rate of the SGD implemented in this train method
        -epochs: number of epochs during which training is done
        """
        
        # we start by building the self.words, self.vocab attributes 
        # using the processing_sentences method
        self.processing_sentences()
        vocab = self.vocab
        words = self.words
        
        # we iniialize embeddings using a matrix filled with random values
        # and we build the embeddings attributes
        targets_emb = np.random.uniform(-1, 1, size=(self.nEmbed, len(vocab) + 1))
        contexts_emb = np.random.uniform(-1, 1, size=(len(vocab) + 1, self.nEmbed))
        self.targets_emb = targets_emb
        self.contexts_emb = contexts_emb
        
        # parameters for negative sampling
        alpha = 0.75
        k = self.negativeRate
        L = self.winSize
        
        # using the alpha parameters and words attribute to build
        # a dictionary giving the probability of a word to be chosen as
        # a negative sample
        dict_interm = {k: v**alpha for k, v in words.items()}
        count_of_words_alpha = sum(list(dict_interm.values()))
        probability_choice = {}
        for word in list(words.keys()):
            probability_choice[word] = dict_interm[word]/count_of_words_alpha
       
        # as np.random.choice is very long, we are not using it in the loop
        # instead, we sample 10 000 000 words according to the probability distribution
        # and initialize an index at 0,
        # we are going to take words from this list as negative words,
        # when we are at the end of the list, we sample 10 000 000 new words
        negative_choices = np.random.choice(list(probability_choice.keys()), size=10000000, p=list(probability_choice.values()))
        index = 0 
        
        for epoch in range(epochs):
            # training for epochs epochs
            for i in range(len(self.sentences)):
                # we go through each sentence in our dataset
                for j in range(len(self.sentences[i])):
                    # we go through each word in our sentence
                    # the considered word is the target
                    target = self.sentences[i][j] 
                    
                    # we are building the context of size winSize using the surrounding words
                    context = []
                    for l in range(max(0, int(j-L/2) + 1), min(len(self.sentences[i]), int(j+L/2) + 2)):
                        if self.sentences[i][l] != target:
                            context.append(self.sentences[i][l])
                    
                    # we are building the negative context of size negativeRate x context size
                    # by selecting randomly words from the vocabulary according to the probability 
                    # distribution in the dictionary probability_choice thanks to the 
                    # negative_choices list, as explained above
                    negative = []
                    while len(negative) != k*len(context):
                        if index >=10000000:
                            # if we are at the end of the negative_choices list, we sample new words
                            index = 0
                            negative_choices = np.random.choice(list(probability_choice.keys()), size=10000000, p=list(probability_choice.values()))
                        choice = negative_choices[index]
                        if choice != target and choice not in context and choice not in negative:
                            # we make sure the selected words are not the target, not in the context and 
                            # not already selected as negative
                            negative.append(negative_choices[index])
                            index += 1
                        else:
                            index += 1
                    
                    # if the target word, a context word or a negative word is not in the vocabulary,
                    # it is map to a common vector represented by the word '<UNK>'
                    if target not in vocab:
                        target = '<UNK>'
                    for l in range(len(context)):
                        if context[l] not in vocab:
                            context[l] = '<UNK>'
                    for l in range(len(negative)):
                        if negative[l] not in vocab:
                            negative[l] = '<UNK>'
                    
                    
                    # we compute the gradient of the loss with respect to the different parameters
                    # (embeddings of target, context and negative samples), using the methods
                    # defined below
                    similarities_c, similarities_n, t_word_embedding, c_emb, n_emb = \
                                        self.similarities_and_embeddings(target, context, negative)
                    grad_target = self.loss_gradient_target(similarities_c, similarities_n, c_emb, n_emb)
                    grad_context = self.loss_gradient_context(similarities_c, t_word_embedding)
                    grad_negative = self.loss_gradient_negative(similarities_n, t_word_embedding)
                    
                    
                    # we update the target, contexts and negative samples embeddings using 
                    # a SGD with a learning rate of stepsize
                    # the word2id dictionnary allows us to easily localize the embedding 
                    # of a word in the matrixes
                    t_word_index = self.word2id[target]
                    self.targets_emb[:,t_word_index] += -stepsize*grad_target
                    for l in range(len(context)):
                        c_word_index = self.word2id[context[l]]
                        self.contexts_emb[c_word_index, :] += -stepsize*grad_context[l]
                    for l in range(len(negative)):
                        n_word_index = self.word2id[negative[l]]
                        self.contexts_emb[n_word_index, :] += -stepsize*grad_negative[l]
            
                    
    def similarities_and_embeddings(self, target, context_list, negative_list):
        """A function which take the target word, the context words, and the 
        negative words and return the simalirities between the target and every 
        context words (dot product), the similarities between the target word 
        and every negative words (dot product), as well as the embeddings of
        all of these words (to avoid redundant operations in other functions)"""
        
        #initialization
        similarities_c = []
        similarities_n = []
        
        #getting the target embedding
        t_word_index = self.word2id[target]
        t_word_embedding = self.targets_emb[:, t_word_index]
        
        #computing the similarities between the target word and the other words
        #getting the embeddings
        c_emb = []
        n_emb = []
        for word in context_list:
            c_word_index = self.word2id[word]
            c_word_embedding = self.contexts_emb[c_word_index, :]
            c_emb.append(c_word_embedding)
            similarities_c.append(np.dot(c_word_embedding, t_word_embedding))
        for word in negative_list:
            n_word_index = self.word2id[word]
            n_word_embedding = self.contexts_emb[n_word_index, :]
            n_emb.append(n_word_embedding)
            similarities_n.append(np.dot(n_word_embedding, t_word_embedding))
            
        return similarities_c, similarities_n, t_word_embedding, c_emb, n_emb
        
    
    def loss(self, similarities_c, similarities_n):
        """ A function computing the loss we are trying to minimize"""
        loss = 0
        for element in similarities_c:
            loss += np.log(expit(element))
        for element in similarities_n:
            loss += np.log(expit(-element))
        # we put a minus as we want to minimize it
        return -loss
    
        
    def loss_gradient_target(self, similarities_c, similarities_n, c_emb, n_emb):
        """A function computing the gradient of the loss with respect to each
        element of the target embedding"""
        gradient_target = np.zeros((self.nEmbed))
        for j in range(len(similarities_c)):
            gradient_target += -c_emb[j]*(1-expit(similarities_c[j]))
        for j in range(len(similarities_n)):
            gradient_target += n_emb[j]*(1-expit(-similarities_n[j]))
        return gradient_target
    
        
    def loss_gradient_context(self, similarities_c, t_word_embedding):
        """A function computing the gradient of the loss with respect to each
        element of each of the context embeddings"""
        gradient_context = np.zeros((len(similarities_c), self.nEmbed))
        for i in range(gradient_context.shape[0]):
            gradient_context[i] = -(1-expit(similarities_c[i]))*t_word_embedding
        return gradient_context
    
    
    def loss_gradient_negative(self, similarities_n, t_word_embedding):
        """A function computing the gradient of the loss with respect to each
        element of each of the negative embeddings"""
        gradient_negative = np.zeros((len(similarities_n), self.nEmbed))
        for i in range(gradient_negative.shape[0]):
            gradient_negative[i] = (1-expit(-similarities_n[i]))*t_word_embedding
        return gradient_negative
                     

    def similarity(self,word1,word2):
        """
            computes similiarity between the two words. unknown words are mapped to one common vector
        :param word1:
        :param word2:
        :return: a float in [0,1] indicating the similarity (the higher the more similar).
        To compute the final embedding of a word used for similarity calculations is the sum
        of the target and the context embedding.
        """
        
        # if the words are not in the vocabulary, they are mapped to the common
        # vector for unknown words : <UNK>
        if word1 not in self.vocab:
            word1 = '<UNK>'
        if word2 not in self.vocab:
            word2 = '<UNK>'
            
        # getting the id of the words
        w1_index = self.word2id[word1]
        w2_index = self.word2id[word2]
        
        # thanks to the id, getting the embeddings of the words
        # targets embeddings
        t1_emb = self.targets_emb[:, w1_index]
        t2_emb = self.targets_emb[:, w2_index]
        # contexts embeddings
        c1_emb = self.contexts_emb[w1_index, :]
        c2_emb = self.contexts_emb[w2_index, :]
        #final embeddings: we are summing them
        w1_emb = t1_emb+c1_emb
        w2_emb = t2_emb+c2_emb
        
        # as we want a number between 0 and 1, we compute the absolute value
        # of the cosine similarity (see readme file)
        cosine_similarity = np.absolute(np.dot(np.transpose(w1_emb), w2_emb)/(np.linalg.norm(w1_emb)*np.linalg.norm(w2_emb)))
        
        return cosine_similarity
        
        
    def save(self,path):
        """ This function save the built embeddings in a h5 file using two keys:
        "contexts" for context embeddings and "target" for target embeddings.
        Embeddings are saved as pandas dataframes.We are saving both separately 
        even though in out final usage we are going to sum them"""
        pd_targets = pd.DataFrame(self.targets_emb, columns=self.vocab+['<UNK>'])
        pd_contexts = pd.DataFrame(self.contexts_emb, index=self.vocab+['<UNK>'])
        pd_targets.to_hdf(path, 'targets')
        pd_contexts.to_hdf(path, 'contexts')   
        
        
    def most_similar(self, word, number_of_words):
        """ A function taking a word and returning the list of the
        number_of_words most similar words to the considered word as well
        as the score of similarity, computed thanks to our embeddings"""
        
        def takeSecond(elem):
            """Tool function to define the key to sort the list of similar words"""
            return elem[1]
    
        similars = []
        vocab = self.vocab
        for i in range(len(vocab)):
            similars.append((vocab[i], self.similarity(word, vocab[i])))
        similars = sorted(similars, reverse=True, key=takeSecond)
        return similars[0:number_of_words]
    
    
    def sentence_embeddings(self, tokenized_sentence):
        """ This function computes a sentence embedding by doing
        the weighted-average of the embedding of the words forming the sentence.
        The weight of a word is its "kind of IDF" value in the corpus of sentences""" 
        
        # we build the IDF dictionary
        idf = self.build_idf()
        
        # we look for needed word embeddings and the "kind of" IDF values of these words
        word_embeddings = []
        idf_values = []
        for word in tokenized_sentence:
            if word not in self.vocab:  #checking that the word is in our vocabulary
                word = '<UNK>'
            wi_index = self.word2id[word]
            ti_emb = self.targets_emb[:, wi_index]
            ci_emb = self.contexts_emb[wi_index, :]
            wi_emb = ti_emb + ci_emb
            word_embeddings.append(wi_emb)
            idf_values.append(idf.get(word, 0))
            
        # we are computing the sentence embedding by doing the described weighted-average
        sent_emb = 0
        for j in range(len(word_embeddings)):
            sent_emb += word_embeddings[j]*idf_values[j]
        sent_emb = sent_emb/len(word_embeddings)
        
        return sent_emb
            
    
    def build_idf(self):
        """ A function building a kind of IDF dictionary which associates
        each word to its IDF value (see report for details)"""
        
        idf = {}
        for sent in self.sentences:
            for w in sent:
                idf[w] = idf.get(w, 0) + 1
        for word in list(idf.keys()):
            idf[word] = max(1, np.log10(len(self.sentences) / (idf[word])))
        return idf #it is not exactly IDF but a proxy allowing to get good results
    
    
    def similarity_sentences(self, sentence1, sentence2):
        """ A function taking two sentences and returning their similarity as a 
        number between 0 and 1"""
        
        # getting the sentences embeddings
        s1_emb = self.sentence_embeddings(sentence1)
        s2_emb = self.sentence_embeddings(sentence2)
        
        # computing absolute value of cosine similarity
        cosine_similarity = np.absolute(np.dot(np.transpose(s1_emb), s2_emb)/(np.linalg.norm(s1_emb)*np.linalg.norm(s2_emb)))
        
        return cosine_similarity
        
    
    def most_similar_sentences(self, sentence, number_of_sentences):
        """These function takes a sentence and returns the number_of_sentences
        most similar ones using the set of sentences provided when instanciating the
        SkipGram class"""
        
        def takeSecond(elem):
            """Tool function to define the key to sort the list of similar sentences"""
            return elem[1]
        
        similars = []
        s_emb = self.sentence_embeddings(sentence)
        for i in range(len(self.sentences)):
            si_emb = self.sentence_embeddings(self.sentences[i])
            cosine_similarity = np.absolute(np.dot(np.transpose(s_emb), si_emb)/(np.linalg.norm(s_emb)*np.linalg.norm(si_emb)))
            similars.append((self.sentences[i], cosine_similarity))
        similars = sorted(similars, reverse=True, key=takeSecond)
        return similars[0:number_of_sentences]


    @staticmethod
    def load(path, sentences=None):
        """ A static method to load already built embeddings of words from  a h5 file
        with two keys : "targets" and "context" embeddings """
        pd_targets = pd.read_hdf(path, "targets")
        pd_contexts = pd.read_hdf(path, "contexts")
        sg = SkipGram(None)
        if sentences != None:
            sg.sentences = sentences
        sg.targets_emb = np.array(pd_targets.values)
        sg.contexts_emb = np.array(pd_contexts.values)
        vocab = list(pd_targets.columns)
        sg.word2id = {k: v for (k,v) in zip(vocab, list(range(len(vocab))))}
        vocab.remove('<UNK>')
        sg.vocab = vocab
        return sg
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='path containing training data', required=True)
    parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
    parser.add_argument('--test', help='enters test mode', action='store_true')

    opts = parser.parse_args()

    if not opts.test:
        sentences = text2sentences(opts.text)
        sg = SkipGram(sentences)
        sg.train(stepsize=0.1, epochs=7)
        sg.save(opts.model)

    else:
        pairs = loadPairs(opts.text)

        sg = SkipGram.load(opts.model)
        for a,b,_ in pairs:
            print(sg.similarity(a,b))



    
    
        


        
