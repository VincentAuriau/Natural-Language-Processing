import math
import numpy as np
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse
import pickle


__authors__ = ['assael_jeremi','auriau_vincent', 'rouanet-labe_etienne']
__emails__  = ['jeremi.assael@supelec.fr', 'vincent.auriau@supelec.fr', 'etienne.rouanet-labe@student.ecp.fr']

def text2sentences2(path):
    sentences = []
    with open(path) as f:
        for l in f:
            sentences.append( l.lower() )
    return sentences


def sep_dial(train):

    indexes = []

    for i in range(len(train)) :

        if (train[i][0:14] == "1 your persona"):

            indexes.append(i)

    indexes.append(len(train))

    ll = []

    for i in range(len(indexes)-1):

        ll.append(train[indexes[i]:indexes[i+1]])

    return ll


def computeIDF(file):
    total_str = []

    word_dict = {}
    i = 0
    for situation in file:
        
        i += 1
        all_sentences = ''
        for sentence in situation:
            if sentence[2] not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
                all_sentences += sentence[2:] + ' '
            else:
                all_sentences += sentence[3:] + ' '
        list_to_remove = ['\n', '\t', ':', '|', '.', ',', "'", '?', '!', 'your persona', 'partners persona']
        for el in list_to_remove:
            all_sentences = all_sentences.replace(el, '')
        # print(len(word_count.keys()))
        
        situation_dict = {}
        for word in all_sentences.split(' '):
            if word not in situation_dict.keys():
                word_dict[word] = word_dict.setdefault(word, 0) + 1
                situation_dict[word] = 1
                
        total_str.append(all_sentences)
                
    for key in word_dict.keys():
        word_dict[key] = word_dict[key] / len(file)
        
    
    word_dict['unknown'] = 1 / len(file)
        
    return word_dict, total_str

def compute_TFIDF(dict_, nbr_docs, string):
    tfidf = 0
    
    words = string.split(' ')
    for word in words:
#        print(word, words, words.count(word))
        tf = words.count(word) / len(words)
        df = dict_.get(word, dict_['unknown'])
        
#        print(tf, df, math.log(nbr_docs / df))
        
        tfidf += tf * math.log(nbr_docs / df)
    
    return tfidf

parser = argparse.ArgumentParser(description='Running')

parser.add_argument('--mode', type=str, default='test', help='train or test')
parser.add_argument('--model', default='model', type=str, help='path to model')
parser.add_argument('--text', default='data/train_both_original.txt', type=str, help='path to text data')

def main():
    
    args = parser.parse_args()
    A = sep_dial(text2sentences2(args.text))
    # print(A[0])
    print(args.mode=='train')
    if args.mode == 'train':
        num_examples = 10
        subfile = A[:num_examples]
        
        IDF_dict, full_text = computeIDF(subfile)
        
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit(full_text)
        
        inputs = []
        outputs = []
        
        for situation in subfile:
            word_dict = {}
            all_sentences = ''
            for sentence in situation:
                if sentence[2] not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
                    all_sentences += sentence[2:] + ' '
                else:
                    all_sentences += sentence[3:] + ' '
            list_to_remove = ['\n', '\t', ':', '|', '.', ',', "'", '?', '!', 'your persona', 'partners persona']
            for el in list_to_remove:
                all_sentences = all_sentences.replace(el, '')
            # print(len(word_count.keys()))
        
            for word in all_sentences.split(' '):
                word_dict[word] = word_dict.setdefault(word, 0) + 1/len(all_sentences.split(' '))
            
            your_persona = []
            partners = []
            dialogs = []
            
            for i in range(len(situation)):
                situation[i] = situation[i].replace('\n', '')
            
            for info in situation:
                if 'your persona' in info:
                    your_persona.append(info)
                elif "partner's persona" in info:
                    partners.append(info)
                else:
                    dialogs.append(info)
                    
            for u in range(len(your_persona)):
                x = your_persona[u].find('persona')
                your_persona[u] =  your_persona[u][x+9:]
                
            for u in range(len(partners)):
                x = partners[u].find('persona')
                partners[u] =  partners[u][x+9:]
            
            input_4 = 0
            input_3 = 0
            dial = ''
            for dialog in dialogs:
                Z = dialog.split('\t')
        
        #        answers = Z[-1].split('|')
        #        true_answer = answers[0]
                
                if Z[-2] == '':
                    true_answer = Z[-3]
                    conversation = Z[:-3]
                else:
                    true_answer = Z[-2]
                    conversation = Z[:-2]
                fake_answers = Z[-1].split('|')
                
        #        input_1 = compute_TFIDF(IDF_dict, len(subfile), your_persona[0])
        #        input_2 = compute_TFIDF(IDF_dict, len(subfile), partners[0])
                
                input_1 = X.transform(your_persona)[0]
                input_2 = X.transform(partners)[0]
                
                for sentence in conversation:
                    
                    if sentence[1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                        sentence = sentence[2:]
                    elif sentence[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                        sentence = sentence[1:]
                    
                    dial += sentence + ' '
                dial += true_answer[0]
                input_3 = X.transform([dial])[0]
                # print('true', true_answer)
                input_4 = X.transform([true_answer])[0]
                
                inputs.append([(input_1.toarray() + input_2.toarray() + input_3.toarray() + input_4.toarray()).tolist()])
                
                outputs.append(1)
                
                for fake_answer in fake_answers:
                    input_5 = X.transform([fake_answer])[0]
                    inputs.append([(input_1.toarray() + input_2.toarray() + input_3.toarray() + input_5.toarray()).tolist()])
                
                    outputs.append(0)
    
    
        inputs = np.array(inputs).reshape(np.array(inputs).shape[0], np.array(inputs).shape[3]).tolist()
        clf = SVC(gamma='auto', class_weight='balanced', probability=True)
        clf.fit(np.array(inputs), np.array(outputs))
        
        dbfile = open(args.model, 'wb') 
        pickle.dump((clf, X), dbfile)                   
        dbfile.close()
    
    elif args.mode == 'test':
        total_ans = []
        num_test = len(A)
        test_subfile = A[:num_test]
        
        dbfile = open(args.model, 'rb')      
        (clf, X) = pickle.load(dbfile) 
        dbfile.close() 

        for situation in test_subfile:
            word_dict = {}
            all_sentences = ''
            for sentence in situation:
                if sentence[2] not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
                    all_sentences += sentence[2:] + ' '
                else:
                    all_sentences += sentence[3:] + ' '
            list_to_remove = ['\n', '\t', ':', '|', '.', ',', "'", '?', '!', 'your persona', 'partners persona']
            for el in list_to_remove:
                all_sentences = all_sentences.replace(el, '')
            # print(len(word_count.keys()))
        
            for word in all_sentences.split(' '):
                word_dict[word] = word_dict.setdefault(word, 0) + 1/len(all_sentences.split(' '))
            
            your_persona = []
            partners = []
            dialogs = []
            
            for i in range(len(situation)):
                situation[i] = situation[i].replace('\n', '')
            
            for info in situation:
                if 'your persona' in info:
                    your_persona.append(info)
                elif "partner's persona" in info:
                    partners.append(info)
                else:
                    dialogs.append(info)
                    
            for u in range(len(your_persona)):
                x = your_persona[u].find('persona')
                your_persona[u] =  your_persona[u][x+9:]
                
            for u in range(len(partners)):
                x = partners[u].find('persona')
                partners[u] =  partners[u][x+9:]
            
            input_4 = 0
            input_3 = 0
            dial = ''
            ins_tfidf = []
            ins_string = []
            
            for dialog in dialogs:
                Z = dialog.split('\t')
        
        #        answers = Z[-1].split('|')
        #        true_answer = answers[0]
                
                if Z[-2] == '':
                    true_answer = Z[-3]
                    conversation = Z[:-3]
                else:
                    true_answer = Z[-2]
                    conversation = Z[:-2]
                fake_answers = Z[-1].split('|')
                
                input_1 = X.transform(your_persona)[0]
                input_2 = X.transform(partners)[0]
                
                for sentence in conversation:
                    
                    if sentence[1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                        sentence = sentence[2:]
                    elif sentence[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                        sentence = sentence[1:]
                    
         
                    dial += sentence + ' '
                dial += true_answer[0]
                input_3 = X.transform([dial])[0]
                # print('true', true_answer)
                input_4 = X.transform([true_answer])[0]
                
                ins_tfidf.append([(input_1.toarray() + input_2.toarray() + input_3.toarray() + input_4.toarray()).tolist()])
                ins_string.append(true_answer[0])
                
                for fake_answer in fake_answers:
                    input_5 = X.transform([fake_answer])[0]
                    ins_tfidf.append([(input_1.toarray() + input_2.toarray() + input_3.toarray() + input_5.toarray()).tolist()])
                    ins_string.append(fake_answer)
                    
            ans = []
            for i in range(len(ins_tfidf)):
                ans.append(clf.predict_proba(np.array(ins_tfidf[i:i+1]).reshape(1, np.array(ins_tfidf[i:i+1]).shape[3]))[0][1])
                
            # print(ans)
            # print('sol', np.argmax(np.array(ans)))
            total_ans.append([np.argmax(np.array(ans)), ins_string[np.argmax(np.array(ans))]])
        return total_ans

if __name__ == "__main__":
    a = main()  
        