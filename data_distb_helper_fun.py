import re
import os
import math
import numpy as np
import random
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
import numpy as np
from collections import OrderedDict
from tqdm import tqdm

import pickle
def store_data(var, path):
    with open(path+'.pkl', 'wb') as file:
        pickle.dump(var, file)

def load_data(path):    
    with open(path+'.pkl', 'rb') as file:
        var = pickle.load(file)
    return var

def get_word_distb(words, vocab):
    
    vocab_dist = copy.deepcopy(vocab)
    n = len(words)
    for word in words:
        vocab_dist[word] += 1
        
    vocab_dist = {k : v/n for k, v in vocab_dist.items()}
    
    return vocab_dist

def cal_b(max_sim):
    return 1 - np.sqrt(max_sim)

def modified_softmax(lst):
    '''
        it contains, additional bias b fun, which is inversely proportional to max ele of array
    '''
    
    lst = np.array(lst)
    
    b = cal_b(max(lst))
    
    lst = np.append(lst, b)
    exp_lst = np.exp(lst)
    exp_lst_sum = exp_lst.sum()
    
    exp_lst = exp_lst / exp_lst_sum # TODO: you may try more efficient implementation of softmax
    
    return exp_lst

def get_embeddings_win1(oovs_lst, model_link = '', add_bias=0):
    '''
        get embeddigs of words and store in dict
    '''
    
    model = SentenceTransformer(sentence_transformer_model)
    n = len(oovs_lst)
    oovs_dct = {}
    
    oovs_dct[oovs_lst[0]] = {}
    oovs_dct[oovs_lst[0]]['emd'] = model.encode(oovs_lst[0]) # window for first word
    oovs_dct[oovs_lst[0]]['score'] = 0

    for i in range(n-1):
        oovs_dct[oovs_lst[i]] = {}
        oovs_dct[oovs_lst[i]]['emd'] = model.encode(oovs_lst[i]) # window size 3
        oovs_dct[oovs_lst[i]]['score'] = 0
        
    oovs_dct[oovs_lst[n-1]] = {}
    oovs_dct[oovs_lst[n-1]]['emd']  = model.encode(oovs_lst[n-1]) # window for last word
    oovs_dct[oovs_lst[n-1]]['score'] = 0

    if add_bias:
        oovs_dct['bias_'] = {'score' : 0 , 'emd' : 0} # add additional word for bias to store noise
    
    return oovs_dct

def get_oov_distb(oov_emb, doc_emb):
    '''
        get distb of oovs wrt doc
    '''
    
    # find sim between 2 words
    for oov, oov_info in oov_emb.items():
        for word, word_info in doc_emb.items():
            sim = util.cos_sim(torch.FloatTensor(oov_info['emd']), torch.FloatTensor(word_info['emd'])).item()
            word_info['score'] = sim
        
        # storing word and score in orderded dict to fetch val and restore modified softmax val back to respective words
        score_ord_dct = OrderedDict()
        for word in doc_emb:
            score_ord_dct[word] = doc_emb[word]['score']
            
        lst = list(score_ord_dct.values())
        
        # apply modified softmax, which contains bias term too as last ele
        lst = modified_softmax(lst)
        i=0
        
        # store normalized values back to ordered dict
        for w in score_ord_dct:
            score_ord_dct[w] = lst[i]
            i += 1
        
        # since bias wasnt exsist in dict, we need to add bias key 
        score_ord_dct['bias__'] = lst[len(lst)-1]
        
        mx, mx_idx, score_sum = 0, 0, 0
        
        
        for w, score in score_ord_dct.items():
            
            if mx < score:
                mx = score # max sim score
                mx_idx = i # idx that contains max sim score
                max_word_match = w # word having high sim score
            
            score_sum += score 
            i+=1
            
        if mx_idx == len(score_ord_dct) - 1: # if max idx is last, means bias won
            oov_info['prob_score'] = 0
        else:
            # final score will be sum of all score except bias: intuition - sum of contribution of each oov in doc to oov 
            score_sum = (score_sum - score_ord_dct['bias__'])
            oov_info['prob_score'] = score_sum # oov_info['score']

    return oov_emb # temp not used, so we added extra attributes to actual data itself