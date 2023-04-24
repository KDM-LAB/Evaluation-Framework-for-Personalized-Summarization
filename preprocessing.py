import re
import os
import nltk
import pandas as pd
import math
import numpy as np
import pickle
import random
import numpy as np
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import OrderedDict
from matplotlib import pyplot as plt
import copy
# from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
from collections import OrderedDict
from tqdm import tqdm

def store_data(var, path):
    with open(path+'.pkl', 'wb') as file:
        pickle.dump(var, file)

def load_data(path):    
    with open(path+'.pkl', 'rb') as file:
        var = pickle.load(file)
    return var

import json
with open('survey_data.json') as f:
    data = json.load(f)

def text_to_tokens(text):

    # set of stopwords
    stopwords = set(nltk.corpus.stopwords.words('english'))
    
    # wordnet lemmatizer
    lemmatizer = nltk.stem.WordNetLemmatizer()

    text = re.sub(r'[^\w\s]', '', text) # remove punctuation

    text = re.sub(r'[\d+]', '', text.lower()) # remove numerical values and convert to lower case

    tokens = nltk.word_tokenize(text) # tokenization

    tokens = [token for token in tokens if token not in stopwords] # removing stopwords

    tokens = [lemmatizer.lemmatize(token) for token in tokens] # lemmatization

    return tokens

def data_preprocessing(data):
    '''
        To apply preprocessing and get data in token form rather string of words 
        
        input := data : dict
        output := data_processed : dict -- text in token form after applying preprocessing 
    '''
    data_processed = {}
    
    # for each doc
    for docid, info in data.items():
        
        # Use deep copy to avoid overriding of values of data
        tmp = copy.deepcopy(info)
        tmp['vocab'] = set()

        tmp['doc_text'] = text_to_tokens(tmp['doc_text'])
        tmp['vocab'].update(tmp['doc_text'])
        tmp['doc_summ'] = text_to_tokens(tmp['doc_summ'])
        tmp['vocab'].update(tmp['doc_summ'])

        # for each summ written by particular user 
        for uid, summ in tmp['u_dict'].items():
            tmp['u_dict'][uid] = text_to_tokens(summ)
            tmp['vocab'].update(tmp['u_dict'][uid])
        
        # for each summ written by model wrt particular user 
        for mid, summs in tmp['m_dict'].items():
            for uid, summ in summs.items():
                tmp['m_dict'][mid][uid] = text_to_tokens(summ)
                tmp['vocab'].update(tmp['m_dict'][mid][uid])

        data_processed[docid] = tmp
        
    return data_processed

# apply preprocessing and get data in token form rather string of words 
data_processed = data_preprocessing(data)