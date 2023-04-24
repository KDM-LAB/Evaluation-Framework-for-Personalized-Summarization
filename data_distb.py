import re
import os
import math
import numpy as np
import pickle
import random
import numpy as np
from collections import OrderedDict
from matplotlib import pyplot as plt
import copy
from sentence_transformers import SentenceTransformer, util
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

sentence_transformer_model = 'sentence-transformers/all-distilroberta-v1'

def get_data_distb(data_processed, want_doc_emb = False, want_user_summ_distb = False):
    '''
        distribution of doc, actual headline, user and model generated headlines/summaries  
    '''
    avg_u_oov_nonBias, avg_u_oov_BiasWon, avg_m_oov_nonBias, avg_m_oov_BiasWon = 0,0,0,0
    # ratio of sum divided by ratio of doc
    data_distb = {}
    i = 0
    tot_num_of_docs = len(data_processed)
    pbar = tqdm(total=tot_num_of_docs)
    for docid, info in data_processed.items():
        tmp = copy.deepcopy(info)
        
        # vocab that contains ratio of word, init with default val 0
        vocab = OrderedDict.fromkeys(tmp['vocab'], 0)
        
        doc_dir_path = 'emb/'+ docid
#         if not os.path.exists(doc_dir_path):
#             os.makedirs(doc_dir_path)
#             os.makedirs(doc_dir_path+'/u_summ')
#             os.makedirs(doc_dir_path+'/m_summ')

        doc_file_path = doc_dir_path+'/'+ docid
        if want_doc_emb:
            if not os.path.isfile(doc_file_path+'.pkl'):
                doc_emb = get_embeddings_win1(tmp['doc_text'])
                store_data(doc_emb, doc_file_path)
            else:
                doc_emb = load_data(doc_file_path)
        
        
        tmp['doc_text'] = get_word_distb(tmp['doc_text'], vocab)
        tmp['doc_summ'] = get_word_distb(tmp['doc_summ'], vocab)
        
        
        if want_user_summ_distb:
            for uid, summ in tmp['u_dict'].items():
                tmp['u_dict'][uid] = get_word_distb(summ, vocab)
                
                oovs_lst = []
                for word in tmp['u_dict'][uid].keys():
                    if tmp['doc_text'][word]:
                        tmp['u_dict'][uid][word] =  tmp['u_dict'][uid][word] / tmp['doc_text'][word]
                    elif tmp['u_dict'][uid][word] and tmp['doc_text'][word] == 0:
                        oovs_lst.append(word)
                
                # if oov exsist 
                if len(oovs_lst): 
                    # get embeddings 
                    u_file_path = doc_dir_path+'/u_summ/'+ uid
                    if not os.path.isfile(u_file_path+'.pkl'):
                        oov_emb = get_embeddings_win1(oovs_lst)
                        store_data(oov_emb, u_file_path)
                    else:
                        oov_emb = load_data(u_file_path)
                    
                        
                    for key in oovs_lst:
                        doc_score =  oov_info_dct[key]['prob_score']
                        if doc_score:
                            tmp['u_dict'][uid][key] = doc_score
                            #print('found oov : ', word, 'in docid and uid : ', docid, uid, tmp['u_dict'][uid][word], tmp['doc_summ'][word])
                            avg_u_oov_nonBias += 1
                        else:
                            tmp['u_dict'][uid][key] = 0 #0.56789 # when bias won we set val 0
                            #print('000 found oov : ', word, 'in docid and uid : ', docid, uid, tmp['u_dict'][uid][word], tmp['doc_summ'][word])
                            avg_u_oov_BiasWon += 1
                        
        #print(' mu')
        for mid, summs in tmp['m_dict'].items():
            for uid, summ in summs.items():
                
                tmp['m_dict'][mid][uid] = get_word_distb(summ, vocab)
                oovs_lst2 = []
                #print('sum : ', summ, 'uid : ', uid)
                for key in tmp['m_dict'][mid][uid].keys():
                    if tmp['doc_text'][key]:
                        tmp['m_dict'][mid][uid][key] = tmp['m_dict'][mid][uid][key] / tmp['doc_text'][key]
                    elif tmp['m_dict'][mid][uid][key] and tmp['doc_text'][key]==0:
                        oovs_lst2.append(key)
                
                if len(oovs_lst2):
                    
                    # get embeddings 
                    m_file_path = doc_dir_path+'/m_summ/'+ uid
                    if not os.path.isfile(m_file_path+'.pkl'):
                        oov_emb2 = get_embeddings_win1(oovs_lst2)
                        store_data(oov_emb2, m_file_path)
                    else:
                        oov_emb2 = load_data(m_file_path)
                        
                    # get distb wrt doc and assign score to oov
                    oov_info_dct2 = get_oov_distb(oov_emb2, doc_emb)

                    for key in oovs_lst2:
                        doc_score =  oov_info_dct2[key]['prob_score']
                        if doc_score:
                            tmp['m_dict'][mid][uid][key] = doc_score
                            #print('M found oov : ', key, 'in docid and M_id : ', docid, uid, tmp['m_dict'][mid][uid][key], tmp['doc_summ'][key])
                            avg_m_oov_nonBias += 1
                        else:
                            tmp['m_dict'][mid][uid][key] = 0 #0.1111111
                            #print('000 M found oov : ', key, 'in docid and M_id : ', docid, uid, tmp['m_dict'][mid][uid][key], tmp['doc_summ'][key])
                            avg_u_oov_BiasWon += 1
                            
        data_distb[docid] = tmp
        i += 1
        if i % 15 == 0: # update progress bar after 15 docs 
            pbar.update(15)
            
        if i % 500 == 0:
            pass
    print(avg_u_oov_nonBias, avg_u_oov_BiasWon, avg_m_oov_nonBias, avg_m_oov_BiasWon)
    print(avg_u_oov_nonBias+ avg_u_oov_BiasWon, avg_m_oov_nonBias+ avg_m_oov_BiasWon)
    return data_distb

