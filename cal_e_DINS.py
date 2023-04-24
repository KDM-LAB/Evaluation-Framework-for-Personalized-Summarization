import numpy as np
import scipy as sp
from scipy.special import softmax
import pickle
from collections import OrderedDict
import pickle
def store_data(var, path):
    with open(path+'.pkl', 'wb') as file:
        pickle.dump(var, file)

def load_data(path):    
    with open(path+'.pkl', 'rb') as file:
        var = pickle.load(file)
    return var

def jsd(p, q, base=np.e):
    '''
        Implementation of pairwise `jsd` based on  
        https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
    '''
        
    ## convert to np.array
    p, q = np.asarray(p), np.asarray(q)

    ## normalize p, q to probabilities
    if p.sum():
        p = p/p.sum()
    else:
        pass

    if q.sum():
        q = q/q.sum()
    else:
        pass
        
    m = (1./2) * (p + q)
    
    val = sp.stats.entropy(p,m, base=base)/2. +  sp.stats.entropy(q, m, base=base)/2.
    
    return val if not np.isnan(val) else 0


def calculate_vanilla_D_INS(dataset, model_name = '2'):
    
    def dev(uid1, model_summ_dict, base_summ_dist):
        '''
            Deviation of summary of uid wrt all other summaries 
        '''
        dev_sum = 0
        w = OrderedDict() # so that it'll same order while fetching values and storing back from list 
        lst = []
        i=0

        # calculate weights 
        for uid2 in model_summ_dict:
            # to make sure each list contains score of same sequence of words
            p, q, r = [], [], []
            for word in model_summ_dict[uid2]:
                p.append(model_summ_dict[uid1][word])
                q.append(model_summ_dict[uid2][word])
                r.append(base_summ_dist[word])

            w[uid2] = jsd(p, q) / jsd(p, r) if jsd(p, r) else 0

            lst.append(w[uid2])

        # apply softmax and store back to dict
        lst = softmax(np.array(lst))
        for uid2 in model_summ_dict:
            w[uid2] = lst[i]
            i+=1
            
        dct = {}
        for uid2 in model_summ_dict:
            if uid1 != uid2:
                p, q = [], []
                for word in model_summ_dict[uid2]:
                    p.append(model_summ_dict[uid1][word])
                    q.append(model_summ_dict[uid2][word])
                
                val = w[uid2] * jsd(p, q) if w[uid2] else 0
                dct[uid2] = val
                dev_sum += val

        return dev_sum, dct

    vanilla_sum = 0
    for doc_id in dataset:
        dev_sum = 0
        
        # get all summaries of 1st model
        model_summ_dict = dataset[doc_id]['m_dict'][model_name] #dictionary of { user_id: model_summary_distribution}
    
        for uid in model_summ_dict:
            val, dct = dev(uid, model_summ_dict, dataset[doc_id]['doc_text'])
            dev_sum += val
        dev_sum /= len(model_summ_dict)

        vanilla_sum += dev_sum

    vanilla_sum /= len(dataset)

    vanilla_D_INS = 1 - vanilla_sum
    
    return vanilla_D_INS


def calculate_penalty_factor(dataset, model_name = '2'):
    user_distb = load_data('/home/sourishd/rahul/data_with_distb_v3_using_doc_text_v2/data_with_distb_v3_roberta')
    
    def dev(uid1, model_summ_dict, base_summ_dist):
        '''
            Deviation of summary of uid wrt all other summaries 
        '''
        dev_sum = 0
        w = OrderedDict() # so that it'll same order while fetching values and storing back from list 
        lst = []
        i=0

        # calculate weights 
        for uid2 in model_summ_dict:
            # to make sure each list contains score of same sequence of words
            p, q, r = [], [], []
            for word in model_summ_dict[uid2]:
                p.append(model_summ_dict[uid1][word])
                q.append(model_summ_dict[uid2][word])
                r.append(base_summ_dist[word])

            w[uid2] = jsd(p, q) / jsd(p, r) if jsd(p, r) else 0

            lst.append(w[uid2])

        # apply softmax and store back to dict
        lst = softmax(np.array(lst))
        for uid2 in model_summ_dict:
            w[uid2] = lst[i]
            i+=1
        dct = {}
        for uid2 in model_summ_dict:
            if uid1 != uid2:
                p, q = [], []
                for word in model_summ_dict[uid2]:
                    p.append(model_summ_dict[uid1][word])
                    q.append(model_summ_dict[uid2][word])
                val = w[uid2] * jsd(p, q) if w[uid2] else 0
                dct[uid2] = val
                dev_sum += val

        return dev_sum, dct
    
    ratio_wrt_doc = 0
    for doc_id in dataset:
        ratio_wrt_summ = 0
        # get all summaries of 1st model
        user_summ_dict = user_distb[doc_id]['u_dict']
        model_summ_dict = dataset[doc_id]['m_dict'][model_name] #dictionary of { user_id: model_summary_distribution}
        
        for uid in model_summ_dict:
            val, u_dct = dev(uid, user_summ_dict, dataset[doc_id]['doc_text'])
            val, m_dct = dev(uid, model_summ_dict, dataset[doc_id]['doc_text'])

            ratio_wrt_uid2 = 0
            for uid2 in u_dct:
                u = u_dct[uid2]
                m = m_dct[uid2]

                if max(u, m):
                    ratio_wrt_uid2 += (min(u, m) / max(u, m))
                else:
                    ratio_wrt_uid2 += 0
            
            if len(u_dct):
                ratio_wrt_uid2 /= len(u_dct)
                
            ratio_wrt_summ += ratio_wrt_uid2

        ratio_wrt_summ /= len(model_summ_dict)
        ratio_wrt_doc += ratio_wrt_summ
        
    ratio_wrt_doc /= len(dataset)
    
    return ratio_wrt_doc
