# -*- coding: utf-8 -*-
#%% env
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
#%%
# =============================================================================
# data_prefix='../data/SMP18/'
# dataset_path=data_prefix+'dataSMP18.txt'
# w2v_path=data_prefix+'sgns_merge_word.txt'
# output_path=data_prefix+'sgns_merge_subsetSMP.txt'
# =============================================================================

data_prefix='../data/ATIS/'
dataset_path=data_prefix+'dataATIS.txt'
w2v_path=data_prefix+'glove_6B_300d.txt'
output_path=data_prefix+'glove300_ATIS.txt'

#%% load w2v
w2v = KeyedVectors.load_word2vec_format(w2v_path, binary=False)

#%%
glove_file = w2v_path
tmp_file = get_tmpfile("test_word2vec.txt")
_ = glove2word2vec(glove_file, tmp_file)
w2v = KeyedVectors.load_word2vec_format(tmp_file)

#%% read in data and create vocab
vocab_class=[]
vocab_query=[]
for line in open(dataset_path,'rb'):
    arr =str(line.strip(),'utf-8')
    arr = arr.split('\t')
    label = [w for w in arr[0].split(' ')]
    question = [w for w in arr[1].split(' ')]
    vocab_class.extend(label)
    vocab_query.extend(question)


vocab_class=np.unique(vocab_class) # zhihu 531/3494
vocab_query=np.unique(vocab_query)
vocab_all=np.unique(np.concatenate((vocab_class,vocab_query)))
print('vocab class:',len(vocab_class))
print('vocab query:',len(vocab_query))
print('vocab all:',len(vocab_all))

#%% check w2v capacity
counter_in_class=0
counter_notin_class=0
indx=np.array([],dtype='int')
w_notin_dict_class=[]
for w in vocab_class:
    if w in w2v:
        counter_in_class+=1
        indx=np.append(indx,w2v.vocab[w].index)
    else: 
        counter_notin_class+=1
        w_notin_dict_class.append(w)
print('for class: in & not in',counter_in_class,counter_notin_class)

counter_in_query=0
counter_notin_query=0
indx=np.array([],dtype='int')
w_notin_dict_query=[]
for w in vocab_query:
    if w in w2v:
        counter_in_query+=1
        indx=np.append(indx,w2v.vocab[w].index)
    else: 
        counter_notin_query+=1
        w_notin_dict_query.append(w)
print('for query: in & not in',counter_in_query,counter_notin_query)
print('check w_notin_dict_class and w_notin_dict_query for details') # SMP: 432/3051

#%% generate w2v subset
def restrict_w2v(w2v, restricted_word_set):
    new_vectors = []
    new_vocab = {}
    new_index2entity = []

    for i in range(len(w2v.vocab)):
        word = w2v.index2entity[i]
        vec = w2v.vectors[i]
        vocab = w2v.vocab[word]
        if word in restricted_word_set:
            vocab.index = len(new_index2entity)
            new_index2entity.append(word)
            new_vocab[word] = vocab
            new_vectors.append(vec)

    w2v.vocab = new_vocab
    w2v.vectors = np.array(new_vectors)
    w2v.index2entity = np.array(new_index2entity)
    w2v.index2word = np.array(new_index2entity)
    
    return w2v.vocab,w2v.vectors

#%%
word_class, vec_class=restrict_w2v(w2v, vocab_all)
#%%
w2v.save_word2vec_format(output_path)
