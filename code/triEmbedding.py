# -*- coding: utf-8 -*-
# Copyright 2019 The Hong Kong Polytechnic University (Xuandi Fu)
#
from numpy import array
from numpy import argmax
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.keyedvectors import Vocab
 

word2tri = {}
triEm = OneHotEncoder(sparse=False)
data_prefix='../data/nlu_data/'
w2v_path = data_prefix+"snip_tri.txt"
w2v = KeyedVectors.load_word2vec_format(w2v_path, binary=False)
#vocab = Vocab()


def readfile(dataset_path):
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
    return vocab_class, vocab_query, vocab_all


def genOneHotDict(data):
    values = array(data)
    onehot_dict = triEm.fit_transform(np.reshape(values,(-1,1)))


def getOneHotEm(vocab):
    '''input: vocab like "restaurant"
     return: embedded vector
    '''
    tri_vocab = word2tri[vocab]
    print(tri_vocab)
    embed = triEm.transform(array(tri_vocab).reshape(-1,1))
    embed = np.sum(embed, axis = 0)
    return embed

def outputTofile(vocab_all):
    
    output_path=data_prefix+'SNIP_triEmbedding.txt'
    
    fo = open(output_path, "w")
    
    content = ""
    content += str(len(vocab_all))+ " " +str(7069) + "\n"
    
    for vocab in vocab_all:
        vector = getOneHotEm(vocab)
        content += vocab+" "+' '.join(str(v) for v in vector) + "\n"
  
    fo.write(content)
    fo.close()
    print ("done")

def genTriEmb():
  
    dataset_path=data_prefix+'dataSNIP.txt'
    output_path=data_prefix+'SNIP_triEmbedding.txt'
    
    vocab_class, vocab_query, vocab_all = readfile(dataset_path)
      
    vocab_hash = ["#"+str(vocab)+"#" for vocab in vocab_all]
    tri_vocab_all = []
    print(vocab_hash[0])
   
    
   # vocab_all--vocab_hash--word2tri[vocab_all]
   #word2tri: vocab_all 2 tri-letter 
    for index,vocab in enumerate(vocab_hash):
        word2tri[vocab_all[index]]=[]
        tmp =[vocab[i:i+3] for i in range(len(vocab)-2)]
        tri_vocab_all.extend(tmp)
        word2tri[vocab_all[index]]=tmp
    
    tri_vocab_all = np.unique(tri_vocab_all)
    genOneHotDict(tri_vocab_all)
    emb = getOneHotEm("!")
    print(emb.shape)
    print(np.sum(emb))
    
    k = np.where(emb == 1)
    print(k)
    #outputTofile(vocab_all)
     
genTriEmb()
'''
output_path=data_prefix+'SNIP_triEmbedding.txt'
w2v = KeyedVectors.load_word2vec_format(output_path, binary=False)
print(w2v.index2entity[1]) # array for word
print(w2v.vectors[1]) # array for embedding
print("77",w2v.vocab[w2v.index2entity[5]]) #number of times it occurred
embedding = w2v.syn0  
'''
