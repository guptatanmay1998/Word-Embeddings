# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 01:23:47 2020

@author: Tanmay Gupta
"""

"""
Created on Mon Jun 15 20:02:30 2020

@author: Tanmay Gupta
"""
#Importing Required Packages
import math
import os
import numpy as np
import tensorflow as tf
import pandas as pd
import re
#import nltk
import sys
from collections import OrderedDict
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
%matplotlib inline



#Reading data from excel sheet
cols=[2,21]#using only columns of name, chain_name
dataset=pd.read_excel('Book3.xlsx',index_col=None,na_values=['NA'],usecols=cols)
dataset=dataset.fillna('')
#Making list1 and list2 with the dataset
list1=dataset['name'].values
list2=dataset['chain_name'].values

print(list2[0])

#using set to remove duplicate merchant names and then appending these names into list named result
seen=set()
result=[]

for item in list1:
    if item not in seen:
        seen.add(item)
        result.append(item)

   

print(result[0])

#appending each of the words into the list named words
words = []
for text in result:
    for word in text.split(' '):
        words.append(word)
#restricting those words whose length is less than or equal to 1 and turning the word to lowercase
words = [ word.lower() for word in words if len(word)>1]

vocab = set(words)#using set to remove duplicate entries from words
print(len(vocab))

#char_to_int dictionary maps each word to a unique index
char_to_int={}
for i,word in enumerate(vocab):
    char_to_int[word] = i

int_to_char = dict((i,c) for i,c in enumerate(vocab))


#Appending the merchant name in splitted manner into list named sentences 
sentences = []
for sentence in result:
    sentences.append(sentence.split())
print(len(sentences))



#dictionary temp_dict maps each of the words(inputs) to another words(targets) within context window
temp_dict = []
window_size = 1
for sentence in sentences:
    for idx, word in enumerate(sentence):
        for neighbor in sentence[max(idx - window_size, 0) : min(idx + window_size, len(sentence)) + 1] :
            if neighbor != word:
                if len(word)>1 and len(neighbor)>1:
                    temp_dict.append([word.lower(), neighbor.lower()])
                   
print(temp_dict[0][0])              
#df = pd.DataFrame(temp_dict, columns = ['input', 'label'])


ONE_HOT_DIM=len(vocab)
B=np.zeros(ONE_HOT_DIM)
print(B)
#function to make one_hot_encoded vectors for words
def to_one_hot_encoding(data_point_index):
    one_hot_encoding = np.zeros(ONE_HOT_DIM)
    one_hot_encoding[data_point_index] = 1
    return one_hot_encoding




#only import tensorflow was giving error while executing
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()  


   
embedding_size = 256#size of the word embeddings
batch_size = 640#size of each batch used during batchwise training
n_batches = int((len(temp_dict))/batch_size)#number of batches=training data size/batch_size

#Details of the architecture
learning_rate= 0.001
x = tf.placeholder(tf.float32,shape = (None,len(vocab)))
y = tf.placeholder(tf.float32,shape = (None,len(vocab)))
w1 = tf.Variable(tf.random_normal([len(vocab),embedding_size]),dtype = tf.float32)
b1 = tf.Variable(tf.random_normal([embedding_size]),dtype = tf.float32)
w2 = tf.Variable(tf.random_normal([embedding_size,len(vocab)]),dtype = tf.float32)
b2 = tf.Variable(tf.random_normal([len(vocab)]),dtype = tf.float32)
hidden_y = tf.matmul(x,w1) + b1
_y = tf.matmul(hidden_y,w2) + b2
cost = tf.reduce_mean(tf.losses.mean_squared_error(_y,y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
print(y.shape)
print(_y.shape)
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
init = tf.global_variables_initializer()
init_l = tf.local_variables_initializer()
saver = tf.train.Saver()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.33)
sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
sess.run(init)
#saver.save(sess,'checkpoint_file')


#the cost converged after some 15 epochs
for epoch in range(25):
    avg_cost = 0
    b=0
    for i in range(n_batches-1):
        #a=0
        X=[]
        Y=[]
        for j in range(batch_size):
            if b==len(temp_dict):
                break
            X.append(to_one_hot_encoding(char_to_int[ temp_dict[b][0] ]))
            Y.append(to_one_hot_encoding(char_to_int[ temp_dict[b][1] ]))
            b+=1
        #print(X[0])
        batch_x=X#made input batch for training
        batch_y=Y#made target batch for training
        #b+=1
        #print(batch_x.shape)
        _,c = sess.run([optimizer,cost],feed_dict = {x:batch_x,y:batch_y})
        #print(test.shape)
       
       
        avg_cost += c/n_batches
    print('Epoch',epoch,' - ',avg_cost)
    if b==len(temp_dict):
        break
saver.save(sess,'my_model3')#saving the learned weights
#save_path = saver.save(sess,checkpoint_file)

#saver = tf.train.Saver()
import os

#tf.reset_default_graph()

#with tf.Session() as sess:
#     saver = tf.train.import_meta_graph('./checkpoints/checkpoint.chk.meta')
#     saver.restore(sess,tf.train.latest_checkpoint('./'))
     
# RUN THE SESSION
#saver = tf.train.Saver()

#with tf.Session() as session:
#     saver.restore(session, checkpoint_file)
#     saver.save(session, checkpoint_file)
     

     
#getting the learned embeddings    
vectors = sess.run(w1 + b1)   
print(vectors[0])

#embeddings dictionary maps each word to its embedding
embeddings=dict()
for i in vocab:
    embeddings[i]=vectors[char_to_int[i]]

   

import math
j=len(sentences)#length of list named sentences
words_frequency_global={}#getting words frequency across all the documents(merchant names)
document_vector_dict={}#maps a merchant name to its embeddings
k=0#using indexing for the list named result
#df1 = pd.DataFrame(data, columns = ['x1', 'x2'])
for sentence in sentences:
    if result[k] not in document_vector_dict:
        Z=np.zeros(embedding_size)
    #document_vector=[0,0]
        words_frequency={}#word frequncy within a merchant name
        TF_IDF={}
        n=len(sentence)#n=number of words present in a merchant name
        for idx, word in enumerate(sentence):
             if word not in words_frequency:
                 words_frequency[word]=1
             else:
                 words_frequency[word]+=1
             words_frequency_global[word]=words_frequency[word]
   
        for idx, word in enumerate(sentence):
            TF_IDF[word]=math.log(10,j/words_frequency_global[word])*(words_frequency[word]/n)#using TF_IDF values
       
        for idx, word in enumerate(sentence):
            if len(word)>1:
             #list3=w2v_df.loc[char_to_int[word.lower()],"x1":"x2"].to_list()
                Z+=embeddings[word.lower()]*TF_IDF[word]
        Z=Z/n
        document_vector_dict[result[k]] =Z
    
    k+=1
    
#mapping each of the unique merchant name to its corresponding chain_name entry
merchant_chain_name=dict()
seen1=set()
t=0
for item in list1:
    if item not in seen1:
        seen1.add(item)
        merchant_chain_name[item]=list2[t]
    t+=1
    
    
    
#print(document_vector_dict)
#function to get cosine_similarity_score of two embeddings
def get_cosine_similarity(feature_vec_1, feature_vec_2):    
    return cosine_similarity(feature_vec_1.reshape(1, -1), feature_vec_2.reshape(1, -1))
import operator

#function to get n closest merchants' names 
def closest(word,n):
    distances = dict()
    p=0
    k=0
    for w in document_vector_dict.keys():
        k+=1
        #if k%200==0 :
        #   print(k)
        #if k==24000:
        #    break
        distances[w] = get_cosine_similarity(document_vector_dict[w],document_vector_dict[word])
    d_sorted = dict(sorted(distances.items(), key=operator.itemgetter(1),reverse=True))
    #d_sorted = OrderedDict(sorted(distances.items(),key = lambda x:x[1] ,reverse = True))
    for items in d_sorted:
        print(items,'$$$$$',merchant_chain_name[items])
        print(d_sorted[items])
        p+=1
        if p==n:
            break

print(n_batches)
print(len(temp_dict))

#processing each query
print(result[687])  
closest(result[687],15)

