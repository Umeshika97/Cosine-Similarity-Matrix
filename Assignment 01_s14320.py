#!/usr/bin/env python
# coding: utf-8

# In[238]:


#imort libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity;

from numpy import dot
from numpy.linalg import norm
import pandas as pd
import sklearn as sk
import numpy as np
from numpy import dot
import re

from nltk.tokenize import word_tokenize
import string
import sys
import gc


# In[239]:


#the function define for read text files
def read_text_doc(text_document): 
    text_data =  open(text_document).read()
    return text_data


# In[240]:


#get the text file data
Doc01=read_text_doc("doc 1.txt")
Doc02=read_text_doc("doc 2.txt")
Doc03=read_text_doc("doc 3.txt")
Doc04=read_text_doc("doc 4.txt")
Doc05=read_text_doc("doc 5.txt")
Doc06=read_text_doc("doc 6.txt")
Doc07=read_text_doc("doc 7.txt")
Doc08=read_text_doc("doc 8.txt")


# In[241]:


#create a single text data file
All_Doc=[Doc01,Doc02,Doc03,Doc04,Doc05,Doc06,Doc07,Doc08]


# In[242]:


#TF-IDF vector 
TF_DF_vector=TfidfVectorizer();
#TF-IDF values is found for all text
TF_DF_vector_Tansform= TF_DF_vector.fit_transform(All_Doc, y=None);
# cosine similarity matrix
Cos_sim_matrix = cosine_similarity(TF_DF_vector_Tansform);
print(Cos_sim_matrix);


# In[243]:


#cosine similarity matrix with in data fram
cos_simM=pd.DataFrame(cosine_similarity(TF_DF_vector_Tansform,TF_DF_vector_Tansform))
cos_simM


# In[244]:


print(TF_DF_vector_Tansform)


# In[275]:


def Identify_doc_topic(Topicname):
    Topic_trans = str.maketrans(string.punctuation+string.ascii_uppercase,
                                " "*len(string.punctuation)+string.ascii_lowercase)
      
    Topicname = Topicname.translate(Topic_trans)
    Topicname=Topicname.split()
    #Topicname = Topicname[int('1')]
    limit=len(Topicname)-1



    ST = pd.DataFrame([TF_IDF[Topicname[0]], TF_IDF[Topicname[1]], TF_IDF[Topicname[2]], TF_IDF[Topicname[3]]
                              + TF_IDF[Topicname[3-1]] + TF_IDF[Topicname[3]]]
                             ,index=[Topicname[0], Topicname[1], Topicname[2], Topicname[0] +Topicname[1] +Topicname[2]]).T
    ST = ST[ST[Topicname[0]+ Topicname[1]+ Topicname[2]] > 0]
    ST = ST[ST[Topicname[0]] > 0]
    ST
    output = ST.sort_values([Topicname[0]+ Topicname[1]+ Topicname[2]], ascending=[True])
    return output


# In[276]:


topic01='Hurricane Gilbert Heads Toward Dominican Coast'
Identify_doc_topic(topic01)


# The document 2, 3,7 are related to the 
# topic of "Hurricane Gilbert Heads Toward Dominican Coast" 

# In[287]:


from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer
stemmer = SnowballStemmer("english")

f=TfidfVectorizer();
def boring_tokenizer(input_word):
    term = re.sub(r"[^A-Za-z0-9\-]", " ", input_word).lower().split()
    return term

def find_topic_in_doc(topic_name,n):
    count_vectorizer = CountVectorizer(stop_words='english', tokenizer=boring_tokenizer)
    X = count_vectorizer.fit_transform(topic_name)
    Topicname=count_vectorizer.get_feature_names()
    N=len(Topicname) 
     
    best_word = Topicname[n].split()
    count = TF_DF_vector.vocabulary_[Topicname[n]]
    
    idf_df = np.transpose(TF_DF_vector_Tansform[:,count])
    print('The no of words in topic',N)
    print("The considering word in topic",best_word)
    print('TF-IDF values for the best identifying word \n', np.transpose(TF_DF_vector_Tansform[:,count]))
    
    

topic01=['Hurricane Gilbert Heads Toward Dominican Coast']
topic02= ["McDonald's Opens First Restaurant in China"]
topic03=['ira terrorist attack']


# In[288]:


find_topic_in_doc(topic01,0)


# In[289]:


find_topic_in_doc(topic01,1)


# In[290]:


find_topic_in_doc(topic01,2)


# In[291]:


find_topic_in_doc(topic01,3)


# In[292]:


find_topic_in_doc(topic03,0)


# In[293]:


find_topic_in_doc(topic03,1)


# In[294]:


find_topic_in_doc(topic02,3)


# In[295]:


find_topic_in_doc(topic02,1)


# In[274]:


gc.collect();


# In[ ]:




