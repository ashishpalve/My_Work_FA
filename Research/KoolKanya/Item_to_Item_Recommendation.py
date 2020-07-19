#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().system('source activate python3')
#get_ipython().system('pip3 install pymongo')
#get_ipython().system('pip3 install dnspython')
#get_ipython().system('pip3 install pymongo[srv]')
#get_ipython().system('source deactivate')


# In[2]:


import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams

import math
import scipy
import sklearn
import configparser
import datetime


# In[3]:


import nltk
nltk.download('stopwords')


# In[4]:


from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from pylab import rcParams
from bs4 import BeautifulSoup
from bson import ObjectId


# # MongoDB Connection

# In[5]:


print("-----------------------------------------------------------------------------------------------------")
print("-----------------------------  Data Creation for Item to Item Similarity ----------------------------")
print("-----------------------------------------------------------------------------------------------------")


# In[6]:


config = configparser.ConfigParser()
config.read('reco_config.ini')
config.sections()


# In[7]:


mongodb_read_url = config['Connection']['mongodb_read_url']
mongodb_read_url


# In[ ]:


mongodb_write_url = config['Connection']['mongodb_write_url']
mongodb_write_url


# In[8]:


print("-----------------------------------------------------------------------------------------------------")
print("\nConnecting to the database:")
from pymongo import MongoClient
# pprint library is used to make the output look more pretty
from pprint import pprint

# connect to MongoDB, change the << MONGODB URL >> to reflect your own connection string
client = MongoClient(mongodb_read_url)


# In[9]:


db=client['prod-dump']


# # Articles Data

# In[10]:


print("-----------------------------------------------------------------------------------------------------")
print("\nReading Articles Data")


# In[11]:


collection = db.feeds
feeds_df = pd.DataFrame(list(collection.find()))
#print(feeds_df.shape)
#feeds_df.head()


# In[12]:


feeds_df = feeds_df.drop(columns = ['__v'])
feeds_df = feeds_df.rename(columns = {"_id":"contentId"})


# In[13]:


feeds_df['contentId'] = [str(st) for st in feeds_df['contentId']]
feeds_df['authorId'] = [str(st) for st in feeds_df['authorId']]
#print(feeds_df.shape)
#feeds_df.head()


# In[14]:


#feeds_df['resource_link'] = [st['link'] for st in feeds_df['resource']]
feeds_df['resource_videoUrl'] = [st['videoUrl'] for st in feeds_df['resource']]
feeds_df['resource_image'] = [st['image'] for st in feeds_df['resource']]


# In[15]:


feeds_df['createdAt'] = pd.to_datetime(feeds_df['createdAt'])
feeds_df['dt'] = feeds_df['createdAt'].dt.date
feeds_df['mnth'] = feeds_df['createdAt'].dt.month
feeds_df['yr'] = feeds_df['createdAt'].dt.year
feeds_df['yr_mnth'] = feeds_df['yr'].map(str) + '-' + feeds_df['mnth'].map(str)
print("Feeds Data:", feeds_df.shape)
#feeds_df.head()


# In[16]:


feeds_df = feeds_df[['contentId', 'anonymous', 'authorId', 'createdAt', 'content', 'isActive', 'isDelete', 'text', 'type', 'updatedAt', 'resource_videoUrl', 'resource_image']]


# In[17]:


articles_df = feeds_df[feeds_df['type'] == 'ARTICLE'].copy()
print("Articles Data: ", articles_df.shape)
articles_df.head(2)


# In[18]:


def parser_article(text):
    soup = BeautifulSoup(text, 'html.parser')
    text_list = soup.find_all('p')
    s = soup.find_all('p')

    s = [st.getText() for st in soup.find_all('p')]
    s2 = ''.join(s)
    return(s2)


# In[19]:


articles_df['parsed_text'] = [parser_article(st) if len(st) > 0 else '' for st in articles_df['content']]


# In[20]:


articles_df['article_text'] = articles_df['text'] + ' ' + articles_df['parsed_text']


# In[21]:


articles_df = articles_df.drop(['content'], axis=1)
articles_df = articles_df.drop(['parsed_text'], axis=1)


# In[ ]:





# # Item Similarity Algorithm

# In[22]:


#Ignoring stopwords (words with no semantics) from English and Portuguese (as we have a corpus with mixed languages)
stopwords_list = stopwords.words('english')

#Trains a model whose vectors size is 5000, composed by the main unigrams and bigrams found in the corpus, ignoring stopwords
vectorizer = TfidfVectorizer(analyzer='word',
                     ngram_range=(1, 2),
                     min_df=0.003,
                     max_df=0.5,
                     max_features=10000,
                     stop_words=stopwords_list)

item_ids = articles_df['contentId'].tolist()
tfidf_matrix = vectorizer.fit_transform(articles_df['article_text'])
tfidf_feature_names = vectorizer.get_feature_names()
#tfidf_matrix


# In[23]:


tfidf_matrix.shape[0]


# In[ ]:





# In[24]:


def get_similar_items_to_item(x, topn):
    item_precessed = item_ids[x]
    cosine_similarities = cosine_similarity(tfidf_matrix[x], tfidf_matrix)
    similar_indices = cosine_similarities.argsort().flatten()[-(topn+1):]
    similar_items = sorted([(item_ids[i], cosine_similarities[0,i]) for i in similar_indices], key=lambda x: -x[1])
    similar_items_df = pd.DataFrame(similar_items, columns = ['similar_contentId', 'cosine_score'])
    similar_items_df = similar_items_df[similar_items_df['similar_contentId'] != item_precessed]
    similar_items_df['contentId'] = item_precessed
    similar_items_df = similar_items_df[['contentId', 'similar_contentId', 'cosine_score']]
    return(similar_items_df)


# In[48]:


result = [get_similar_items_to_item(n, 100) for n in range(tfidf_matrix.shape[0])]


# In[49]:


item_recommendations = pd.concat(result)
#print(item_recommendations.shape)


# In[50]:


#item_recommendations.head()


# In[51]:


item_reco = item_recommendations.copy()
item_reco['cosine_score'] = item_reco['cosine_score'].round(3).map(str)
item_reco['content'] = item_reco.groupby(['contentId'])['similar_contentId'].transform(lambda x: ','.join(x))
item_reco['cosine_score_list'] = item_reco.groupby(['contentId'])['cosine_score'].transform(lambda x: ','.join(x))
#item_reco


# In[52]:


item_reco = item_reco[['contentId', 'content', 'cosine_score_list']]
item_reco = item_reco.drop_duplicates()
item_reco['content'] = item_reco['content'].str.split(',')
item_reco['cosine_score_list'] = item_reco['cosine_score_list'].str.split(',')
item_reco = item_reco.rename(columns = {'content':'similar_contentId', 'cosine_score_list':'cosine_score'})
item_reco['contentId'] = [ObjectId(st) for st in item_reco['contentId']]
item_reco['Active'] = 1


# In[53]:


today = datetime.datetime.now()
today = today.strftime("%Y-%m-%d %H:%M:%S")

item_reco['updatedAt'] = today


# In[54]:


item_reco.index = range(item_reco.shape[0])
#item_reco.head()


# In[55]:


item_recommendation_dict = item_reco.to_dict('records')


# In[ ]:





# # Write Item Recommendations to Collection

# In[ ]:


print("-----------------------------------------------------------------------------------------------------")
print("\nConnecting to the database:")
from pymongo import MongoClient
# pprint library is used to make the output look more pretty
from pprint import pprint

# connect to MongoDB, change the << MONGODB URL >> to reflect your own connection string
client = MongoClient(mongodb_write_url)


# In[ ]:


db=client['prod-dump']


# In[56]:


collection = db.item_recommendations


# In[34]:


#db.item_recommendations.drop()


# In[ ]:


collection.update_many({"Active":1}, {"$set": {"Active":0}})


# In[35]:


collection.insert_many(item_recommendation_dict)


# In[ ]:


collection.delete_many({"Active":0})


# In[36]:


collection = db.item_recommendations
item_recommendations = pd.DataFrame(list(collection.find()))
print("Data Written to Item recommendations Collection: ", item_recommendations.shape)
print("Data written at: ", item_recommendations['updatedAt'][0])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




