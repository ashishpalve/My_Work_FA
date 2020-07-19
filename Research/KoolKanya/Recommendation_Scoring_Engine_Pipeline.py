#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().system('source activate python3')
#get_ipython().system('pip3 install pymongo')
#get_ipython().system('pip3 install dnspython')
#get_ipython().system('pip3 install pymongo[srv]')
#get_ipython().system('source deactivate')


# In[2]:


from Recommendation_Modelling_Pipeline_2 import *


# In[ ]:





# In[ ]:





# In[2]:


import os
import warnings
import re
import random
import math
import pandas as pd
import numpy as np
import scipy
import sklearn
import datetime
from bson import ObjectId
import configparser


# In[4]:


import nltk
nltk.download('stopwords')


# In[5]:


from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from pylab import rcParams


# # User List

# In[6]:


from pylab import rcParams


# In[7]:


config = configparser.ConfigParser()
config.read('reco_config.ini')
config.sections()


# In[8]:


mongodb_read_url = config['Connection']['mongodb_read_url']
mongodb_read_url


# In[9]:


mongodb_write_url = config['Connection']['mongodb_write_url']
mongodb_write_url


# In[10]:


print("-----------------------------------------------------------------------------------------------------")
print("\nConnecting to the database:")
from pymongo import MongoClient
# pprint library is used to make the output look more pretty
from pprint import pprint

# connect to MongoDB, change the << MONGODB URL >> to reflect your own connection string
client = MongoClient(mongodb_read_url)


# In[11]:


db=client['prod-dump']


# In[12]:


collection = db.users
users_df = pd.DataFrame(list(collection.find()))
print(users_df.shape)


# In[13]:


users_df = users_df.drop(columns = ['__v'])
users_df = users_df[['_id', 'firstName', 'lastName', 'isActive', 'points']]
users_df = users_df.rename(columns = {"_id":"userId"})
users_df['userId'] = [str(st) for st in users_df['userId']]
users_df = users_df[users_df['isActive'] == True]
users_df = users_df.drop_duplicates()
users_df.head()


# In[ ]:





# # Feeds Data

# In[14]:


#feeds_df = pd.read_csv(os.getcwd() + '/Model_Output/feeds_df.csv')
#feeds_df['createdAt'] = pd.to_datetime(feeds_df['createdAt'])
#feeds_df['updatedAt'] = pd.to_datetime(feeds_df['updatedAt'])
#feeds_df = feeds_df[~feeds_df['text'].isna()]
#print(feeds_df.shape)
#feeds_df.head()


# # Other Datasets

# In[15]:


interaction_total_indexed_df['personId'] = interaction_total_indexed_df.index
interaction_total_indexed_df = interaction_total_indexed_df[['personId', 'contentId', 'eventStrength']]
interaction_total_indexed_df.index = range(interaction_total_indexed_df.shape[0])
interaction_total_indexed_df.head()


# In[16]:


#interaction_total_indexed_df = pd.read_csv(os.getcwd() + '/Model_Output/interaction_total_indexed_df.csv')
#interaction_total_indexed_df.head()


# In[17]:


def get_items_interacted(person_id, interactions_df):
    # Get the user's data and merge in the movie information.
    if(((pd.Series(person_id).isin(pd.Series(interactions_df.index.unique().to_list()))) == True)[0]):
        interacted_items = interactions_df.loc[person_id]['contentId']
        return(set(interacted_items if type(interacted_items) == pd.Series else [interacted_items]))
    else:
        interacted_items = set()
        return(interacted_items)
    


# In[18]:


recency_df = feeds_df[['contentId', 'createdAt']].copy()
recency_df['time_since_post'] = recency_df['createdAt'].max() - recency_df['createdAt']
recency_df = recency_df.sort_values(['time_since_post'], ascending=True)
recency_df['recency_rank'] = list(range(recency_df.shape[0]))
print(recency_df.shape)
recency_df.head()


# In[19]:


def recency_of_recommendation_func(all_recommendations, person_id):
    recommendation_with_recency = pd.merge(all_recommendations, recency_df[['contentId', 'createdAt', 'time_since_post', 'recency_rank']], how = 'left', on = 'contentId')
    recommendation_with_recency = recommendation_with_recency[recommendation_with_recency['recency_rank']<=recency_cutoff]
    return(recommendation_with_recency)


# # Separate Interaction & Non Interaction data for Scoring

# In[20]:


interaction_users = interaction_total_indexed_df.groupby(['personId'])['contentId'].count().reset_index()
interaction_users = interaction_users.rename(columns = {'contentId':'content_Ct'})

users_interaction_df = pd.merge(users_df, interaction_users, left_on = ['userId'], right_on = ['personId'], how = 'left')
users_interaction_df = users_interaction_df.drop(columns = ['personId'])
print(users_interaction_df.shape)

users_with_interaction = users_interaction_df[~(users_interaction_df['content_Ct'].isna())].copy()
users_with_interaction = users_with_interaction.drop(columns = ['content_Ct'])
print("Users with Interaction: ", users_with_interaction.shape)

users_without_interaction = users_interaction_df[users_interaction_df['content_Ct'].isna()].copy()
users_without_interaction = users_without_interaction.drop(columns = ['content_Ct'])
print("Users without Interaction: ", users_without_interaction.shape)

del interaction_users, users_interaction_df


# In[ ]:





# In[ ]:





# # Popularity Model

# In[21]:


#item_popularity_df = pd.read_csv(os.getcwd() + '/Model_Output/item_popularity_df.csv')
#item_popularity_df.head()


# In[22]:


class PopularityRecommender:
    
    MODEL_NAME = 'Popularity'
    
    def __init__(self, popularity_df, items_df=None):
        self.popularity_df = popularity_df
        self.items_df = items_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        # Recommend the more popular items that the user hasn't seen yet.
        recommendations_df = self.popularity_df[~self.popularity_df['contentId'].isin(items_to_ignore)]                                .sort_values('eventStrength', ascending = False)                                .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                          left_on = 'contentId', 
                                                          right_on = 'contentId')[['eventStrength', 'contentId']]
        
        #recommendations_df = recency_of_recommendation_func(recommendations_df, user_id)


        return recommendations_df
    
popularity_model = PopularityRecommender(item_popularity_df, feeds_df)


# In[23]:


warnings.simplefilter(action='ignore', category=FutureWarning)


# In[24]:


#u_id = '5edd2d51d78504002252480f'
#user_rec = popularity_model.recommend_items(u_id, items_to_ignore=get_items_interacted(u_id, 
#                                         interaction_total_indexed_df), 
#                                         topn=100)
#user_rec = pd.merge(user_rec, feeds_df[['contentId', 'text']], how = 'left', on = 'contentId')
#user_rec


# In[ ]:





# In[ ]:





# # Content-Based Filtering model

# In[25]:


item_ids = feeds_df['contentId'].tolist()


# In[26]:


#import pickle

#with open(os.getcwd() + '/Model_Output/tfidf_matrix.pkl', 'rb') as f:
#    tfidf_matrix = pickle.load(f)
#tfidf_matrix


# In[27]:


import pickle


# In[28]:


#with open(os.getcwd() + '/Model_Output/user_profiles_all.pkl', 'rb') as f:
#    user_profiles = pickle.load(f)
#user_profiles


# In[29]:


user_profiles = user_profiles_all


# In[30]:


class ContentBasedRecommender:
    
    MODEL_NAME = 'Content-Based'
    
    def __init__(self, items_df=None):
        self.item_ids = item_ids
        self.items_df = items_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def _get_similar_items_to_user_profile(self, person_id, topn=1000):
        #Computes the cosine similarity between the user profile and all item profiles
        cosine_similarities = cosine_similarity(user_profiles[person_id], tfidf_matrix)
        #Gets the top similar items
        similar_indices = cosine_similarities.argsort().flatten()[-topn:]
        #Sort the similar items by similarity
        similar_items = sorted([(item_ids[i], cosine_similarities[0,i]) for i in similar_indices], key=lambda x: -x[1])
        return similar_items
        
    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        similar_items = self._get_similar_items_to_user_profile(user_id)
        #Ignores items the user has already interacted
        similar_items_filtered = list(filter(lambda x: x[0] not in items_to_ignore, similar_items))
        
        recommendations_df = pd.DataFrame(similar_items_filtered, columns=['contentId', 'recStrength'])                                     .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                          left_on = 'contentId', 
                                                          right_on = 'contentId')[['recStrength', 'contentId']]
        
        recommendations_df = recency_of_recommendation_func(recommendations_df, user_id)
        
        return recommendations_df
    
content_based_recommender_model = ContentBasedRecommender(feeds_df)


# In[31]:


warnings.simplefilter(action='ignore', category=FutureWarning)


# In[ ]:





# In[32]:


#u_id = '5e85f9294841b100281d4709'
#user_rec = content_based_recommender_model.recommend_items(u_id, items_to_ignore=get_items_interacted(u_id, 
#                                         interaction_total_indexed_df), 
#                                         topn=100)
#user_rec = pd.merge(user_rec, feeds_df[['contentId', 'text']], how = 'left', on = 'contentId')
#user_rec


# In[ ]:





# In[ ]:





# # Collaborative Filtering model

# ### Matrix Factorization

# In[33]:


#import pickle

#with open(os.getcwd() + '/Model_Output/cf_preds_df_all.pkl', 'rb') as f:
#    cf_preds_df = pickle.load(f)
#cf_preds_df.shape


# In[34]:


cf_preds_df = cf_preds_df_all


# In[35]:


cf_preds_df.head(2)


# In[36]:


class CFRecommender:
    
    MODEL_NAME = 'Collaborative Filtering'
    
    def __init__(self, cf_predictions_df, items_df=None):
        self.cf_predictions_df = cf_predictions_df
        self.items_df = items_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        # Get and sort the user's predictions
        sorted_user_predictions = self.cf_predictions_df[user_id].sort_values(ascending=False)                                     .reset_index().rename(columns={user_id: 'recStrength'})

        # Recommend the highest predicted rating movies that the user hasn't seen yet.
        recommendations_df = sorted_user_predictions[~sorted_user_predictions['contentId'].isin(items_to_ignore)]                                .sort_values('recStrength', ascending = False)                                .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                          left_on = 'contentId', 
                                                          right_on = 'contentId')[['recStrength', 'contentId']]

        recommendations_df = recency_of_recommendation_func(recommendations_df, user_id)
        return recommendations_df
    
cf_recommender_model = CFRecommender(cf_preds_df, feeds_df)


# In[ ]:





# In[ ]:





# In[ ]:





# ### Hybrid Recommendation 

# In[37]:


class HybridRecommender:
    
    MODEL_NAME = 'Hybrid'
    
    def __init__(self, cb_rec_model, cf_rec_model, items_df, cb_ensemble_weight=1.0, cf_ensemble_weight=1.0):
        self.cb_rec_model = cb_rec_model
        self.cf_rec_model = cf_rec_model
        self.cb_ensemble_weight = cb_ensemble_weight
        self.cf_ensemble_weight = cf_ensemble_weight
        self.items_df = items_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        #Getting the top-1000 Content-based filtering recommendations
        cb_recs_df = self.cb_rec_model.recommend_items(user_id, items_to_ignore=items_to_ignore, verbose=verbose,
                                                           topn=1000).rename(columns={'recStrength': 'recStrengthCB'})
        cb_recs_df = cb_recs_df.drop(['createdAt', 'time_since_post', 'recency_rank'], axis=1)
        
        #Getting the top-1000 Collaborative filtering recommendations
        cf_recs_df = self.cf_rec_model.recommend_items(user_id, items_to_ignore=items_to_ignore, verbose=verbose, 
                                                           topn=1000).rename(columns={'recStrength': 'recStrengthCF'})
        cf_recs_df = cf_recs_df.drop(['createdAt', 'time_since_post', 'recency_rank'], axis=1)
        
        #Combining the results by contentId
        recs_df = cb_recs_df.merge(cf_recs_df,
                                   how = 'outer', 
                                   left_on = 'contentId', 
                                   right_on = 'contentId').fillna(0.0)
        
        #Computing a hybrid recommendation score based on CF and CB scores
        #recs_df['recStrengthHybrid'] = recs_df['recStrengthCB'] * recs_df['recStrengthCF'] 
        recs_df['recStrengthHybrid'] = (recs_df['recStrengthCB'] * self.cb_ensemble_weight)                                      + (recs_df['recStrengthCF'] * self.cf_ensemble_weight)
        
        #Sorting recommendations by hybrid score
        recommendations_df = recs_df.sort_values('recStrengthHybrid', ascending=False).head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                          left_on = 'contentId', 
                                                          right_on = 'contentId')[['recStrengthHybrid', 'contentId']]

        recommendations_df = recency_of_recommendation_func(recommendations_df, user_id)
        return recommendations_df
    
hybrid_recommender_model = HybridRecommender(content_based_recommender_model, cf_recommender_model, feeds_df,
                                             cb_ensemble_weight=1.0, cf_ensemble_weight=50.0)


# In[ ]:





# In[38]:


#u_id = '5df72acdd092c70016203ac5'
#user_rec = hybrid_recommender_model.recommend_items(u_id, items_to_ignore=get_items_interacted(u_id, 
#                                         interaction_total_indexed_df), 
#                                         topn=100)
#user_rec = pd.merge(user_rec, feeds_df[['contentId', 'text']], how = 'left', on = 'contentId')
#user_rec


# In[ ]:





# # Hybrid Recommendations for users with Interaction

# In[39]:


userid_wth_interaction = pd.Series(interaction_total_indexed_df['personId'].unique())
def final_recommendation(userid):
    if (pd.Series(userid).isin(userid_wth_interaction)[0]):
        user_rec = hybrid_recommender_model.recommend_items(userid, items_to_ignore=get_items_interacted(userid, 
                                         interaction_total_indexed_df), 
                                         topn=100)
        user_rec = user_rec[['contentId', 'recStrengthHybrid']]
        user_rec = user_rec.rename(columns = {'recStrengthHybrid':'recommendation_score'})
        user_rec['score_type'] = 'Hybrid Reco Strength'
        user_rec['model'] = 'Hybrid'
        
    else:
        user_rec = popularity_model.recommend_items(userid, items_to_ignore=get_items_interacted(userid, 
                                         interaction_total_indexed_df), 
                                         topn=100)
        user_rec = user_rec[['contentId', 'eventStrength']]
        user_rec = user_rec.rename(columns = {'eventStrength':'recommendation_score'})
        user_rec['score_type'] = 'event Strength'
        user_rec['model'] = 'Popularity'
    user_rec['userId'] = userid
    user_rec = user_rec[['userId', 'contentId', 'recommendation_score', 'score_type', 'model']]
    return(user_rec)


# In[40]:


#recommendations_for_interaction_users = pd.DataFrame(columns = ['userId', 'contentId', 'recommendation_score', 'score_type', 'model'])
#recommendations_for_interaction_users

#for i in users_with_interaction['userId'].to_list():
#    user_recommendation = final_recommendation(u_id)
#    recommendations_for_interaction_users = recommendations_for_interaction_users.append(user_recommendation)


# In[ ]:


print("Generating All Interaction based recommendation Starts: ", datetime.datetime.now())


# In[41]:


user_list = users_with_interaction['userId'].to_list()
result = [final_recommendation(u_id) for u_id in user_list]


# In[42]:


recommendations_for_interaction_users = pd.concat(result)
del result


# In[ ]:


print("Generating All Interaction based recommendation Ends: ", datetime.datetime.now())


# In[43]:


print(recommendations_for_interaction_users.shape)
recommendations_for_interaction_users.head()


# # Popularity Based Recommendation for users without interaction

# In[44]:


u_id = 'dummy'
user_rec = popularity_model.recommend_items(u_id, items_to_ignore=get_items_interacted(u_id, 
                                         interaction_total_indexed_df), 
                                         topn=100)
user_rec = user_rec[['contentId', 'eventStrength']]
user_rec = user_rec.rename(columns = {'eventStrength':'recommendation_score'})
user_rec['score_type'] = 'event Strength'
user_rec['model'] = 'Popularity'
user_rec.shape


# In[45]:


user_rec.head()


# In[46]:


#users_without_interaction['key'] = 0
#user_rec['key'] = 0


# In[47]:


#recommendations_for_non_interaction_users = pd.merge(users_without_interaction[['userId', 'key']], user_rec, how = 'outer', on = 'key')
#recommendations_for_non_interaction_users = recommendations_for_non_interaction_users.drop(columns = ['key'])
#print(recommendations_for_non_interaction_users.shape)
#recommendations_for_non_interaction_users.head()


# In[48]:


recommendations_for_non_interaction_users = user_rec.copy()


# In[49]:


recommendations_for_non_interaction_users['userId'] = '9e909b900d999c0099fa9f9e'
recommendations_for_non_interaction_users = recommendations_for_non_interaction_users.drop_duplicates()
recommendations_for_non_interaction_users = recommendations_for_non_interaction_users[['userId', 'contentId', 'recommendation_score', 'score_type', 'model']]
recommendations_for_non_interaction_users.shape


# In[ ]:





# In[50]:


all_recommendations = recommendations_for_interaction_users.append(recommendations_for_non_interaction_users)
all_recommendations.shape


# In[ ]:





# In[51]:


all_reco = all_recommendations.copy()
all_reco['recommendation_score'] = all_reco['recommendation_score'].round(2).map(str)
all_reco['content'] = all_reco.groupby(['userId'])['contentId'].transform(lambda x: ','.join(x))
all_reco['reco_score'] = all_reco.groupby(['userId'])['recommendation_score'].transform(lambda x: ','.join(x))


# In[52]:


all_reco = all_reco[['userId', 'content', 'reco_score', 'score_type', 'model']]
all_reco = all_reco.drop_duplicates()
all_reco['content'] = all_reco['content'].str.split(',')
all_reco['reco_score'] = all_reco['reco_score'].str.split(',')
all_reco = all_reco.rename(columns = {'content':'contentId', 'reco_score':'recommendation_score'})
all_reco['userId'] = [ObjectId(st) for st in all_reco['userId']]
all_reco['Active'] = 1
all_reco.head()


# In[ ]:





# In[53]:


today = datetime.datetime.now()
today = today.strftime("%Y-%m-%d %H:%M:%S")

all_reco['updatedAt'] = today


# In[54]:


all_reco.index = range(all_reco.shape[0])
all_reco.head()


# In[55]:


recommendation_dict = all_reco.to_dict('records')


# In[ ]:





# In[ ]:





# In[56]:


print("-----------------------------------------------------------------------------------------------------")
print("\nConnecting to the database:")
from pymongo import MongoClient
# pprint library is used to make the output look more pretty
from pprint import pprint

# connect to MongoDB, change the << MONGODB URL >> to reflect your own connection string
client = MongoClient(mongodb_write_url)


# In[57]:


db=client['prod-dump']


# In[58]:


collection = db.user_recommendations


# In[59]:


#db.user_recommendations.drop()


# In[59]:


collection.update_many({"Active":1}, {"$set": {"Active":0}})


# In[60]:


collection.insert_many(recommendation_dict)


# In[61]:


collection.delete_many({"Active":0})


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[62]:


collection = db.user_recommendations
user_recommendations = pd.DataFrame(list(collection.find()))
print("Data Written to User recommendations Collection: ", user_recommendations.shape)
print("Data written at: ", user_recommendations['updatedAt'][0])


# In[63]:


user_recommendations.head()


# In[64]:


user_recommendations.shape


# In[65]:


user_recommendations.head()


# In[ ]:




