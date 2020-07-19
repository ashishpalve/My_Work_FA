#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import warnings
import re
import random
import math
import pandas as pd
import numpy as np
import scipy
import sklearn


# In[2]:


import nltk
nltk.download('stopwords')


# In[3]:


from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from pylab import rcParams


# In[4]:


feeds_df = pd.read_csv(os.getcwd() + '/Model_Output/feeds_df.csv')
feeds_df['createdAt'] = pd.to_datetime(feeds_df['createdAt'])
feeds_df['updatedAt'] = pd.to_datetime(feeds_df['updatedAt'])
feeds_df = feeds_df[~feeds_df['text'].isna()]
print(feeds_df.shape)
feeds_df.head()


# # Other Datasets

# In[5]:


#interactions_train_indexed_df = pd.read_csv(os.getcwd() + '/Model_Output/interactions_train_indexed_df.csv')
#interactions_train_indexed_df.head()


# In[6]:


interaction_total_indexed_df = pd.read_csv(os.getcwd() + '/Model_Output/interaction_total_indexed_df.csv')
interaction_total_indexed_df.head()


# In[ ]:





# In[7]:


def get_items_interacted(person_id, interactions_df):
    # Get the user's data and merge in the movie information.
    if(((pd.Series(person_id).isin(pd.Series(interactions_df.index.unique().to_list()))) == True)[0]):
        interacted_items = interactions_df.loc[person_id]['contentId']
        return(set(interacted_items if type(interacted_items) == pd.Series else [interacted_items]))
    else:
        interacted_items = set()
        return(interacted_items)
    


# In[8]:


recency_df = feeds_df[['contentId', 'createdAt']].copy()
recency_df['time_since_post'] = recency_df['createdAt'].max() - recency_df['createdAt']
recency_df = recency_df.sort_values(['time_since_post'], ascending=True)
recency_df['recency_rank'] = list(range(recency_df.shape[0]))
print(recency_df.shape)
recency_df.head()


# In[9]:


def recency_of_recommendation_func(all_recommendations, person_id):
    recommendation_with_recency = pd.merge(all_recommendations, recency_df[['contentId', 'createdAt', 'time_since_post', 'recency_rank']], how = 'left', on = 'contentId')
    recommendation_with_recency = recommendation_with_recency[recommendation_with_recency['recency_rank']<=800]
    return(recommendation_with_recency)


# # Popularity Model

# In[10]:


item_popularity_df = pd.read_csv(os.getcwd() + '/Model_Output/item_popularity_df.csv')
item_popularity_df.head()


# In[11]:


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


# In[12]:


warnings.simplefilter(action='ignore', category=FutureWarning)


# In[13]:


u_id = '5df76af011788b0016813414'
user_rec = popularity_model.recommend_items(u_id, items_to_ignore=get_items_interacted(u_id, 
                                         interaction_total_indexed_df), 
                                         topn=100)
user_rec = pd.merge(user_rec, feeds_df[['contentId', 'text']], how = 'left', on = 'contentId')
user_rec


# In[ ]:





# In[ ]:





# # Content-Based Filtering model

# In[14]:


item_ids = feeds_df['contentId'].tolist()


# In[15]:


import pickle

with open(os.getcwd() + '/Model_Output/tfidf_matrix.pkl', 'rb') as f:
    tfidf_matrix = pickle.load(f)
tfidf_matrix


# In[16]:


import pickle


# In[17]:


with open(os.getcwd() + '/Model_Output/user_profiles_all.pkl', 'rb') as f:
    user_profiles = pickle.load(f)
user_profiles


# In[18]:


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


# In[19]:


warnings.simplefilter(action='ignore', category=FutureWarning)


# In[ ]:





# In[20]:


u_id = '5e85f9294841b100281d4709'
user_rec = content_based_recommender_model.recommend_items(u_id, items_to_ignore=get_items_interacted(u_id, 
                                         interaction_total_indexed_df), 
                                         topn=100)
user_rec = pd.merge(user_rec, feeds_df[['contentId', 'text']], how = 'left', on = 'contentId')
user_rec


# In[ ]:





# In[ ]:





# In[ ]:





# # Collaborative Filtering model

# ### Matrix Factorization

# In[21]:


import pickle

with open(os.getcwd() + '/Model_Output/cf_preds_df_all.pkl', 'rb') as f:
    cf_preds_df = pickle.load(f)
cf_preds_df.shape


# In[22]:


cf_preds_df.head(2)


# In[23]:


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





# In[24]:


u_id = '5df72acdd092c70016203ac5'
user_rec = cf_recommender_model.recommend_items(u_id, items_to_ignore=get_items_interacted(u_id, 
                                         interaction_total_indexed_df), 
                                         topn=1000)
user_rec = pd.merge(user_rec, feeds_df[['contentId', 'text']], how = 'left', on = 'contentId')
user_rec


# In[ ]:





# In[ ]:





# ### Hybrid Recommendation 

# In[25]:


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





# In[26]:


u_id = '5df72acdd092c70016203ac5'
user_rec = hybrid_recommender_model.recommend_items(u_id, items_to_ignore=get_items_interacted(u_id, 
                                         interaction_total_indexed_df), 
                                         topn=100)
user_rec = pd.merge(user_rec, feeds_df[['contentId', 'text']], how = 'left', on = 'contentId')
user_rec


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




