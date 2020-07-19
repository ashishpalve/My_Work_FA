#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import warnings
import pandas as pd
import numpy as np
from datetime import timedelta
import pickle


# In[2]:


import nltk


# In[3]:


#import nltk
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

#from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
#from sklearn import decomposition, ensemble
#from sklearn.model_selection import train_test_split

#import xgboost

from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
from tensorflow import keras


# In[4]:


nltk.download(['punkt', 'stopwords'])


# In[5]:


def del_stop(sent, stop_word_list):
    return [term for term in sent if term not in stop_word_list]

porter = PorterStemmer()
def stem_tokens(token_list):
    token_stem = [porter.stem(term) for term in token_list]
    return(token_stem)

def get_clean_text(txt):
    txt = [txt]
    feed_token = [word_tokenize(sent.lower()) for sent in txt]
    
    stop_punct = list(punctuation)
    stop_nltk = stopwords.words("english")
    stop_updated = stop_nltk + stop_punct
    
    feed_token_clean = [del_stop(sent, stop_updated) for sent in feed_token]
    feed_token_stemmed = [stem_tokens(tk) for tk in feed_token_clean]
    
    clean_text = [" ".join(sent) for sent in feed_token_clean]
    return(clean_text)


# # Load Model & Feature Generator

# In[6]:


def load_objects_func():
    #Load the TFIDF Vectoriser
    with open(os.getcwd() + '/Model_Output/feature_transformer.pkl', 'rb') as f:
        tfidf_vect = pickle.load(f)
    
    #Load the pre-trained model
    with open(os.getcwd() + '/Model_Output/model_file.pkl', 'rb') as f:
        model = pickle.load(f)
    
    #Load Encoder mappings used in the model to get the name of the primary hastags
    enc_mapping = pd.read_csv(os.getcwd() + '/Model_Output/encoder_mapping.csv')
    enc_mapping = enc_mapping.sort_values(['hashtag_encoding'])
    
    return(tfidf_vect, model, enc_mapping)
    


# In[ ]:





# In[ ]:





# # Generate Predictions

# In[7]:


def get_hashtag_prediction(input_text, topN):
    input_text = [input_text]
    
    #Load the tfidf vectorizer, model and encoder mappings
    tfidf_vectorizer, classifier, encoder_mapping = load_objects_func()
    
    #Clean the text
    clean_text = get_clean_text(input_text)
    
    #Generate Features for scoring
    feature_vector = tfidf_vectorizer.transform(clean_text)
    
    #Generate Predictions
    prob_predictions = classifier.predict(feature_vector)
    
    prediction_df = encoder_mapping.copy()
    prediction_df = prediction_df.sort_values(['hashtag_encoding'])
    prediction_df['probability'] = prob_predictions.tolist()[0]
    prediction_df = prediction_df.sort_values(['probability'], ascending=False)
    topN_hashtag = prediction_df[0:topN]['hashtag'].to_list()
    return(topN_hashtag)


# In[9]:


input_txt = 'Hello. I am Yashi. I have just started freelancing as a digital marketer. I want to know how to get my freelance digital marketing business off the ground even when I have zero experience?'

get_hashtag_prediction(input_txt, 3)


# In[10]:


input_txt = 'Hello. I am XYZ. i am looking for freelancing jobs in the field of analytics'

get_hashtag_prediction(input_txt, 3)


# In[11]:


input_txt = 'Hi everyone I am Abhilasha like most of us my mother is also my inspiration.I always wanted to do something for her give her the recognisation that she deserves and also to every other homemakers like her so I came across an idea of creating an online platform for the showcase of every things that they make(not just food)and started telling the world about them.I run my page in instagram but I have little help from the people I know. I lack in networking.I do not want to stop my work and in need to more stories. Can you please show me some way that how I should attract people to tell their stories to the world  how I may convince my friends to share their mothers tales or my followers'

get_hashtag_prediction(input_txt, 3)


# In[12]:


input_txt = 'Hi , Hope everyone is safe, healthy and having a great weekend. I am a second-year MBA student and the past few months have been very transformational for me on a professional and personal level. I am looking to collaborate on exciting new projects and ideas. To give you an overview of my skills I am attaching my LinkedIn profile (https://www.linkedin.com/in/tanya-r-dwivedii/). My fields of interest are marketing and data analytics. Cheers! #networking #marketing #data analytics #productdesign'

get_hashtag_prediction(input_txt, 3)


# In[13]:


input_txt = 'How depressing is depression ?  I sit watching the clock tick  Anxious its reaching the sun  Nervous it doesnt stop  The pale blue now a taboo  As I wish I slept through the depths  Not knowing the blue. Life as a boat had me stowing on it  The tides hitting the sides  And the waters downcasting my eyes  The sailors pricking my skin  Akin my abused core  My head bowling against the wind  Flowing Tears of contempt, result of the force . The voice box scarred from the screams  Locked with chains of grief  Never reefed a ripple in the sea, Could never reef a ripple in the sea... The steady warm currents of the deep  Promising the bracing  As tempting and rushing  For a refugee of yesterday. The shaken Impulse Met the reflecting bed And now The corals chattered her ignored fretting waves.'

get_hashtag_prediction(input_txt, 3)


# In[ ]:




