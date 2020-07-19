#!/usr/bin/env python
# coding: utf-8

# In[46]:


#get_ipython().system('source activate python3')
#get_ipython().system('pip3 install pymongo')
#get_ipython().system('pip3 install dnspython')
#get_ipython().system('pip3 install pymongo[srv]')
#get_ipython().system('pip3 install xgboost')
#get_ipython().system('source deactivate')


# In[47]:


from Data_Creation_for_Tag_Classification import *


# In[48]:


import nltk
nltk.download(['punkt', 'stopwords'])


# In[49]:


import os
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import timedelta
import pickle
import configparser


# In[50]:


from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from sklearn.model_selection import train_test_split

import xgboost

#from keras.preprocessing import text, sequence
#from keras import layers, models, optimizers
#from tensorflow import keras


# In[51]:


config = configparser.ConfigParser()
config.read('reco_config_Tag_Classification.ini')
config.sections()


# In[52]:


model_type = config['Modelling']['MODEL_NAME']
model_type


# In[53]:


#feed_hashtag = pd.read_csv(os.getcwd() + '/Datasets/feed_hashtag_long.csv')
#feed_hashtag['createdAt'] = pd.to_datetime(feed_hashtag['createdAt'])
#print(feed_hashtag.shape)
#feed_hashtag.head(1)


# In[ ]:





# In[54]:


feed_hashtag = feed_hashtag[feed_hashtag['isPrimary'] == True]
feed_hashtag = feed_hashtag[~feed_hashtag['text'].isna()]
print("Data Size:", feed_hashtag.shape[0])
print("Unique Content:", feed_hashtag['contentId'].nunique())
print("Unique tags:", feed_hashtag['hashtag'].nunique())


# In[55]:


feed_hashtag['createDate'] = feed_hashtag['createdAt'].dt.date
feed_hashtag.head(1)


# In[56]:


#feed_hashtag.groupby(['createDate']).agg({'contentId':'nunique', 'tagIds':'count'})


# In[57]:


feed_hashtag = feed_hashtag.sort_values(['contentId'])


# In[58]:


#feed_hashtag.groupby(['contentId'])['hashtag'].count().reset_index().sort_values(['hashtag'], ascending = False).head()


# In[59]:


hashtag_class_ct = feed_hashtag.groupby(['hashtag'])['contentId'].agg([('row_count','count'), ('content_count','nunique')]).reset_index()
hashtag_class_ct = hashtag_class_ct.sort_values(['content_count'], ascending=False)
hashtag_class_ct
f, ax = plt.subplots(figsize=(18,5))
plt.bar(hashtag_class_ct['hashtag'], hashtag_class_ct['content_count'])
plt.xlabel('hashtag', fontsize=20)
plt.ylabel('No of Feeds', fontsize=20)
plt.xticks(hashtag_class_ct['hashtag'], fontsize=20, rotation=30)
plt.title('hashtag counts')

plt.title("Number of Post for every Primary Hashtag")
plt.savefig(os.getcwd()+'/Charts/Total_Post_Per_Primary_Hashtag.png')

plt.show()
plt.close()


# In[60]:


feed_hashtag_rollup = feed_hashtag.groupby(['contentId', 'authorId_content', 'createdAt', 'text', 'type', 'isActive', 'isPrimary'])['hashtag'].agg([('hashtag_count','count'), ('hashtag_list', ','.join)]).reset_index()
print(feed_hashtag_rollup.shape)
feed_hashtag_rollup.sort_values(['hashtag_count'],ascending=False).head(2)


# In[61]:


train_test_cutoff = pd.to_datetime(feed_hashtag['createDate'].max() - timedelta(10))
train_test_cutoff


# In[62]:


train = feed_hashtag[feed_hashtag['createdAt'] <= train_test_cutoff].copy()
print("train data:", train.shape[0])
test = feed_hashtag[feed_hashtag['createdAt'] > train_test_cutoff].copy()
print("test data:", test.shape[0])


# In[63]:


print("---------------------------------------------------------------------------------------------------")
print("----------------------------------     Data Processing Started     --------------------------------")
print("---------------------------------------------------------------------------------------------------")


# # Text Cleansing

# In[64]:


def del_stop(sent, stop_word_list):
    return [term for term in sent if term not in stop_word_list]

porter = PorterStemmer()
def stem_tokens(token_list):
    token_stem = [porter.stem(term) for term in token_list]
    return(token_stem)

def get_clean_text(txt):
    feed_token = [word_tokenize(sent.lower()) for sent in txt]
    
    stop_punct = list(punctuation)
    stop_nltk = stopwords.words("english")
    stop_updated = stop_nltk + stop_punct
    
    feed_token_clean = [del_stop(sent, stop_updated) for sent in feed_token]
    feed_token_stemmed = [stem_tokens(tk) for tk in feed_token_clean]
    
    clean_text = [" ".join(sent) for sent in feed_token_clean]
    return(clean_text)


# In[65]:


train['text_clean'] = get_clean_text(train.text)
train.head(1)


# In[66]:


test['text_clean'] = get_clean_text(test.text)
test.head(1)


# In[67]:


train_rollup = train.groupby(['contentId', 'authorId_content', 'createdAt', 'text', 'type', 'isActive', 'isPrimary', 'text_clean'])['hashtag'].agg([('hashtag_count','count'), ('hashtag_list', ','.join)]).reset_index()
train_rollup.shape


# In[ ]:





# In[68]:


train_df_rollup, validation_rollup = model_selection.train_test_split(train_rollup, test_size=0.20)
print(train_df_rollup.shape, validation_rollup.shape)
train_df = train[train['contentId'].isin(train_df_rollup['contentId'])]
validation = train[train['contentId'].isin(validation_rollup['contentId'])]
print(train_df.shape)
print(validation.shape)


# In[69]:


test_rollup = test.groupby(['contentId', 'authorId_content', 'createdAt', 'text', 'type', 'isActive', 'isPrimary', 'text_clean'])['hashtag'].agg([('hashtag_count','count'), ('hashtag_list', ','.join)]).reset_index()
print(test_rollup.shape)
test_rollup.sort_values(['hashtag_count'],ascending=False).head(2)


# In[ ]:





# In[70]:


train_x = train_df['text_clean']
train_y = train_df['hashtag']
valid_x = validation_rollup['text_clean']
valid_y = validation_rollup['hashtag_list']
test_x = test_rollup['text_clean']
test_y = test_rollup['hashtag_list']


# In[71]:


# label encode the target variable 
encoder = preprocessing.LabelEncoder()
train_y_enc = encoder.fit_transform(train_y)


# In[72]:


mapping = pd.DataFrame({'hashtag':train_y, 'hashtag_encoding':train_y_enc})
mapping = mapping.drop_duplicates()
mapping.sort_values(['hashtag_encoding'])
mapping.to_csv(os.getcwd() + '/Model_Output/encoder_mapping.csv', index=False)


# In[ ]:





# # Feature Creation

# In[73]:


text_for_feature = train['text_clean']


# In[90]:


def feature_creation_func(feature_type):
    if(feature_type == 'Count Vectorizer'):
        # create a count vectorizer object 
        count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
        count_vect.fit(text_for_feature)
        # transform the training and validation data using count vectorizer object
        xtrain_feature =  count_vect.transform(train_x)
        xvalid_feature =  count_vect.transform(valid_x)
        xtest_feature = count_vect.transform(test_x)
        pickle.dump(count_vect, open(os.getcwd() + '/Model_Output/feature_transformer.pkl', "wb"))
    
    if(feature_type == 'Word TFIDF'):
        # word level tf-idf
        tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
        tfidf_vect.fit(text_for_feature)
        xtrain_feature =  tfidf_vect.transform(train_x)
        xvalid_feature =  tfidf_vect.transform(valid_x)
        xtest_feature = tfidf_vect.transform(test_x)
        pickle.dump(tfidf_vect, open(os.getcwd() + '/Model_Output/feature_transformer.pkl', "wb"))
        
    if(feature_type == 'N-gram TFIDF'):
        # ngram level tf-idf 
        tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
        tfidf_vect_ngram.fit(text_for_feature)
        xtrain_feature =  tfidf_vect_ngram.transform(train_x)
        xvalid_feature =  tfidf_vect_ngram.transform(valid_x)
        xtest_feature =  tfidf_vect_ngram.transform(test_x)
        pickle.dump(tfidf_vect_ngram, open(os.getcwd() + '/Model_Output/feature_transformer.pkl', "wb"))
        
    if(feature_type == 'Character TFIDF'):
        # characters level tf-idf
        tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
        tfidf_vect_ngram_chars.fit(text_for_feature)
        xtrain_feature =  tfidf_vect_ngram_chars.transform(train_x) 
        xvalid_feature =  tfidf_vect_ngram_chars.transform(valid_x) 
        xtest_feature =  tfidf_vect_ngram_chars.transform(test_x)
        pickle.dump(tfidf_vect_ngram_chars, open(os.getcwd() + '/Model_Output/feature_transformer.pkl', "wb"))
        
    return(xtrain_feature, xvalid_feature, xtest_feature)


# In[76]:


FEATURE_TYPE = 'Word TFIDF'
xtrain_var, xvalid_var, xtest_var = feature_creation_func(FEATURE_TYPE)


# # Model Building

# ### Shallow Neural Networks

# In[77]:


def create_model_architecture(input_size):
    # create input layer 
    input_layer = layers.Input((input_size, ), sparse=True)
    
    # create hidden layer
    hidden_layer = layers.Dense(1000, activation="relu")(input_layer)
    
    # create output layer
    output_layer = layers.Dense(mapping.shape[0], activation="softmax")(hidden_layer)

    classifier = models.Model(inputs = input_layer, outputs = output_layer)
    classifier.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
    return(classifier)


# In[78]:


#train_y_onehot_enc = pd.get_dummies(train_y_enc)
#classifier = create_model_architecture(xtrain_var.shape[1])
#classifier.fit(xtrain_var, train_y_onehot_enc, epochs=5)
#prob_predictions = classifier.predict(xvalid_var)


# ### All Models

# In[79]:


def build_model(model_name):
    #Naive Bayes Model
    if(model_name == 'Naive Bayes'):
        model = naive_bayes.MultinomialNB()
        model.fit(xtrain_var, train_y_enc)
    
    #SVM Model
    if(model_name == 'SVM'):
        model = svm.SVC(probability=True)
        model.fit(xtrain_var, train_y_enc)
    
    #Random Forest Model
    if(model_name == 'Random Forest'):
        model = ensemble.RandomForestClassifier()
        model.fit(xtrain_var, train_y_enc)
    
    #XGBoost Model
    if(model_name == 'XGBoost'):
        model = xgboost.XGBClassifier()
        model.fit(xtrain_var, train_y_enc)
    
    #Shallow Neural Networks Model
    if(model_name == 'Shallow Neural Networks'):
        train_y_onehot_enc = pd.get_dummies(train_y_enc)
        
        model = create_model_architecture(xtrain_var.shape[1])
        model.fit(xtrain_var, train_y_onehot_enc, epochs=5)
    
    return(model)


# In[80]:


MODEL_NAME = model_type
classifier = build_model(MODEL_NAME)


# In[ ]:





# In[82]:


#weight = classifier.get_weights()
pklfile= os.getcwd() + '/Model_Output/model_file.pkl'

try:
    fpkl= open(pklfile, 'wb')    #Python 3     
    pickle.dump(classifier, fpkl, protocol= pickle.HIGHEST_PROTOCOL)
    fpkl.close()
except:
    fpkl= open(pklfile, 'w')    #Python 2      
    pickle.dump(classifier, fpkl, protocol= pickle.HIGHEST_PROTOCOL)
    fpkl.close()


# In[83]:


with open(os.getcwd() + '/Model_Output/model_file.pkl', 'rb') as f:
    classifier = pickle.load(f)


# In[ ]:





# In[ ]:





# In[ ]:





# # Prediction for Accuracy

# In[84]:


def prediction_accuracy_func(model_name, x_var, measurement_df):
    if(model_name == 'Shallow Neural Networks'):
        prob_predictions = classifier.predict(x_var)
    else:
        prob_predictions = classifier.predict_proba(x_var)
        
    print("Evaluating: ", model_name)
    prob_prediction_df = pd.DataFrame(prob_predictions)
    print("prob_prediction_df", prob_prediction_df.shape)
    prob_prediction_df['contentId'] = list(measurement_df['contentId'])
    
    prob_prediction_df_long = pd.melt(prob_prediction_df, id_vars='contentId', value_vars=range(mapping.shape[0]))
    prob_prediction_df_long.columns = ['contentId', 'hashtag_encoding', 'predicted_prob']
    prob_prediction_df_long['hashtag_rank'] = prob_prediction_df_long.groupby(['contentId'])['predicted_prob'].rank(ascending=False, method = 'first')
    prob_prediction_df_long = prob_prediction_df_long.sort_values(['contentId', 'hashtag_rank'])
    prob_prediction_df_long = pd.merge(prob_prediction_df_long, mapping, how = 'left', on = 'hashtag_encoding')
    prob_prediction_df_long['pred_rank'] = 'pred_rank_' + prob_prediction_df_long['hashtag_rank'].astype(int).map(str)
    prob_prediction_df_long = prob_prediction_df_long.sort_values(['contentId', 'hashtag_rank'])
    print("prob_prediction_df_long", prob_prediction_df_long.shape)
    
    prob_prediction_df_wide = prob_prediction_df_long.pivot_table(index = ['contentId'], columns = 'pred_rank', values = ['hashtag'], aggfunc=lambda x: ''.join(x))
    prob_prediction_df_wide = prob_prediction_df_wide.reset_index()
    prob_prediction_df_wide.columns = ['%s%s' % (a, '_%s' % b if b else '') for a, b in prob_prediction_df_wide.columns]
    print("prob_prediction_df_wide", prob_prediction_df_wide.shape)
    
    prediction_results = pd.merge(measurement_df[['contentId', 'text', 'hashtag_list', 'hashtag_count']], prob_prediction_df_wide, how = 'right', on='contentId', sort=False, copy=True)
    print("prediction_results", prediction_results.shape)
    
    print("")
    
    print("Exact Match 1st Prediction:", sum(prediction_results['hashtag_list'] == prediction_results['hashtag_pred_rank_1']))
    print("Exact Match 2nd Prediction:", sum(prediction_results['hashtag_list'] == prediction_results['hashtag_pred_rank_2']))
    print("Exact Match 3rd Prediction:", sum(prediction_results['hashtag_list'] == prediction_results['hashtag_pred_rank_3']))
    print("Exact Match 4th Prediction:", sum(prediction_results['hashtag_list'] == prediction_results['hashtag_pred_rank_4']))
    print("Exact Match 5th Prediction:", sum(prediction_results['hashtag_list'] == prediction_results['hashtag_pred_rank_5']))
    
    h_list = prediction_results['hashtag_list'].str.split(',')

    top_3_pred = prediction_results[['hashtag_pred_rank_1', 'hashtag_pred_rank_2', 'hashtag_pred_rank_3']].values.tolist()
    
    prediction_results['matching_hashtag_ct'] = [len(set(a) & set(b)) for a,b in zip(h_list, top_3_pred)]
    prediction_results['matching_percent'] = (prediction_results['matching_hashtag_ct']/prediction_results['hashtag_count'])*100
    
    pred_summary = prediction_results.groupby(['matching_percent'])['contentId'].count().reset_index()
    pred_summary['content_proportion'] = (pred_summary['contentId']/sum(pred_summary['contentId']))*100
    pred_summary.sort_values(['matching_percent'], ascending=False)
    
    pred_summary_by_hash_ct = prediction_results.groupby(['matching_percent', 'hashtag_count'])['contentId'].count().reset_index()
    pred_summary_by_hash_ct.sort_values(['hashtag_count', 'matching_percent'], ascending=False)
    
    return(pred_summary, pred_summary_by_hash_ct, prediction_results)


# In[85]:


pred_summary_valid, pred_summary_by_hash_ct_valid, prediction_results_valid = prediction_accuracy_func(MODEL_NAME, xvalid_var, validation_rollup)
print("---------------------------------------------------------------------------------------")
print("Validation Data Accuracy")
print(pred_summary_valid)


# In[86]:


print(pred_summary_by_hash_ct_valid)


# In[87]:


prediction_results_valid.head()


# In[88]:


pred_summary_test, pred_summary_by_hash_ct_test, prediction_results_test = prediction_accuracy_func(MODEL_NAME, xtest_var, test_rollup)
print("---------------------------------------------------------------------------------------")
print("Test Data Accuracy")
print(pred_summary_test)


# In[89]:


print(pred_summary_by_hash_ct_test)


# In[ ]:




