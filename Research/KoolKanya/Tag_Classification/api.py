
import os
import warnings
import pandas as pd
import numpy as np
from datetime import timedelta
import pickle

import nltk

#import nltk
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

#from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
#from sklearn import decomposition, ensemble
#from sklearn.model_selection import train_test_split

import xgboost

#from keras.preprocessing import text, sequence
#from keras import layers, models, optimizers
#from tensorflow import keras

nltk.download(['punkt', 'stopwords'])

from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
#from model import NLPModel

app = Flask(__name__)
api = Api(app)



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

tfidf_vectorizer, classifier, encoder_mapping = load_objects_func()

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')


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

class Get_Hashtag(Resource):
    def get(self):
        #tfidf_vectorizer, classifier, encoder_mapping = load_objects_func()
        # use parser and find the user's query
        args = parser.parse_args()
        input_txt = args['query']
        print(input_txt)

        clean_text = get_clean_text(input_txt)
        print(clean_text)
    
        #Generate Features for scoring
        feature_vector = tfidf_vectorizer.transform(clean_text)
    
        #Generate Predictions
        prob_predictions = classifier.predict_proba(feature_vector)
        #print(prob_predictions)
    
        prediction_df = encoder_mapping.copy()
        prediction_df = prediction_df.sort_values(['hashtag_encoding'])
        prediction_df['probability'] = prob_predictions.tolist()[0]
        prediction_df = prediction_df.sort_values(['probability'], ascending=False)
        #print(prediction_df)
        topN_hashtag = prediction_df[0:3]['hashtag'].to_list()
        return(topN_hashtag)

        

# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(Get_Hashtag, '/')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=False)
