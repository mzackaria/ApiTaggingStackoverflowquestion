# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 13:34:54 2018

@author: zakis
"""
from .get_result import lr, vectorizer, multilabel_binarizer
from flask import Flask, jsonify, request
import numpy as np
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords    
from nltk.stem.porter import PorterStemmer 

app = Flask(__name__)
    
@app.route('/predict_tags', methods = ['GET'])
def index():
    #get the question from the request
    text = request.args.get('question')
    
    #clean the text
    review_cleaned = review_to_words(text)
    
    #vectorize the cleaned text
    review_vectorized = vectorizer.transform([review_cleaned]).toarray()
    
    #predict tags
    y_pred = lr.predict(review_vectorized).toarray()
    
    #add a 0 for the nan value
    x = np.insert(y_pred, 29, 0)
    
    #get the tags in text form
    tags_pred = multilabel_binarizer.inverse_transform(np.array(x).reshape(1,51))
    
    #reshape the array for response
    tags_pred = tags_pred[0]
    result = ''
    for tag in tags_pred:
        result += tag + ', '
    if len(result) > 0:
        result = result[:len(result)-2]
    return  jsonify({"question":text, "tags":result})


@app.route('/')
def ind():
        return "hello"
    
    
#defining the funtion that will be used to create the dictionnary
def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()      
    #
    # 4. Stem all the words
    ps = PorterStemmer()
    words = [ps.stem(word) for word in words]                       
    #
    # 5. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 6. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 7. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words )) 
    
if __name__ == "__main__":
        app.run()