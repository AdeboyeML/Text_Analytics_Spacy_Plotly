import spacy
nlp = spacy.load('en_core_web_lg')

import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk import ne_chunk

import random
from nltk.corpus import names
from spacy import displacy

import collections, functools, operator

import warnings
warnings.filterwarnings("ignore")

import plotly.graph_objects as go
import pandas as pd
import numpy as np
import re

class gender_distribution:
    
    def __init__(self, characters):
        
        self.characters = characters
        
    """
    This class checks for the gender distribution in the Novel
    i.e. Total number of observed female and male characters
    """
        
    def gender_types(self):
        
        #Gender identification
        nltk.download('names')
        labeled_names = ([(name, 'male') for name in names.words('male.txt')] + \
                         [(name, 'female') for name in names.words('female.txt')])
        random.shuffle(labeled_names)
        
        def gender_features(word):
            return {'suffix1': word[-1:].lower(), 'suffix2': word[-2:].lower(),
                    "first_letter" : word[0].lower(),"last_letter" : word[-1].lower()}
        
        train_names = labeled_names[500:]
        test_names = labeled_names[:500]
        train_set = [(gender_features(n), gender) for (n, gender) in train_names]
        test_set = [(gender_features(n), gender) for (n, gender) in test_names]
        classifier = nltk.NaiveBayesClassifier.train(train_set)
        
        def gender_id(major_characters):
            
            female = {}
            male = {}
            for word in major_characters:
                gender = classifier.classify(gender_features(word))
                if gender == 'male':
                    male[word] = gender
                else:
                    female[word] = gender
            return female, male
        
        female_xters, male_xters = gender_id(self.characters)
        female_xters = list(female_xters.keys())
        male_xters = list(male_xters.keys())
        
        gender_dict = {}
        gender_dict['Male'] = len(male_xters)
        gender_dict['Female'] = len(female_xters)
        df_gend = pd.DataFrame(gender_dict.items(), columns = ['gender', 'size'])
        
        return df_gend