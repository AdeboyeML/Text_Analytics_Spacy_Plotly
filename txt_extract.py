from collections import Counter

import spacy
nlp = spacy.load('en_core_web_lg')

import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk import ne_chunk

import os
import sys
import six
import json

from fuzzywuzzy import fuzz
from fuzzywuzzy import process

import pandas as pd
import numpy as np
import re
from itertools import chain
import random
from nltk.corpus import names
from spacy import displacy

import collections, functools, operator


class text_extract:
    
    def __init__(self, input_filepath):

        self.file = input_filepath
    """
    This class perform data cleaning on the text data, extract entities,
    remove words that likely to be the same with the Top 5 characters and also 
    count the total number of times each of the specific entities were mentioned in the
    novel
    """
        
    def clean_text(self):
        
        # read the data into a list (each row is one list element)
        with open(self.file, "r", encoding='utf-8', errors='ignore') as f:
            data = [row for row in f]
        
        #Extract the Chapters headers
        chapters = []
        for l in data:
            match = re.search(r'(Chapter.*)\n', l)
            if match:
                chapters.append(match.group(1).strip())
        chapters = [x.replace('\t', ' ').strip() for x in chapters]
        
        #Clean the Data
        text = []
        for x in data:
            tx = re.sub(r'Chapter\s+\d+', r'Chapter', x)
            tx = tx.replace('\n', ' ')
            tx = tx.replace('\t', ' ')
            tx = re.sub("(^|\s)(\S)([A-Z]+)", lambda m: m.group(1) + m.group(2).upper() + m.group(3).lower(), tx)
            tx = re.sub(r'(dr\.|Dr\.)', r'Dr', tx)
            tx = re.sub(r'(Mrs\.|mrs\.)', r'Mrs', tx)
            tx = re.sub(r'(Mr\.|mr\.)', r'Mr', tx)
            tx = re.sub(r'(p\.m\.|P\.M\.)', r'PM', tx)
            tx = re.sub(r'(A\.M\.|a\.m\.)', r'AM', tx)
            tx = re.sub(r'(\'s|\'S|\’s|\’S)', r' ', tx)
            tx = re.sub(r'(\'m|\’m)', r' am', tx)
            tx = re.sub(r'(\'ll|\’l)', r' will', tx)
            tx = re.sub(r'(\'re|\’re)', r' are', tx)
            tx = re.sub(r'(\'d|\’d)', r' had', tx)
            tx = re.sub(r'(\'ve|\’ve)', r' have', tx)
            tx = re.sub(r'(\'t|\’t)', r't', tx)
            tx = re.sub(r"\'", r' ', tx)
            tx = re.sub(r'[^a-zA-Z0-9.,:?" ]+', '', tx)
            text.append(tx)
        
        text = "".join([s for s in text if s.strip()])
        re_data = text.split('Chapter')
        #print(len(re_data))
        re_data = re_data[1:]
        
        #chapters into dataframes
        data_tuples = list(zip(chapters,re_data))
        df_data = pd.DataFrame(data_tuples, columns=['Chapters','Contents'])
        
        #Extract sentence based on period(.) --> This is not the best approach but it gives consistent results
        sent_data = re.sub(r'[^a-zA-Z0-9.,:?" ]+', '', str(re_data))
        sent_data = sent_data.split('.')
        
        sent_names = [('sentence ' + str(x + 1)) for x in range(0, len(sent_data), 1)]
        data_tuples2 = list(zip(sent_names,sent_data))
        df_sent = pd.DataFrame(data_tuples2, columns=['names','Contents'])
        return re_data, df_data, df_sent
    
    #Here, we are only extracting the TIME, GEOPOLITICAL LOCATIONS AND PERSON Entities
    
    def extract_entities(self, df1, df2):
        
        def NER (df):
            
            person = []
            time = []
            gpe = []
            
            def most_common(entities):
                most_com = dict(Counter(entities).most_common())
                return most_com
            
            for x in range(0, len(df), 1):
                chpt = df.Contents[x]
                parse_chpt = nlp(chpt)
                
                persons_chpt = []
                gpe_chpt = []
                time_chpt = []
                
                for entity in (parse_chpt.ents):
                    if entity.label_ is 'PERSON':
                        persons_chpt.append(str(entity))
                    elif entity.label_ is 'GPE':
                        gpe_chpt.append(str(entity))
                    elif entity.label_ is 'TIME':
                        time_chpt.append(str(entity))
                        
                person.append(most_common(persons_chpt))
                gpe.append(most_common(gpe_chpt))
                time.append(most_common(time_chpt))
                
            return person, gpe, time
        
        chpt_person, chpt_gpe, chpt_time = NER(df1)
        sent_person, sent_gpe, sent_time = NER(df2)
        
        #assuming persons names and geopolitical places in the novel are starting with Capital letters not small letters
        def clean_proper(person):
            for x in range(0, len(person), 1):
                for keys in list(person[x].keys()):
                    values = person[x].pop(keys)
                    new_key = re.sub(r'^([a-z].*)|([A-Z]+[a-z]+\s\d.*)|(\d.*)', '', keys).lstrip(" ")
                    person[x][new_key] = values
            return person
        
        #clean the names and gpe
        sent_person = clean_proper(sent_person)
        chpt_gpe = clean_proper(chpt_gpe)
        
        # sum the values with same keys and extract entities greater than one
        def sum_ents(entity):
            result = dict(functools.reduce(operator.add, map(collections.Counter, entity)))
            sorted_result = sorted(result.items(), key=lambda x: x[1], reverse = True)
            result_ = [i  for i in sorted_result if i[1] > 1 and i[0] != '']
            return result_
        
        sum_person = sum_ents(sent_person)
        sum_gpe = sum_ents(chpt_gpe)
        sum_time = sum_ents(chpt_time)
        
        return sum_person, sum_gpe, sum_time
    
    #This function is primarily for removing words that are similar to the top 5 characters in the novel
    #I also remove characters that are misclassified as GPE from the gpe
    def remove_similar(self, characters, gpes):
        
        chters = [i[0] for i in characters]
        refined_gpe = [x[0] for x in gpes if x[0] not in chters]
        ct = 2
        
        for z in range(0, ct):
            #print(z)
            if z == 1:
                top_ = chters[:4]
            else:
                top_ = chters[:2+z]
            
            similar_xters = []
            refined_xters = []
            count = 0
            for x in range(0, len(chters), 1):
                for y in chters[x:]:
                    if (fuzz.partial_ratio(chters[x], y)) >= 90:
                        similar_xters.append(y)
                        #print(y)
                count += 1
                if count == 2+z:
                    break
            top_.extend([x for x in chters if x not in similar_xters])
            for x in top_:
                if x not in refined_xters:
                    refined_xters.append(x)
            chters = refined_xters
        return chters, refined_gpe
    
    #Count the amount of times entities appeared in the whole Novel
    #Check how many chapters and sentences entities appear in 
    def count(self, df, entity):
        ent_ = []
        for x in range(0,len(df),1):
            kk = re.compile("({})+".format("|".join(re.escape(c) for c in entity)))
            xters = kk.findall(df.Contents[x])
            ent_.extend(xters)
        ent_cts = dict(Counter(ent_).most_common())
        
        #count the chapters the xters appear in
        xtrs = [keys for keys in ent_cts]
        #print(xtrs)
        per_chpt_sent = {}
        for word in xtrs:
            count = 0
            for x in range(0, len(df), 1):
                if word in df.Contents[x]:
                    count += 1
                    per_chpt_sent[word] = count
        
        return ent_cts, per_chpt_sent