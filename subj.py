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

import pandas as pd
import numpy as np
import re

class subjects:
    
    def __init__(self, text, characters):
        
        self.text = text
        self.characters = characters
        
    """
    This class basically extract sentences where major characters are the
    Subjects of the sentences with the corresponding "action verbs"
    """
        
    def subject_action(self):
        
        txt = re.sub(r'[^a-zA-Z0-9,.? ]+', '', str(self.text))
        x_sent = []
        parse_txt = nlp(txt)
        for x in parse_txt.sents:
            #print(x)
            x_sent.append(str(x.text))
            
        x_names = [('sentence ' + str(x + 1)) for x in range(0, len(x_sent), 1)]
        data = list(zip(x_names,x_sent))
        df = pd.DataFrame(data, columns=['names','Contents'])
        
        def token_is_subject_with_action(token):
            nsubj = token.dep_ == 'nsubj'
            head_verb = token.head.pos_ == 'VERB'
            person = token.ent_type_ == 'PERSON'
            return nsubj and head_verb and person
        
        spans = []
        for x in range(0, len(df), 1):
            clean_sentence = re.sub(r'[^a-zA-Z0-9 ]', '', str(df.Contents[x])).strip()
            doc = nlp(clean_sentence)
            for z in doc:
                if str(z) in self.characters:
                    if token_is_subject_with_action(z):
                        #print(doc[z.head.left_edge.i:z.head.right_edge.i+1])
                        sp = doc[z.head.left_edge.i:z.head.right_edge.i+1]
                        objts = ""
                        for toks in doc:
                            if toks.dep_ in ['pobj', 'dobj']:
                                objts += " " +  ''.join(str(toks))
                        data = dict(Subject=z.orth_, Action_Sentence=sp.text,
                                    Action_verb =z.head.lemma_, 
                                    log_prob=z.head.prob, objects=objts, Sentence_Number = x)
                        spans.append(data)
                        
        df_span = pd.DataFrame(spans)
        
        return spans, df_span

