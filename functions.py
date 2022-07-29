"""
FUNCTIONS FILE CONTAINNG ALL FUNCTIONS TO BE USED WITHIN THIS PROJECT
"""

# MODULES
# -------

import numpy as np
import pandas as pd
from datetime import datetime

import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector



# FUNCTIONS
# ---------


# Creates a language detection SpaCy pipeline
def get_lang_detect_nlp_pipe():

    @Language.factory("language_detector")
    def get_lang_detector(nlp, name):
        return LanguageDetector()

    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe('language_detector', last=True)

    return nlp


# Detects language from post, returning language and confidence score
def detect_language(post, nlp=get_lang_detect_nlp_pipe()):
    lang_dict = nlp(str(post))._.language
    return pd.Series([lang_dict['language'], lang_dict['score']])



# Takes PhantomBuster df as input, removes all non english posts
def get_english_posts(df, language='en', confidence=0.95):
    
    print("Detecting language of each post...")
    now = datetime.now()
    
    df[['language', 'score']] = df['description'].apply(detect_language)
    
    print(f"Language detection complete.\nTime taken: {datetime.now()-now}")
    
    return df[(df['score']>0.95) & (df['language']=='en')]