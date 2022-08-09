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

import re
import demoji



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




# Function to preprocess singular posts descriptions
def preprocess_text(text):
    
    # Convert to lower case
    clean_text = text.lower()
    
    # Removing all @mentions
    clean_text = re.sub(r"@[a-z0-9_]+","", clean_text)
    
    # Removing all #hashtags
    clean_text = re.sub(r"#[^\s]+","", clean_text)
    
    # Removing all emojis
    clean_text = demoji.replace(clean_text)
    
    
    # Replacing unicode fractions with ascii representations
    fractions = {
        '\x215b': '1/8',
        '\x215c': '3/8',
        '\x215d': '5/8',
        '\x215e': '7/8',
        '\x2159': '1/6',
        '\x215a': '5/6',
        '\x2155': '1/5',
        '\x2156': '2/5',
        '\x2157': '3/5',
        '\x2158': '4/5',
        '\xbc': '1/4',
        '\xbe': '3/4',
        '\x2153': '1/3',
        '\x2154': '2/3',
        '\xbd': '1/2'
        }
    for f_unicode, f_ascii in fractions.items():
        clean_text = clean_text.replace(f_unicode, f_ascii)
        
        
    # Add whitespace between quantities and measurements, e.g. "200ml milk" -> "200 ml milk"
    clean_text = re.sub(r"(\d+(.\d+)?)",r"\1 ", clean_text)
    
    # Removing unwanted double whitespaces
    while '  ' in clean_text:
        clean_text = clean_text.replace('  ', ' ')
    
    # Remove leading and trailing whitespaces
    clean_text = clean_text.strip()
    
    
    # Standardise measurements
    measurements = {
        "cups": "cup",
        "tablespoons": "tablespoon",
        "tbsp": "tablespoon",
        "tbsps": "tablespoon",
        "teaspoons": "teaspoon",
        "tsp": "teaspoon",
        "tsps": "teaspoon",
        "pounds": "pound",
        " lb ": " pound ",
        " lbs ": " pound ",
        "ounces": "ounce",
        " oz ": " ounce ",
        "cloves": "clove",
        "sprigs": "sprig",
        "pinches": "pinch",
        "bunches": "bunch",
        "slices": "slice",
        "grams": "gram",
        " g ": " gram ",
        " gs ": " gram ",
        " gm ": " gram ",
        " gms ": " gram ",
        " grm ": " gram ",
        " grms ": " gram ",
        "litres": "litre",
        " cls ": " cl ",
        "centilitres": "cl",
        "centiliters": "cl",
        " mls ": " ml ",
        "millilitres": "ml",
        "milliliters": "ml",
        "heads": "head",
        "quarts": "quart",
        "stalks": "stalk",
        "pints": "pint",
        "pieces": "piece",
        "sticks": "stick",
        "dashes": "dash",
        "fillets": "fillet",
        "cans": "can",
        "ears": "ear",
        "packages": "package",
        "strips": "strip",
        "bulbs": "bulb",
        "bottles": "bottle"
        }
    for measure_original, measure_standardised in measurements.items():
        clean_text = clean_text.replace(measure_original, measure_standardised)
    
    return clean_text
