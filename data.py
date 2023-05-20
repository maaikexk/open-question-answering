# Import needed libraries
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import spacy


# Function to read the pdf file. If file name not specified it will read a predetermined excel file
def get_data(file_name='Data/eCG_sport_subject_v2.xlsx'):
    return pd.read_excel(file_name)


# Preprocess the data and putting it in a pandas format
def preprocess(data, bigram_model):
    # Initializing functions needed
    lemmatizer = WordNetLemmatizer()
    NER = spacy.load("en_core_web_sm")

    data['sentences'] = [passage.split('. ') for passage in data['Passage']]

    # Translate everything into lowercase
    data['Lower'] = [passage.lower() for passage in data['Passage']]

    # Split passage into sentences
    data['Lower'] = [sentence.split('. ') for sentence in data['Lower']]

    tokens = []
    tokens_v2 = []
    for passage in data['Lower']:
        # Tokenization
        tokenizer = RegexpTokenizer(r'\w+')
        tokens_passage = [tokenizer.tokenize(sentence) for sentence in passage]
        tokens.append(tokens_passage)

        # Making a list of lemmatized tokens not including stopwords
        tokens_no_stop_sentence = []
        for tokens_sentence in tokens_passage:
            # print(tokens_sentence)
            temp_tokens = [lemmatizer.lemmatize(token) for token in tokens_sentence if token not in stopwords.words('english')]
            temp_tokens = bigram_model[temp_tokens]
            tokens_no_stop_sentence.append(temp_tokens)
        tokens_v2.append(tokens_no_stop_sentence)

    data['Tokens'] = tokens
    data['Tokens V2'] = tokens_v2

    # Return the pandas
    return data
