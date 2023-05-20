# Import libraries needed
import itertools
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import pandas as pd
from nltk.stem import WordNetLemmatizer
import spacy

# Uncomment if needed
# nltk.download('omw-1.4')
# nltk.download('stopwords')


# Get wordnet similarities and score them based upon their similarity
def get_similarities(tags, score):
    synonyms = []

    # For each word in the set of tags
    for tag in tags:
        syn_set = []

        # For each synonym check if it is similar enough and at it to the set of synonyms of that tag.
        for syn in wordnet.synsets(tag):
            for lem in syn.lemmas():
                similarity = wordnet.synsets(tag)[0].wup_similarity(wordnet.synsets(lem.name())[0])
                if similarity >= score:
                    syn_set.append([lem.name(), similarity])

        # Remove duplicates and the tag itself.
        syn_set.sort()
        syn_set = list(l for l, syn in itertools.groupby(syn_set))
        if [tag, 1] in syn_set: syn_set.remove([tag, 1])

        # Check for manual synonyms
        if tag == 'hcm':
            synonyms.append([['hypertrophic', 1.0], ['cardiomyopathy', 1.0]])
        else:
            # Make into one list and return
            synonyms.append(syn_set)

    # Add tags and synonyms to the dataframe
    tags = pd.DataFrame(list(zip(tags, synonyms)), columns=['tags', 'synonyms'])
    return tags


# Preprocessing the question
def preprocess(question, bigram_model, score=0.7):

    # Lowercase and tokenizer
    question = question.lower()

    # Tokenization
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(question)

    # Translate into lemmatized token list without stopwords.
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Get the known bigrams
    tokens = bigram_model[tokens]

    # Get similarities
    tags = get_similarities(tokens, score)

    return tags
