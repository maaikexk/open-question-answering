# Import needed libraries
import warnings
import logging
import gensim
import pickle
from nltk.stem import WordNetLemmatizer
from alive_progress import alive_bar
from nltk.tokenize import RegexpTokenizer
import spacy

warnings.filterwarnings(action='ignore')


# Preprocess the training data needed for the wordembedding. Training on a dump of wiki sentences.
def preprocess_training_data(train_data='Data/wikisent2.txt', save_as='Data/processed_sent_embeddings.txt'):
    # Initializing tools needed
    lemmatizer = WordNetLemmatizer()
    tokenizer = RegexpTokenizer(r'\w+')
    NER = spacy.load("en_core_web_sm")

    # Getting the data from file and splitting txt into list
    file = open(train_data)
    contents = file.read()
    data = contents.split('\n')
    data = data[:3000000]

    # Empty list to save tokens
    tokens = []

    # The load bar indicating the eta of the data
    with alive_bar(len(data), title='Tokenization', force_tty=True, dual_line=True, length=20) as bar:
        for sentence in data:

            # Empty list used for temporary tokens
            temp_tokens = []

            # Updating the load bar
            bar()

            # Tokenize the sentences
            for token in tokenizer.tokenize(sentence):
                temp_tokens.append(lemmatizer.lemmatize(token.lower()))

            # Add tokens to list
            tokens.append(temp_tokens)

    # Save training data for later use
    with open(save_as, 'wb') as f:
        pickle.dump(tokens, f)


# Load the preprocessed embeddings
def load_preprocessed_data(file_name='Data/processed_sent_embeddings.txt'):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data


# Train the word embedding with gensim and save it for future use
def train_word_embeddings(train_data, save_as="model_cbow.model"):
    logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt='%H:%M:%S', level=logging.INFO)
    model_cbow = gensim.models.Word2Vec(train_data, min_count=1, vector_size=100, window=5)
    model_cbow.save(save_as)
    print("Embeddings trained")


# Train the bigram model with gensim and save it for future use
def train_bigrams(train_data, save_as="model_bigrams", save_bigrams="bigrams"):
    phrase_model = gensim.models.phrases.Phrases(train_data, min_count=1, threshold=1, connector_words=gensim.models.phrases.ENGLISH_CONNECTOR_WORDS)
    phrase_model.save(save_as)
    print("Bigram model trained")


# Load and return saved bigram model
def load_bigram_model(model="model_bigrams"):
    return gensim.models.phrases.Phrases.load(model)


# Apply bigram model to data and return data
def retrieve_bigrams(data, model):
    return [model[sentence] for sentence in data]


# Load and return saved embeddings model
def load_word_embeddings(model="model_cbow.model"):
    return gensim.models.Word2Vec.load(model)


# Score similarity between question and all potential answers
def score_word_similarity(question_tags, passages, embeddings_model, threshold=0.6):

    # Get length of question
    number_tags = len(question_tags)

    # Empty list to save probabilities for each passage
    probabilities = []
    best_sentences = []

    # For every list of tokens in a potential answer
    for passage, sentences in zip(passages['Tokens V2'], passages['sentences']):
        score = 0

        # Best sentence score
        best_score_sentence = 0
        best_sentence = ""
        best_paragraph_tokens = []

        # For every token in a sentence
        for tokens, sentence in zip(passage, sentences):
            sentence_score = 0

            best_sentence_tokens = []

            # For every tag in the question
            for index, row in question_tags.iterrows():
                highest_similarity = 0
                best_token = ""

                # For every token in a potential answer
                for token in tokens:

                    # Check if the token of the question and passage occur in the embeddings model
                    if token in embeddings_model.wv and row['tags'] in embeddings_model.wv:

                        # Score similarity
                        similarity = embeddings_model.wv.similarity(row['tags'], token)

                        # Save the highest similarity for a question token
                        if similarity > threshold and similarity > highest_similarity:
                            highest_similarity = similarity
                            best_token = token

                sentence_score += highest_similarity
                best_sentence_tokens.append(best_token)

            if sentence_score > best_score_sentence:
                best_score_sentence = sentence_score
                best_paragraph_tokens = best_sentence_tokens
                best_sentence = sentence

        # Add the highest similarity to the score
        score += best_score_sentence

        # Add the probabilities of passages to the list; probability is weighted score over number of tokens
        probabilities.append(score / number_tags) if number_tags != 0 else probabilities.append(0)
        best_sentences.append([best_sentence, best_paragraph_tokens])

    # Add the probabilities to the panda and return the passages with score
    passages['Score'] = probabilities
    passages['Best sentence'] = best_sentences
    return passages


def context_score(question_tags, passages, embeddings_model, threshold=0.6):

    tot_context_score = []

    for passage in passages['Tokens V2']:
        num_of_tokens = sum([len(tokens) for tokens in passage])
        passage_context_score = 0
        for tokens in passage:
            sentence_score = 0

            for token in tokens:
                token_score = 0

                for index, row in question_tags.iterrows():

                    if token in embeddings_model.wv and row['tags'] in embeddings_model.wv:
                        similarity = embeddings_model.wv.similarity(row['tags'], token)

                        if similarity > token_score:
                            token_score = similarity

                sentence_score += token_score
            passage_context_score += sentence_score
        tot_context_score.append(passage_context_score/num_of_tokens)
    passages['context'] = tot_context_score
    return passages


