# Import all functions of programme
import urllib.parse
import datetime
import pytz
import passage_retrieval
import question
import data
import sentence_structure
import wordembedding
from googletrans import Translator
from http.server import BaseHTTPRequestHandler, HTTPServer


# The entire process of going from question to answer
def get_response(user_input, passages, embeddings_model, bigram_model):
    translator = Translator(service_urls=[
        'translate.google.com',
    ])
    user_input = translator.translate(user_input, src='nl').text
    print(user_input)

    # Translate the input question into preprocessed tags
    tags = question.preprocess(user_input, bigram_model)

    # Score the possible answers and retrieve the answer with the highest score
    # passages = passage_retrieval.passage_score(tags, passages, embeddings_model)
    passages = wordembedding.score_word_similarity(tags, passages, embeddings_model)
    passages = sentence_structure.score_word_similarity(user_input, tags, passages)
    passages = wordembedding.context_score(tags, passages, embeddings_model)
    passage = passage_retrieval.best_passage(passages)

    return passage


def main():

    # Change training mode if model needs to be retrained
    training_mode = False

    if training_mode:
        # Preprocess and train wordembedding data
        wordembedding.preprocess_training_data()
        train_data = wordembedding.load_preprocessed_data()
        wordembedding.train_bigrams(train_data)

        # Load bigram model and retrieve bigrams
        print("loading bigram model")
        bigram_model = wordembedding.load_bigram_model()
        bigrams = wordembedding.retrieve_bigrams(train_data, bigram_model)

        # Train the word embeddings and train the model. Comment this out if model is trained and load the model instead
        wordembedding.train_word_embeddings(bigrams)
        print("loading word embedding")
        word_embeddings = wordembedding.load_word_embeddings()
    else:
        # Retrieve bigram model
        print("loading bigram model")
        bigram_model = wordembedding.load_bigram_model()

        # Load word embeddings
        print("loading word embedding")
        word_embeddings = wordembedding.load_word_embeddings()

    # Get the passages data and pre-process this
    print("load data source")
    passages = data.get_data()
    passages = data.preprocess(passages, bigram_model)

    print("Ready to rumble!!")

    return passages, word_embeddings, bigram_model


# Call the main function
passages, word_embeddings, bigram_model = main()


class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        if parsed_path.path == '/ecg':
            request_question = urllib.parse.parse_qs(parsed_path.query).get('question', [''])[0]

            response = get_response(request_question, passages, word_embeddings, bigram_model)

            with open('Data/log.txt', 'a') as f:
                f.write('Question: ' + request_question)
                f.write('\n')
                f.write('Timestamp: ' + str(datetime.datetime.now(pytz.timezone('Europe/Amsterdam'))))
                f.write('\n')
                f.write(response)
                f.write('\n')
                f.write('\n')

            self.send_response(200)
            self.send_header('Content-type', 'text/json')
            self.end_headers()
            self.wfile.write(response.encode('utf-8'))
        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/json')
            self.end_headers()
            self.wfile.write('{ error: "not found" }'.encode('utf-8'))


def run(server_class=HTTPServer, handler_class=SimpleHTTPRequestHandler):
    server_address = ('', 8000)
    httpd = server_class(server_address, handler_class)
    print('Starting httpd...')
    httpd.serve_forever()


if __name__ == '__main__':
    run()
