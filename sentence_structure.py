import spacy
from nltk.stem import WordNetLemmatizer
from functools import reduce
from itertools import chain
from nltk import Tree
en_nlp = spacy.load('en_core_web_sm')


def get_sentence_dependencies(sentence):
    nlp_sentence = en_nlp(sentence)
    lemmatizer = WordNetLemmatizer()
    return [[lemmatizer.lemmatize(str(token)).lower(), token.dep_] for token in nlp_sentence if token.dep_ != 'punct']


def match_dependencies(tokens, dependencies):

    token_dependencies = [token.split('_') if token.count('_') != 0 else [token] for token in tokens]
    for token_pairs in range(len(token_dependencies)):
        for token in range(len(token_dependencies[token_pairs])):
            for dependency_pair in dependencies:
                if token_dependencies[token_pairs][token] == dependency_pair[0]:
                    token_dependencies[token_pairs][token] = (token_dependencies[token_pairs][token], dependency_pair[1])

    return token_dependencies


def compare_dependencies(question_dependencies, answer_dependencies):
    score = 0
    for question_dependency, answer_dependency in zip(question_dependencies, answer_dependencies):
        if question_dependency[-1] != '' and answer_dependency[-1] != '':
            if question_dependency[-1][1] == answer_dependency[-1][1]:
                score += 1

    score = score/len(question_dependencies)
    return score


def score_word_similarity(question_sentence, question_tags, passages):

    question_dependencies = get_sentence_dependencies(question_sentence)

    question_tags = match_dependencies(question_tags['tags'], question_dependencies)
    passage_tags = [match_dependencies(passage[1], get_sentence_dependencies(passage[0])) for passage in passages['Best sentence']]

    passages['structure_score'] = [compare_dependencies(question_tags, passage_tags_sentence) for passage_tags_sentence in passage_tags]

    return passages
