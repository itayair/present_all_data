import spacy
import itertools
from src.question_to_query.POSTree import POSTree as POSTree
from spacy import displacy
import os

java_path = "C:/Program Files/Java/jdk-19/bin/java.exe"
os.environ['JAVAHOME'] = java_path
from nltk.parse import stanford

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'stanford-parser-full-2020-11-17/stanford-parser.jar')
os.environ['STANFORD_PARSER'] = filename
filename = os.path.join(dirname, 'stanford-parser-full-2020-11-17/stanford-parser-4.2.0-models.jar')
os.environ['STANFORD_MODELS'] = filename
filename = os.path.join(dirname, 'stanford-parser-full-2020-11-17/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz')
parser = stanford.StanfordParser(model_path=filename)


class tokenQuery:
    def __init__(self, token, rests, query_id):
        self.id = query_id
        self.token = token
        self.rests = rests
        self.dep = None
        self.children = []

    def addDep(self, dep):
        self.dep = dep


nlp = spacy.load("en_core_web_sm")

dictionary_question_to_query = {}
dictionary_question = {}
kind_of_question = 0
lst_wh_question = ["what", "why", "where", "when", "who", "how"]
entity_types = ["something", "somewhere", "someone"]


def get_root_index(query):
    query = query.replace("**blank**", "something")
    question_tokenized = nlp(query)
    idx = 0
    for token in question_tokenized:
        if token.head == token:
            return idx
        idx += 1


def conversion_question_to_statement(question):
    dictionary_question[question] = dictionary_question.get(question, 0) + 1
    if question in dictionary_question_to_query:
        query, index_of_answer = dictionary_question_to_query[question]
        return (query, index_of_answer)
    parse_tree = parser.raw_parse(question)
    parse_tree_str = ""
    for iter in parse_tree:
        parse_tree_str += str(iter)
    tree = POSTree(parse_tree_str)
    try:
        query = tree.adjust_order()
    except:
        print("error question:", question)
        return None
    return query


def getLastToken(query_dep):
    last_token = None
    for token in query_dep:
        last_token = token
    return last_token


def getFirstToken(query_dep):
    for token in query_dep:
        return token


def getTokenOfAnswer(query_dep, index_answer):
    idx = 0
    for token in query_dep:
        if idx == index_answer:
            return token
        idx += 1
    return None


def getTokenIndex(answer_token, sent_captured_tokens):
    idx = 0
    for token in sent_captured_tokens:
        if token.token == answer_token:
            return idx
        idx += 1
    return -1


def getSentenceAnswer(sent, answer_token):
    sent_captured_tokens = sent[0]
    query_captured_tokens = sent[1]
    result_index = getTokenIndex(answer_token, query_captured_tokens)
    if result_index == -1:
        return None
    return sent_captured_tokens[result_index]


def createQuestion(quest):
    new_quest = ""
    for word in quest:
        if word == '_':
            continue
        new_quest += word
        if word != '?':
            new_quest += ' '
    return new_quest


def isValid(token_query, token_sent):
    if token_query.dep:
        if token_sent.dep_ != token_query.dep[1]:
            return False
    if token_query.rests == 'l':
        if token_query.token.lemma_ == token_sent.lemma_:
            return True
        else:
            return False
    elif token_query.rests == 'w':
        if str(token_query.token) == str(token_sent):
            return True
        else:
            return False
    elif not token_query.rests:
        return True
    else:
        if token_query.rests == token_sent.ent_type_:
            return True
        else:
            return False
