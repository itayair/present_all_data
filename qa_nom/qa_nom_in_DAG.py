# import DAG.hierarchical_structure_algorithms as hierarchical_structure_algorithms
import DAG.NounPhraseObject as NounPhrase
import DAG.DAG_utils as DAG_utils
from combine_spans import utils as combine_spans_utils
import pickle
from qasem.end_to_end_pipeline import QASemEndToEndPipeline
import sys
from question_to_query import utils as question_to_query_utils
from topic_clustering import utils_clustering as utils_clustering
import spacy
import torch

sys.setrecursionlimit(10000)

device = 2 if torch.cuda.is_available() else -1
# pipe = QASemEndToEndPipeline(qasrl_model="biu-nlp/qanom-seq2seq-model-joint",
#                              qanom_model="biu-nlp/qanom-seq2seq-model-joint", annotation_layers=('qanom'),
#                              nominalization_detection_threshold=0.75, contextualize=True,
#                              device=device)
pipe = QASemEndToEndPipeline(annotation_layers=('qanom'),
                             nominalization_detection_threshold=0.7, contextualize=True,
                             device=device)


class qanom_question:
    def __init__(self, question_data):
        self.question = question_data['question']
        self.answers = question_data['answers']
        self.role = question_data['question-role']
        self.contextual_question = question_data['contextual_question']


class qanom_data:
    def __init__(self, question_data_lst, predicate, verb_form):
        self.question_data_lst = question_data_lst
        self.predicate = predicate
        self.verb_form = verb_form


sentences = ["Injection of streptozotocin",
             "Injection of insulin"]


def get_qanom_objects_for_span_lst(span_lst):
    with torch.no_grad():
        outputs = pipe(span_lst)
    span_to_qanom_object_lst = {}
    for span_data, span in zip(outputs, span_lst):
        span_data_qanom_object_lst = []
        qanom_sentence_data = span_data['qanom']
        for qanom_span_predicate_data in qanom_sentence_data:
            qa_data_lst = []
            for question_data in qanom_span_predicate_data['QAs']:
                qanom_question_object = qanom_question(question_data)
                qa_data_lst.append(qanom_question_object)
            predicate = qanom_span_predicate_data['predicate']
            verb_form = qanom_span_predicate_data['verb_form']
            qanom_data_object = qanom_data(qa_data_lst, predicate, verb_form)
            span_data_qanom_object_lst.append(qanom_data_object)
        if span_data_qanom_object_lst:
            span_to_qanom_object_lst[span] = span_data_qanom_object_lst
    return span_to_qanom_object_lst


def get_key_with_max_value(dic):
    maximum = 0
    max_key = None
    for key, value in dic.items():
        if value > maximum:
            max_key = key
            maximum = value
    return max_key


def get_predicate_to_question_and_question_to_nodes_nested_dictionary(span_to_qanom_object_lst, dict_span_to_object,
                                                                      pred_to_que_nodes_dict, topic,
                                                                      pred_to_object_dict):
    dict_predicate_to_occurrences = {}
    question_to_node_dict = {}
    for span, qanom_object_lst in span_to_qanom_object_lst.items():
        span_object = dict_span_to_object[span]
        for qanom_object in qanom_object_lst:
            predicate = combine_spans_utils.dict_word_to_lemma.get(qanom_object.predicate, qanom_object.predicate)
            predicate_in_topic = False
            if qanom_object.predicate in topic.span_lst:
                predicate_in_topic = True
            if not predicate_in_topic:
                continue
            dict_predicate_to_occurrences[predicate] = dict_predicate_to_occurrences.get(predicate, 0)
            dict_predicate_to_occurrences[predicate] += 1
            for qanom_question_object in qanom_object.question_data_lst:
                question_to_node_dict[qanom_question_object.contextual_question] = \
                    question_to_node_dict.get(qanom_question_object.contextual_question, [])
                question_to_node_dict[qanom_question_object.contextual_question].append(
                    (span_object, qanom_question_object))
    max_predicate = get_key_with_max_value(dict_predicate_to_occurrences)
    if max_predicate:
        pred_to_que_nodes_dict[max_predicate] = question_to_node_dict
        pred_to_object_dict[max_predicate] = topic


def get_potential_qa_nom(topic_objects):
    span_lst = set()
    span_to_topic_object = {}
    visited = set()
    for topic in topic_objects:
        if len(topic.children) < 2:
            continue
        num_of_span = max(min(0.2 * len(topic.children), 5), 2)
        for child in topic.children[:int(num_of_span)]:
            most_frequent_span = combine_spans_utils.get_most_frequent_span(child.span_lst)
            span_to_topic_object[most_frequent_span] = span_to_topic_object.get(most_frequent_span, set())
            span_to_topic_object[most_frequent_span].add(topic)
            if most_frequent_span in visited:
                continue
            span_lst.add(most_frequent_span)
    span_to_qanom_object_lst = get_qanom_objects_for_span_lst(list(span_lst))
    predicate_lst = set()
    filtered_topics = set()
    for span, qanom_object_lst in span_to_qanom_object_lst.items():
        for qanom_object in qanom_object_lst:
            predicate_lst.add(combine_spans_utils.dict_word_to_lemma.get(qanom_object.predicate, qanom_object.predicate))
        topics_mapped_to_span = span_to_topic_object[span]
        for topic in topics_mapped_to_span:
            if combine_spans_utils.is_similar_meaning_between_span(topic.common_lemmas_in_spans, predicate_lst):
                filtered_topics.add(topic)
    return filtered_topics


def get_topics_qa_nom_relations(topic_objects, dict_span_to_object):
    visited = []
    span_to_object = {}
    pred_to_que_nodes_dict = {}
    span_to_qanom_object_lst = {}
    object_to_most_frequent_span = {}
    pred_to_object_dict = {}
    filtered_topics = get_potential_qa_nom(topic_objects)
    for topic in filtered_topics:
        span_lst = []
        span_to_qanom_object_topic_children_lst = {}
        for child in topic.children:
            if child in visited:
                span = object_to_most_frequent_span[hash(child)]
                if span:
                    qanom_object_lst = span_to_qanom_object_lst.get(span, None)
                    if qanom_object_lst:
                        span_to_qanom_object_topic_children_lst[span] = qanom_object_lst
                continue
            visited.append(child)
            span = combine_spans_utils.get_most_frequent_span(child.span_lst)
            object_to_most_frequent_span[hash(child)] = span
            span_to_object[span] = child
            span_lst.append(span)
        if not span_lst:
            continue
        span_to_qanom_object_lst_temp = get_qanom_objects_for_span_lst(span_lst)
        span_to_qanom_object_topic_children_lst.update(span_to_qanom_object_lst_temp)
        span_to_qanom_object_lst.update(span_to_qanom_object_lst_temp)
        get_predicate_to_question_and_question_to_nodes_nested_dictionary(span_to_qanom_object_topic_children_lst,
                                                                          dict_span_to_object,
                                                                          pred_to_que_nodes_dict, topic,
                                                                          pred_to_object_dict)
    pred_to_que_nodes_filtered_dict = {}
    for pred, que_to_nodes in pred_to_que_nodes_dict.items():
        for que, nodes in que_to_nodes.items():
            if len(nodes) < 2:
                continue
            pred_to_que_nodes_filtered_dict[pred] = pred_to_que_nodes_filtered_dict.get(pred, {})
            pred_to_que_nodes_filtered_dict[pred][que] = pred_to_que_nodes_filtered_dict[pred].get(que, [])
            pred_to_que_nodes_filtered_dict[pred][que].extend(nodes)
    return pred_to_que_nodes_filtered_dict, pred_to_object_dict


def initialize_span_to_object_dict(dict_span_to_object, np_object, visited):
    if np_object in visited:
        return
    for span in np_object.span_lst:
        dict_span_to_object[span] = np_object
    visited.add(np_object)
    for child in np_object.children:
        initialize_span_to_object_dict(dict_span_to_object, child, visited)


def initialize_data():
    topic_objects = pickle.load(open("../results_disease/diabetes/topic_object_lst.p", "rb"))
    global_index_to_similar_longest_np = pickle.load(
        open("../results_disease/diabetes/global_index_to_similar_longest_np.p", "rb"))
    dict_span_to_rank = pickle.load(open("../results_disease/diabetes/dict_span_to_rank.p", "rb"))
    global_dict_label_to_object = pickle.load(open("../results_disease/diabetes/global_dict_label_to_object.p", "rb"))
    visited = set()
    dict_span_to_object = {}
    for np_object in topic_objects:
        initialize_span_to_object_dict(dict_span_to_object, np_object, visited)
    return topic_objects, global_index_to_similar_longest_np, dict_span_to_rank, \
           global_dict_label_to_object, dict_span_to_object


def get_answers_from_objects(object_tuple_lst):
    answers = set()
    for np_object, quest_object in object_tuple_lst:
        answers.add("(" + ', '.join(quest_object.answers) + ")")
        # answers.add(np_object)
    return ', '.join(list(answers))


def get_majority_role(objects):
    role_to_occurrences = {}
    for _, quest_object in objects:
        role_to_occurrences[quest_object.role] = role_to_occurrences.get(quest_object.role, 0)
        role_to_occurrences[quest_object.role] += 1
    return get_key_with_max_value(role_to_occurrences)


def union_children(questions_to_objects):
    tuple_nodes_lst = []
    visited = set()
    for question, nodes in questions_to_objects.items():
        for node_tuple in nodes:
            if node_tuple[0] in visited:
                continue
            visited.add(node_tuple[0])
            tuple_nodes_lst.append(node_tuple)
    return tuple_nodes_lst


def get_label_lst(tuple_nodes_lst):
    label_lst = set()
    for node, quest_object in tuple_nodes_lst:
        label_lst.update(node.label_lst)
    return label_lst


# add qanom to DAG
def add_object_to_DAG(pred_object, np_object_lst, new_np_object):
    for np_object in np_object_lst:
        if np_object in pred_object.children:
            pred_object.children.remove(np_object)
        if pred_object in np_object.parents:
            np_object.parents.remove(pred_object)
        new_np_object.children.append(np_object)
        new_np_object.parents.add(pred_object)
        np_object.parents.add(new_np_object)
    pred_object.children.append(new_np_object)


def from_tuple_node_lst_to_node_lst(tuple_nodes_lst):
    node_lst = set()
    for node, _ in tuple_nodes_lst:
        node_lst.add(node)
    return node_lst


def from_tuple_nodes_to_np_object(tuple_nodes_lst, questions, pred_object):
    statement = None
    selected_question = None
    for question in questions:
        statement = question_to_query_utils.conversion_question_to_statement(question)
        if statement:
            statement = statement.replace("**blank**", 'something')
            selected_question = question
            break
    if not statement:
        return None, None, None
    statement_doc = question_to_query_utils.nlp(statement)
    lemma_lst = utils_clustering.from_tokens_to_lemmas(statement_doc)
    label_lst = get_label_lst(tuple_nodes_lst)
    new_np_object = NounPhrase.NP([(statement, lemma_lst)], label_lst)
    np_object_lst = from_tuple_node_lst_to_node_lst(tuple_nodes_lst)
    add_object_to_DAG(pred_object, np_object_lst, new_np_object)
    return statement, selected_question, new_np_object


def create_node_for_wh_question(role_to_questions, questions_to_objects, pred_object):
    questions_to_objects = {k: v for k, v in sorted(questions_to_objects.items(), key=lambda item: len(item[1]),
                                                    reverse=True)}
    added_objects = []
    for role, questions in role_to_questions.items():
        wh_question_to_objects = {k: v for k, v in questions_to_objects.items() if k in questions}
        tuple_nodes_lst = union_children(wh_question_to_objects)
        statement, selected_question, new_np_object = from_tuple_nodes_to_np_object(tuple_nodes_lst, questions,
                                                                                    pred_object)
        if new_np_object:
            added_objects.append((statement, selected_question, new_np_object))
    return added_objects


def create_statements_objects(pred_to_que_nodes_filtered_dict, pred_to_object):
    added_objects = []
    for pred, questions_to_objects in pred_to_que_nodes_filtered_dict.items():
        pred_object = pred_to_object[pred]
        question_type_to_questions = {}
        for question, objects in questions_to_objects.items():
            wh_word = question.split()[0]
            role = get_majority_role(objects)
            question_type_to_questions[wh_word] = question_type_to_questions.get(wh_word, {})
            question_type_to_questions[wh_word][role] = question_type_to_questions[wh_word].get(role, [])
            question_type_to_questions[wh_word][role].append(question)
        for wh_type, role_to_questions in question_type_to_questions.items():
            wh_type_added_objects = create_node_for_wh_question(role_to_questions, questions_to_objects, pred_object)
            added_objects.append(wh_type_added_objects)
    return added_objects


def add_qanom_to_DAG(topic_objects, dict_span_to_object):
    pred_to_que_nodes_filtered_dict, pred_to_object_dict = get_topics_qa_nom_relations(topic_objects,
                                                                                       dict_span_to_object)
    dict_span_to_object.update(pred_to_object_dict)
    create_statements_objects(pred_to_que_nodes_filtered_dict, dict_span_to_object)
