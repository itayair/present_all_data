import pickle
import DAG.hierarchical_structure_algorithms as hierarchical_structure_algorithms
import DAG.NounPhraseObject as NounPhrase
import DAG.DAG_utils as DAG_utils
from combine_spans import utils as combine_spans_utils
import requests
import spacy
from nltk.corpus import stopwords
from topic_clustering import utils_clustering
import json
import torch

# outputs = pipe(sentences)
# for sentence in outputs:
#     qanom_sentence_data = sentence['qanom']
#     for qanom_sentence_predicate in qanom_sentence_data:
#         for question_data in qanom_sentence_predicate['QAs']:
#             question = question_data['question']
#             answers = question_data['answers']
#             role = question_data['question-role']
#             contextual_question = question_data['contextual_question']
#         predicate = qanom_sentence_predicate['predicate']
#         verb_form = qanom_sentence_predicate['verb_form']

# print(outputs)
cos = torch.nn.CosineSimilarity(dim=0, eps=1e-08)
nlp = spacy.load("en_ud_model_sm")
stop_words = set(stopwords.words('english'))


# neglect_deps = ['neg', 'case', 'mark', 'auxpass', 'aux', 'nummod', 'quantmod', 'cop']


def create_dict_RB_to_objects_lst(dict_RB_to_objects, np_object, visited, relation_type='RB'):
    if np_object in visited:
        return
    visited.add(np_object)
    for span in np_object.span_lst:
        if not span:
            continue
        dict_response = requests.post('http://127.0.0.1:5000/get_broader_terms/',
                                      params={"word": span, "relation_type": relation_type})
        output = dict_response.json()
        broader_terms = output['broader_term']
        if not broader_terms:
            continue
        broader_terms_set = set()
        for term in broader_terms:
            broader_terms_set.update(term)
        for term in broader_terms_set:
            dict_RB_to_objects[term] = dict_RB_to_objects.get(term, set())
            dict_RB_to_objects[term].add(np_object)
        break
    for child in np_object.children:
        create_dict_RB_to_objects_lst(dict_RB_to_objects, child, visited)


def is_parent_in_lst(np_object, object_lst):
    intersect_parents = np_object.parents.intersection(object_lst)
    if intersect_parents:
        return True
    for parent in np_object.parents:
        is_parent_in_list = is_parent_in_lst(parent, object_lst)
        if is_parent_in_list:
            return True
    return False


def initialize_span_to_object_dict(dict_span_to_object, np_object, visited):
    if np_object in visited:
        return
    for span in np_object.span_lst:
        dict_span_to_object[span] = np_object
    visited.add(np_object)
    for child in np_object.children:
        initialize_span_to_object_dict(dict_span_to_object, child, visited)


def update_parents_with_new_labels(node, label_lst):
    for parent in node.parents:
        parent.label_lst.update(label_lst)
        update_parents_with_new_labels(parent, label_lst)


def link_np_object_to_RB_related_nodes(np_object, object_lst, added_edges, added_taxonomic_relation,
                                       covered_labels_by_new_topics):
    for np_object_NT in object_lst:
        if np_object_NT == np_object:
            continue
        opposite_relation_in_our_DAG = is_parent_in_lst(np_object, [np_object_NT])
        if opposite_relation_in_our_DAG:
            continue
        is_already_ancestor = is_parent_in_lst(np_object_NT, [np_object])
        if is_already_ancestor:
            continue
        added_edges.append(np_object_NT)
        added_taxonomic_relation.add(np_object_NT)
        covered_labels_by_new_topics.update(np_object_NT.label_lst)
        np_object.add_children([np_object_NT])
        np_object_NT.parents.add(np_object)
    update_parents_with_new_labels(np_object, np_object.label_lst)


# def from_tokens_to_lemmas(tokens):
#     lemma_lst = []
#     for token in tokens:
#         if token.dep_ in neglect_deps or token.lemma_ in stop_words or token.text == '-':
#             continue
#         lemma_lst.append(token.lemma_.lower())
#     return lemma_lst


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


def filter_duplicate_relation(dict_RB_to_objects):
    keys_lst = list(dict_RB_to_objects.keys())
    black_lst = set()
    dict_span_to_equivalent = {}
    for idx, key in enumerate(keys_lst):
        if key in black_lst:
            continue
        for idx_ref in range(idx + 1, len(keys_lst)):
            key_ref = keys_lst[idx_ref]
            if key_ref in black_lst:
                continue
            object_lst = dict_RB_to_objects[key]
            object_lst_ref = dict_RB_to_objects[key_ref]
            if 0.67 * len(object_lst) > len(object_lst_ref):
                break
            intersect_lst = object_lst.intersection(object_lst_ref)
            if len(object_lst) != len(intersect_lst):
                continue
            black_lst.add(key_ref)
            dict_span_to_equivalent[key] = dict_span_to_equivalent.get(key, set())
            dict_span_to_equivalent[key].add(key_ref)

    dict_RB_to_objects = {key: dict_RB_to_objects[key] for key in dict_RB_to_objects if
                          key not in black_lst}
    return dict_RB_to_objects, dict_span_to_equivalent


def initialize_taxonomic_relations(topic_objects):
    dict_RB_to_objects = {}
    # Detect spans with broader term in UMLS
    visited = set()
    for np_object in topic_objects:
        create_dict_RB_to_objects_lst(dict_RB_to_objects, np_object, visited)
    dict_RB_to_objects = {k: v for k, v in
                          sorted(dict_RB_to_objects.items(), key=lambda item: len(item[1]),
                                 reverse=True)}
    # dict_RB_to_objects = {key: dict_RB_to_objects[key] for key in dict_RB_to_objects if
    #                       len(dict_RB_to_objects[key]) > 1}
    # Filter objects that their parents expressed by the new taxonomic relations
    for key, object_lst in dict_RB_to_objects.items():
        remove_lst = set()
        for np_object in object_lst:
            is_parent_in_list = is_parent_in_lst(np_object, object_lst)
            if is_parent_in_list:
                remove_lst.add(np_object)
        for np_object in remove_lst:
            object_lst.remove(np_object)
    dict_RB_to_objects = {k: v for k, v in
                          sorted(dict_RB_to_objects.items(), key=lambda item: len(item[1]),
                                 reverse=True)}
    dict_RB_to_objects = {key: dict_RB_to_objects[key] for key in dict_RB_to_objects if
                          len(dict_RB_to_objects[key]) > 1}
    # Filter duplicate relation
    dict_RB_to_objects, dict_span_to_equivalent = filter_duplicate_relation(dict_RB_to_objects)
    return dict_RB_to_objects, dict_span_to_equivalent


def detect_and_update_existing_object_represent_taxonomic_relation(dict_RB_to_objects,
                                                                   dict_span_to_equivalent, dict_span_to_object):
    entries_already_counted = set()
    dict_RB_exist_objects = {}
    for RB, object_lst in dict_RB_to_objects.items():
        if RB in dict_span_to_object:
            dict_RB_exist_objects[RB] = dict_span_to_object[RB]
            entries_already_counted.add(RB)
            continue
        equivalent_span_lst = dict_span_to_equivalent.get(RB, set())
        for equivalent_span in equivalent_span_lst:
            if equivalent_span in dict_span_to_object:
                entries_already_counted.add(RB)
                dict_RB_exist_objects[RB] = dict_span_to_object[equivalent_span]
                break
    added_edges = []
    added_taxonomic_relation = set()
    covered_labels_by_new_topics = set()
    # Link exist np_object_with the new taxonomic relation to other np objects
    counter = 0
    for RB, np_object in dict_RB_exist_objects.items():
        object_lst = dict_RB_to_objects[RB]
        link_np_object_to_RB_related_nodes(np_object, object_lst, added_edges, added_taxonomic_relation,
                                           covered_labels_by_new_topics)
        counter += 1
    return dict_RB_exist_objects, added_edges, added_taxonomic_relation, covered_labels_by_new_topics


def get_most_descriptive_span(nodes_lst, span_lst):
    max_score = 0
    best_span = ""
    max_represented_vector = None
    for span in span_lst:
        represented_vector = DAG_utils.get_represented_vector(span)
        cos_similarity_val = 0.0
        for node in nodes_lst:
            cos_similarity_val += cos(represented_vector, node.weighted_average_vector)
        if cos_similarity_val >= max_score:
            best_span = span
            max_score = cos_similarity_val
            max_represented_vector = represented_vector
    return best_span, max_represented_vector


# Create and add the new taxonomic relation to the DAG
def create_and_add_new_taxonomic_object_to_DAG(dict_RB_exist_objects, dict_RB_to_objects,
                                               dict_span_to_equivalent, dict_span_to_rank):
    black_lst = set(dict_RB_exist_objects.keys())
    new_taxonomic_np_objects = set()
    for RB, object_lst in dict_RB_to_objects.items():
        if RB in black_lst:
            continue
        equivalent_span_lst = dict_span_to_equivalent.get(RB, set())
        equivalent_span_lst.add(RB)
        span_tuple_lst = []
        span, represented_vector = get_most_descriptive_span(object_lst, equivalent_span_lst)
        # for span in equivalent_span_lst:
        span_as_doc = nlp(span)
        lemma_lst = utils_clustering.from_tokens_to_lemmas(span_as_doc)
        span_tuple_lst.append((span, lemma_lst))
        dict_span_to_rank[span] = len(lemma_lst)
        label_lst = set()
        for np_object in object_lst:
            label_lst.update(np_object.label_lst)
        new_np_object = NounPhrase.NP(span_tuple_lst, label_lst)
        new_np_object.weighted_average_vector = represented_vector
        new_taxonomic_np_objects.add(new_np_object)
        new_np_object.add_children(object_lst)
        for np_object in object_lst:
            np_object.parents.add(new_np_object)
    return new_taxonomic_np_objects


def covered_by_taxonomic_relation(new_taxonomic_np_objects, added_edges, added_taxonomic_relation,
                                  covered_labels_by_new_topics):
    # Check the coverage by the new components
    for np_object in new_taxonomic_np_objects:
        added_edges.extend(np_object.children)
        added_taxonomic_relation.update(np_object.children)
        covered_labels_by_new_topics.update(np_object.label_lst)


def initialize_nodes_weighted_average_vector(nodes_lst, global_index_to_similar_longest_np):
    for node in nodes_lst:
        DAG_utils.initialize_node_weighted_vector(node)
        node.frequency = DAG_utils.get_frequency_from_labels_lst(global_index_to_similar_longest_np,
                                                                 node.label_lst)


def combine_nodes_by_umls_spans_synonyms_dfs_helper(dict_span_to_object, np_object, visited,
                                                    dict_object_to_global_label,
                                                    global_dict_label_to_object):
    if np_object in visited:
        return
    visited.add(np_object)
    equivalent_object_lst = set()
    post_data = json.dumps(list(np_object.span_lst))
    dict_response = requests.post('http://127.0.0.1:5000/create_synonyms_dictionary/',
                                  params={"words": post_data})
    output = dict_response.json()
    synonyms_dict = output['synonyms']
    synonyms = set()
    for key, synonyms_lst in synonyms_dict.items():
        synonyms.update(synonyms_lst)
    if synonyms:
        for term in synonyms:
            equivalent_np_object = dict_span_to_object.get(term, None)
            if equivalent_np_object:
                if equivalent_np_object == np_object:
                    continue
                equivalent_object_lst.add(equivalent_np_object)
        if equivalent_object_lst:
            equivalent_object_lst = [np_object] + list(equivalent_object_lst)
            combine_nodes_lst = set()
            combine_spans_utils.combine_nodes_lst(equivalent_object_lst, dict_span_to_object, dict_object_to_global_label,
                                                  global_dict_label_to_object, combine_nodes_lst)
            for node in combine_nodes_lst:
                visited.add(node)

    children_lst = np_object.children.copy()
    for child in children_lst:
        combine_nodes_by_umls_spans_synonyms_dfs_helper(dict_span_to_object, child, visited,
                                                        dict_object_to_global_label,
                                                        global_dict_label_to_object)


def combine_nodes_by_umls_spans_synonyms(dict_span_to_object, dict_object_to_global_label, global_dict_label_to_object,
                                         topic_objects):
    visited = set()
    topic_object_lst = topic_objects.copy()
    for topic_object in topic_object_lst:
        if topic_object in visited:
            topic_objects.remove(topic_object)
            continue
        combine_nodes_by_umls_spans_synonyms_dfs_helper(dict_span_to_object, topic_object, visited,
                                                        dict_object_to_global_label,
                                                        global_dict_label_to_object)


def add_taxonomies_to_DAG_by_UMLS(topic_objects, dict_span_to_rank, dict_span_to_object, dict_object_to_global_label,
                                  global_dict_label_to_object):
    combine_nodes_by_umls_spans_synonyms(dict_span_to_object, dict_object_to_global_label, global_dict_label_to_object,
                                         topic_objects)
    # print("after synonyms taxonomic")
    # DAG_utils.check_symmetric_relation_in_DAG(topic_objects)
    # print("symetric is good after synonyms taxonomic")
    DAG_utils.update_symmetric_relation_in_DAG(topic_objects)
    dict_RB_to_objects, dict_span_to_equivalent = initialize_taxonomic_relations(topic_objects)
    # Find exist np objects that represent the taxonomic relation
    dict_RB_exist_objects, added_edges, added_taxonomic_relation, covered_labels_by_new_topics = \
        detect_and_update_existing_object_represent_taxonomic_relation(dict_RB_to_objects, dict_span_to_equivalent,
                                                                       dict_span_to_object)
    print("after update existing objects broader terms taxonomic")
    DAG_utils.update_symmetric_relation_in_DAG(topic_objects)
    print("symetric is good after update existing objects broader terms taxonomic")
    DAG_utils.check_symmetric_relation_in_DAG(topic_objects)
    new_taxonomic_np_objects = create_and_add_new_taxonomic_object_to_DAG(dict_RB_exist_objects,
                                                                          dict_RB_to_objects, dict_span_to_equivalent,
                                                                          dict_span_to_rank)
    print("after update new objects broader terms taxonomic")
    topic_objects.extend(new_taxonomic_np_objects)
    covered_by_taxonomic_relation(new_taxonomic_np_objects, added_edges, added_taxonomic_relation,
                                  covered_labels_by_new_topics)
    return new_taxonomic_np_objects
