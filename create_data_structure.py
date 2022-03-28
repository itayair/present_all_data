import spacy
import utils as ut
import valid_deps
import sentence_representation
import csv
# import json
# import jsonpickle

noun_tags_lst = ['NN', 'NNS', 'WP', 'NNP', 'NNPS']


def get_all_closest_noun(head):
    noun_lst = []
    if head.tag_ in noun_tags_lst:
        return [head]
    for child in head.children:
        noun_lst.extend(get_all_closest_noun(child))
    return noun_lst


def get_head_noun_in_sentence(sentence_dep_graph):
    head_lst = []
    for token in sentence_dep_graph:
        if token.head == token:
            head_lst.append(token)
    noun_lst = []
    for head in head_lst:
        noun_lst.extend(get_all_closest_noun(head))
    return noun_lst


def fill_all_head_phrase_in_tree(root, sentence, dict_noun_to_object):
    head_phrase = dict_noun_to_object.get(root.basic_span, None)
    if head_phrase is None:
        head_phrase = sentence_representation.head_phrase(root.basic_span)
        if root.basic_span == "":
            print("error")
        dict_noun_to_object[root.basic_span] = head_phrase
    head_phrase.add_new_node(root, sentence)
    dict_noun_to_counter[root.basic_span] = dict_noun_to_counter.get(root.basic_span, 0) + 1
    for child in root.children_to_the_right:
        fill_all_head_phrase_in_tree(child, sentence, dict_noun_to_object)


dict_noun_to_counter = {}


def create_data_structure(sentences, nlp):
    dict_noun_to_object = {}
    counter = 0
    for sent in sentences:
        # if counter > 200:
        #     break
        sentence_dep_graph = nlp(sent)
        head_noun_lst = get_head_noun_in_sentence(sentence_dep_graph)
        if head_noun_lst is []:
            continue
        for head_noun in head_noun_lst:
            noun_phrase, _, boundary_length = ut.get_np_boundary(head_noun.i, sentence_dep_graph)
            if noun_phrase is None:
                continue
            if boundary_length > 20:
                continue
            all_valid_sub_np = valid_deps.get_all_valid_sub_np(head_noun, head_noun.i)
            sub_np_final_lst, root = ut.from_lst_to_sequence(all_valid_sub_np, [], None)
            fill_all_head_phrase_in_tree(root, sent, dict_noun_to_object)
        counter += 1
        if counter % 1000 == 0:
            print(counter)
    return dict_noun_to_object

