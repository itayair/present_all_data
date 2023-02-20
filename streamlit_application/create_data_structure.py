import spacy
from expansions_legacy import valid_deps, utils as ut
from streamlit_application import sentence_representation
from src.expansions import parse_medical_data

# import json
# import jsonpickle

noun_tags_lst = ['NN', 'NNS', 'WP', 'NNP', 'NNPS']


def get_all_closest_noun(head):
    noun_lst = []
    if head.tag_ in noun_tags_lst and head.text not in ['-', '(', ')', '"']:
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


def fill_all_head_phrase_in_tree(root, sentence, dict_noun_to_object, dict_noun_to_counter, is_the_first=False):
    head_phrase = dict_noun_to_object.get(root.basic_span, None)
    if head_phrase is None:
        head_phrase = sentence_representation.head_phrase(root)
        dict_noun_to_object[root.basic_span] = head_phrase
    head_phrase.add_new_node(root, sentence, is_the_first)
    if root.type == 1 and is_the_first:
        dict_noun_to_counter[root.basic_span] = dict_noun_to_counter.get(root.basic_span, 0) + 1
    for child in root.children_to_the_right:
        fill_all_head_phrase_in_tree(child, sentence, dict_noun_to_object, dict_noun_to_counter)


def combine_nodes_when_possible(node):
    new_children_to_the_right = []
    new_children_to_the_left = []
    num_of_children_with_relation_type = 0
    child_to_combine = None
    for child in node.children_to_the_right:
        combine_nodes_when_possible(child)
        if node.type == 2 and child.type == 2:
            child_to_combine = child
            num_of_children_with_relation_type += 1
            new_children_to_the_right.extend(child.children_to_the_right)
            new_children_to_the_left.extend(child.children_to_the_left)

    if num_of_children_with_relation_type == 1:
        node.span.extend(child_to_combine.span)
        node.span.sort(key=lambda x: x.i)
        node.initialize_attr_and_basic_span(node.span)
        node.span.sort(key=lambda x: x.i)
        node.children_to_the_right.remove(child_to_combine)
        node.children_to_the_right.extend(new_children_to_the_right)
        node.children_to_the_left.extend(new_children_to_the_left)


def create_data_structure(sentences, nlp):
    dict_noun_to_object = {}
    dict_noun_to_counter = {}
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
            all_valid_sub_np = valid_deps.get_all_valid_sub_np(head_noun, 1, head_noun.i)
            sub_np_final_lst, root = ut.from_lst_to_sequence(all_valid_sub_np, [], None)
            combine_nodes_when_possible(root)
            fill_all_head_phrase_in_tree(root, sent, dict_noun_to_object, dict_noun_to_counter, True)
        counter += 1
        if counter % 1000 == 0:
            print(counter)
    return dict_noun_to_object, dict_noun_to_counter


def create_data_structure_by_list_of_head_and_sentence():
    examples = parse_medical_data.get_examples_from_special_format()
    dict_noun_to_object = {}
    dict_noun_to_counter = {}
    counter = 0
    for example in examples:
        sent = example[0]
        head_of_span = example[4]
        if counter % 100 == 0:
            print(counter)
        counter += 1
        all_valid_sub_np = valid_deps.get_all_valid_sub_np(head_of_span, 1, head_of_span.i)
        sub_np_final_lst, root = ut.from_lst_to_sequence(all_valid_sub_np, [], None)
        combine_nodes_when_possible(root)
        fill_all_head_phrase_in_tree(root, sent, dict_noun_to_object, dict_noun_to_counter, True)
    return dict_noun_to_object, dict_noun_to_counter


nlp = spacy.load("en_ud_model_sm")
sentences_test = ["Ada Yoant is an Israely researcher who won a nobel prize in chemistry in 2004"]
create_data_structure(sentences_test, nlp)
