import nltk
from nltk import tokenize
import streamlit as st
import create_data_structure
import json
import csv
import spacy
import pip

# def install(package):
#     if hasattr(pip, 'main'):
#         pip.main(['install', package])
#     else:
#         pip._internal.main(['install', package])
#
# # Example
# if __name__ == '__main__':
#     install("https://storage.googleapis.com/en_ud_model/en_ud_model_sm-2.0.0.tar.gz")
DEFAULT_TEXT = "Used in select mask models , this new material improves upon silicone used for three decades in mask skirts with improved light transmission and much greater resistance to discoloration."


@st.cache(allow_output_mutation=True)
def load_model(name):
    nlp_object = spacy.load(name)
    # Add abbreviation detector
    # abbreviation_pipe = AbbreviationDetector(nlp)
    # nlp.add_pipe(abbreviation_pipe)
    return nlp_object


nlp = load_model("en_ud_model_sm")


class obj_to_json:
    def __init__(self, label, value):
        self.label = label
        self.value = value

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)

        # self.kind = kind


def initialize_data_by_mode(sentences, data, dict_noun_to_object):
    st.session_state.dict_noun_to_object = dict_noun_to_object
    st.session_state.all_data = data
    st.session_state.data = data
    st.session_state.all_sentences = sentences
    st.session_state.sentences = sentences


def initialize_data(sentences):
    st.session_state.sentences = sentences
    dict_noun_to_object, dict_noun_to_counter = create_data_structure.create_data_structure(sentences, nlp)
    st.session_state.data = {k + " (" + str(v) + ")": k for k, v in
                             sorted(dict_noun_to_counter.items(), key=lambda item: item[1], reverse=True)}
    st.session_state.all_data = st.session_state.data
    st.session_state.dict_noun_to_object = dict_noun_to_object
    st.session_state.all_sentences = st.session_state.sentences
    return st.session_state.sentences, st.session_state.dict_noun_to_object, st.session_state.data


def initialize_head_phrase(nodes_for_lst):
    # new_sentences_lst = []
    # nodes_for_lst = []
    relation_counter = {}
    nodes_for_lst = list(set(nodes_for_lst) & set(st.session_state.expanded_nodes))
    # for sentence, nodes in st.session_state.selected_node.sent_to_head_node_dict.items():
    #     if sentence in st.session_state.sentences:
    #         new_sentences_lst.append(sentence)
    #         nodes_for_lst.extend(nodes)
    bridge_to_head_phrase_child_dict = {}
    for node in nodes_for_lst:
        for child in node.children_to_the_right:
            if child.bridge_to_head:
                relation_counter[child.bridge_to_head] = relation_counter.get(child.bridge_to_head, 0) + 1
                bridge_to_head_phrase_child_dict[child.bridge_to_head] = bridge_to_head_phrase_child_dict.get(
                    child.bridge_to_head, [])
                bridge_to_head_phrase_child_dict[child.bridge_to_head].append(child)
    st.session_state.bridge_to_head_phrase_child_dict = bridge_to_head_phrase_child_dict
    st.session_state.relation_counter = {k + " (" + str(v) + ")": k for k, v in
                                         sorted(relation_counter.items(), key=lambda item: item[1], reverse=True)}
    # st.session_state.sentences = new_sentences_lst
    st.session_state.valid_nodes = nodes_for_lst


def update_when_relation_was_chose(relation_dep_option):
    children = st.session_state.bridge_to_head_phrase_child_dict[relation_dep_option]
    span_for_head_lst = set()
    dict_noun_to_counter = {}
    dict_basic_span_to_nodes = {}
    for child in children:
        dict_noun_to_counter[child.basic_span] = dict_noun_to_counter.get(child.basic_span, 0) + 1
        span_for_head_lst.add(child.basic_span)
        dict_basic_span_to_nodes[child.basic_span] = dict_basic_span_to_nodes.get(child.basic_span, [])
        dict_basic_span_to_nodes[child.basic_span].append(child)
    st.session_state.data = {k + " (" + str(v) + ")": k for k, v in
                             sorted(dict_noun_to_counter.items(), key=lambda item: item[1], reverse=True)}
    st.session_state.dict_basic_span_to_nodes = dict_basic_span_to_nodes


kind_of_data = st.sidebar.radio(
    "Which data do you want to use:",
    ('application data', 'add some sentences'))

st.title("Aggregated Data")

if "id" not in st.session_state:
    st.session_state.id = 0
    st.session_state.new = True
    st.session_state.span = ""
    st.session_state.is_head_state = True
    st.session_state.text_in_manual_mode = ""
    st.session_state.is_application_data = True
else:
    st.session_state.id += 1
item_lst = []
if st.button("restart"):
    st.session_state.span = ""
    st.session_state.is_head_state = True
    st.session_state.data = st.session_state.all_data
    st.session_state.sentences = st.session_state.all_sentences

if st.session_state.span != "":
    word_to_complete = ""
    if st.session_state.is_head_state:
        word_to_complete = " {{something}}"
    st.write(st.session_state.span + word_to_complete)

if st.session_state.id == 0:
    used_for_examples = open('./csv/examples_used_for.csv', encoding="utf8")
    csv_reader_used_for_examples = csv.reader(used_for_examples)
    header = next(csv_reader_used_for_examples)
    sent_to_collect = []
    st.session_state.all_sentences = []
    for row in csv_reader_used_for_examples:
        sent_to_collect.append(row[13])
    st.session_state.all_sentences_application_data, st.session_state.dict_noun_to_object_application_data, st.session_state.all_data_application_data = initialize_data(
        sent_to_collect)
    nltk.download('punkt')

if kind_of_data == 'application data' and not st.session_state.is_application_data:
    st.session_state.is_application_data = True
    initialize_data_by_mode(st.session_state.all_sentences_application_data, st.session_state.all_data_application_data,
                            st.session_state.dict_noun_to_object_application_data)
    st.session_state.span = ""
    st.session_state.is_head_state = True
if kind_of_data == 'add some sentences':
    st.header("Enter a sentence:")
    text = st.text_area("", DEFAULT_TEXT)
    if st.session_state.text_in_manual_mode != text or st.session_state.is_application_data:
        st.session_state.text_in_manual_mode = text
        text = tokenize.sent_tokenize(text)
        st.session_state.all_sentences_manual_data, st.session_state.dict_noun_to_object_application_manual_data, st.session_state.all_data_manual_data = initialize_data(
            text)
        st.session_state.span = ""
        st.session_state.is_head_state = True
    st.session_state.is_application_data = False
if st.session_state.is_head_state:
    st.session_state.expanded_nodes = []
    option = st.selectbox(
        'Choose Head Phrase to expand',
        st.session_state.data)
    option = st.session_state.data[option]
    st.session_state.selected_node = st.session_state.dict_noun_to_object[option]
    span_to_add = option
    agree = st.checkbox(
        "press here to get all the optional expansions of " + span_to_add)
    if agree:
        if st.session_state.span == "":
            nodes_to_get_all_expansions = st.session_state.selected_node.head_node_lst
        else:
            nodes_to_get_all_expansions = st.session_state.dict_basic_span_to_nodes[option]
        from_node_to_all_his_expansion_to_the_left = st.session_state.selected_node.from_node_to_all_his_expansion_to_the_left
        counter_of_expansion_occurrences = {}
        expansion_to_node = {}
        for node in nodes_to_get_all_expansions:
            all_his_expansion_to_the_left = from_node_to_all_his_expansion_to_the_left[node]
            for expansion in all_his_expansion_to_the_left:
                expansion_to_node[expansion] = expansion_to_node.get(expansion, [])
                expansion_to_node[expansion].append(node)
                counter_of_expansion_occurrences[expansion] = counter_of_expansion_occurrences.get(expansion, 0) + 1
        counter_of_expansion_occurrences = {k + " (" + str(v) + ")": k for k, v in sorted(counter_of_expansion_occurrences.items(), key=lambda item: item[1], reverse=True)}
        expanded_option = st.selectbox(
            'Choose the expansion',
            counter_of_expansion_occurrences)
        expanded_option = counter_of_expansion_occurrences[expanded_option]
        st.session_state.expanded_nodes = expansion_to_node[expanded_option]
        span_to_add = expanded_option
else:
    if st.session_state.bridge_to_head_phrase_child_dict:
        relation_dep_option = st.selectbox(
            'Choose relation to another Head Phrase',
            st.session_state.relation_counter)
        relation_dep_option = st.session_state.relation_counter[relation_dep_option]
        span_to_add = relation_dep_option
if (st.session_state.is_head_state and st.button("add to sentence")) or (
        not st.session_state.is_head_state and st.session_state.bridge_to_head_phrase_child_dict and st.button(
    "add to sentence")):
    if st.session_state.is_head_state:
        st.session_state.is_head_state = False
        if st.session_state.span == "":
            initialize_head_phrase(st.session_state.selected_node.head_node_lst)
        else:
            initialize_head_phrase(st.session_state.dict_basic_span_to_nodes[option])
    else:
        update_when_relation_was_chose(relation_dep_option)
        st.session_state.is_head_state = True
    if st.session_state.span != "":
        st.session_state.span += " "
    st.session_state.span += span_to_add
    st.experimental_rerun()
#
# print(result)
