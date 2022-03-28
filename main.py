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
nlp = spacy.load("en_ud_model_sm")


class obj_to_json:
    def __init__(self, label, value):
        self.label = label
        self.value = value

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)

        # self.kind = kind


kind_of_data = st.radio(
    "Which data do you want to use:",
    ('application data', 'add some sentences'))


def initialize_data_by_mode(sentences, data, dict_noun_to_object):
    st.session_state.dict_noun_to_object = dict_noun_to_object
    st.session_state.all_data = data
    st.session_state.data = data
    st.session_state.all_sentences = sentences
    st.session_state.sentences = sentences


def initialize_data(sentences):
    st.session_state.sentences = sentences
    dict_noun_to_object = create_data_structure.create_data_structure(sentences, nlp)
    item_lst_manual = []
    for key, value in dict_noun_to_object.items():
        item_lst_manual.append(key)
    st.session_state.data = item_lst_manual
    st.session_state.all_data = st.session_state.data
    st.session_state.dict_noun_to_object = dict_noun_to_object
    st.session_state.all_sentences = st.session_state.sentences
    return st.session_state.sentences, st.session_state.dict_noun_to_object, st.session_state.data


def initialize_head_phrase():
    new_sentences_lst = []
    nodes_for_lst = []
    span_for_lst = []
    dict_noun_to_object = {}
    for sentence, nodes in st.session_state.selected_node.sent_to_head_node_dict.items():
        if sentence in st.session_state.sentences:
            new_sentences_lst.append(sentence)
            nodes_for_lst.extend(nodes)
    bridge_to_head_phrase_child_dict = {}
    for node in nodes_for_lst:
        for child in node.children_to_the_right:
            span_for_lst.append(child.basic_span)
            # dict_noun_to_object[child.basic_span] = dict_noun_to_object.get(
            #     child.basic_span, [])
            # dict_noun_to_object[child.basic_span].append(child)
            if child.bridge_to_head:
                bridge_to_head_phrase_child_dict[child.bridge_to_head] = bridge_to_head_phrase_child_dict.get(
                    child.bridge_to_head, [])
                bridge_to_head_phrase_child_dict[child.bridge_to_head].append(child.basic_span)
    st.session_state.bridge_to_head_phrase_child_dict = bridge_to_head_phrase_child_dict
    st.session_state.sentences = new_sentences_lst
    st.session_state.valid_nodes = nodes_for_lst
    # st.session_state.dict_noun_to_object = dict_noun_to_object
    st.session_state.data = list(set(span_for_lst))


st.title("Aggregated Data")
# _my_component = components.declare_component(                gvvvvvvvvvvvvvvvvvvvvvvvvvvvb  x
#     "my_component",
#     url="http://localhost:3001"
# )

#
# def my_component(data, key=None):
#     return _my_component(data=data, default="", key=key)


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
    # nltk.download('punkt')

if kind_of_data == 'application data' and not st.session_state.is_application_data:
    st.session_state.is_application_data = True
    initialize_data_by_mode(st.session_state.all_sentences_application_data, st.session_state.all_data_application_data, st.session_state.dict_noun_to_object_application_data)
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
    option = st.selectbox(
        'Choose Head Phrase to expand',
        st.session_state.data)
    st.session_state.selected_node = st.session_state.dict_noun_to_object[option]
    span_to_add = option
else:
    if st.session_state.bridge_to_head_phrase_child_dict:
        bridge_option = st.selectbox(
            'Choose relation to another Head Phrase',
            st.session_state.bridge_to_head_phrase_child_dict)
        span_to_add = bridge_option
if (st.session_state.is_head_state and st.button("add to sentence")) or (
        not st.session_state.is_head_state and st.session_state.bridge_to_head_phrase_child_dict and st.button(
    "add to sentence")):
    if not st.session_state.is_head_state:
        item_lst = []
        for child_span in st.session_state.bridge_to_head_phrase_child_dict[bridge_option]:
            item_lst.append(child_span)
        st.session_state.data = list(set(item_lst))
    if st.session_state.span != "":
        st.session_state.span += " "
    if st.session_state.is_head_state:
        st.session_state.is_head_state = False
        initialize_head_phrase()
    else:

        st.session_state.is_head_state = True
    st.session_state.span += span_to_add
    st.experimental_rerun()

if st.session_state.span != "":
    word_to_complete = ""
    if st.session_state.is_head_state:
        word_to_complete = " {{something}}"
    st.write(st.session_state.span + word_to_complete)
#
# print(result)
