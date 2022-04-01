import nltk
from nltk import tokenize
import streamlit as st
import create_data_structure
import json
import csv
import spacy
import pandas as pd
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


@st.cache(
    hash_funcs={spacy.lang.en.English: lambda _: initialize_data, spacy.tokens.token.Token: lambda _: initialize_data},
    allow_output_mutation=True)
def initialize_data(sentences):
    dict_noun_to_object, dict_noun_to_counter = create_data_structure.create_data_structure(sentences, nlp)
    data = {k + " (" + str(v) + ")": k for k, v in
            sorted(dict_noun_to_counter.items(), key=lambda item: item[1], reverse=True)}
    return data, dict_noun_to_object, dict_noun_to_counter


@st.cache(
    hash_funcs={spacy.lang.en.English: lambda _: initialize_data, spacy.tokens.token.Token: lambda _: initialize_data},
    allow_output_mutation=True)
def initialize_data_from_csv():
    used_for_examples = open('./csv/examples_used_for.csv', encoding="utf8")
    csv_reader_used_for_examples = csv.reader(used_for_examples)
    next(csv_reader_used_for_examples)
    sent_to_collect = []
    for row in csv_reader_used_for_examples:
        sent_to_collect.append(row[13])
        sent_to_collect = list(set(sent_to_collect))
    return initialize_data(sent_to_collect)


def initialize_manual_data(sentences):
    dict_noun_to_object, dict_noun_to_counter = create_data_structure.create_data_structure(sentences, nlp)
    data = {k + " (" + str(v) + ")": k for k, v in
            sorted(dict_noun_to_counter.items(), key=lambda item: item[1], reverse=True)}
    return data, dict_noun_to_object, dict_noun_to_counter


class obj_to_json:
    def __init__(self, label, value):
        self.label = label
        self.value = value

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)

        # self.kind = kind


def initialize_data_by_mode(data, dict_noun_to_object):
    st.session_state.dict_noun_to_object = dict_noun_to_object
    st.session_state.all_data = data
    st.session_state.data = data


def initialize_head_phrase(nodes_for_lst):
    relation_counter = {}
    nodes_for_lst = list(set(nodes_for_lst) & set(st.session_state.expanded_nodes))
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
    st.session_state.valid_nodes = nodes_for_lst


def update_when_relation_was_chose(chosen_relation_dep_option):
    children = st.session_state.bridge_to_head_phrase_child_dict[chosen_relation_dep_option]
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
st.sidebar.write(
    "The restart button is used for the development(loading the data and do all the preprocessing from the beginning)")
if st.sidebar.button("Restart"):
    st.session_state.id = 0
    st.session_state.span = ""
    st.session_state.is_head_state = True
    st.session_state.data = st.session_state.all_data

kind_of_data = st.sidebar.radio(
    "Which data do you want to use:",
    ('Application data', 'Your own sentences'))

if st.button("start from the beginning"):
    st.session_state.span = ""
    st.session_state.is_head_state = True
    st.session_state.data = st.session_state.all_data

if st.session_state.span != "":
    word_to_complete = ""
    if st.session_state.is_head_state:
        word_to_complete = " {{something}}"
    st.write(st.session_state.span + word_to_complete)

if "id" not in st.session_state or st.session_state.id == 0:
    st.session_state.data, st.session_state.dict_noun_to_object, dict_noun_to_counter = initialize_data_from_csv()
    st.session_state.all_data = st.session_state.data
    st.session_state.all_data_application_data = st.session_state.data
    st.session_state.dict_noun_to_counter = dict_noun_to_counter
    st.session_state.dict_noun_to_object_application_data = st.session_state.dict_noun_to_object

if kind_of_data == 'Application data' and not st.session_state.is_application_data:
    st.session_state.is_application_data = True
    initialize_data_by_mode(st.session_state.all_data_application_data,
                            st.session_state.dict_noun_to_object_application_data)
    st.session_state.span = ""
    st.session_state.is_head_state = True
if kind_of_data == 'Your own sentences':
    st.header("Please enter here some sentences:")
    text = st.text_area("", DEFAULT_TEXT)
    if st.session_state.text_in_manual_mode != text or st.session_state.is_application_data:
        st.session_state.text_in_manual_mode = text
        text = tokenize.sent_tokenize(text)
        st.session_state.dict_noun_to_object_application_manual_data, st.session_state.all_data_manual_data, dict_noun_to_counter = initialize_manual_data(
            text)
        st.session_state.dict_noun_to_counter = dict_noun_to_counter
        st.session_state.data = st.session_state.dict_noun_to_object_application_manual_data
        st.session_state.all_data = st.session_state.data
        st.session_state.span = ""
        st.session_state.is_head_state = True
    st.session_state.is_application_data = False
if st.session_state.is_head_state:
    st.session_state.expanded_nodes = []
    option = st.selectbox(
        'Choose span to expand',
        st.session_state.data)
    option = st.session_state.data[option]
    st.session_state.selected_node = st.session_state.dict_noun_to_object[option]
    span_to_add = option
    agree = st.checkbox(
        "Press here to get all the optional expansions of " + span_to_add)
    if st.session_state.span == "":
        st.session_state.expanded_nodes = st.session_state.selected_node.head_node_lst
        nodes_to_get_all_expansions = st.session_state.selected_node.head_node_lst
    else:
        nodes_to_get_all_expansions = st.session_state.dict_basic_span_to_nodes[option]
        st.session_state.expanded_nodes = st.session_state.dict_basic_span_to_nodes[option]
    if agree:
        from_node_to_all_his_expansion_to_the_left = st.session_state.selected_node.from_node_to_all_his_expansion_to_the_left
        counter_of_expansion_occurrences = {}
        expansion_to_node = {}
        for node in nodes_to_get_all_expansions:
            all_his_expansion_to_the_left = from_node_to_all_his_expansion_to_the_left[node]
            for expansion in all_his_expansion_to_the_left:
                expansion_to_node[expansion] = expansion_to_node.get(expansion, [])
                expansion_to_node[expansion].append(node)
                counter_of_expansion_occurrences[expansion] = counter_of_expansion_occurrences.get(expansion, 0) + 1
        counter_of_expansion_occurrences = {k + " (" + str(v) + ")": k for k, v in
                                            sorted(counter_of_expansion_occurrences.items(), key=lambda item: item[1],
                                                   reverse=True)}
        expanded_option = st.selectbox(
            'Choose the expanded span',
            counter_of_expansion_occurrences)
        expanded_option = counter_of_expansion_occurrences[expanded_option]
        st.session_state.expanded_nodes = expansion_to_node[expanded_option]
        span_to_add = expanded_option
    placeholder = st.empty()
    sent_lst = []
    for node in st.session_state.expanded_nodes:
        sent_lst.append(st.session_state.selected_node.from_node_to_sentence[node])
    df = pd.DataFrame(sent_lst, columns=["sentences"])
    dfStyler = df.style.set_properties(**{'text-align': 'left'})
    dfStyler.set_table_styles([dict(selector='th', props=[('text-align', 'left')])])
    st.dataframe(df)
else:
    if st.session_state.bridge_to_head_phrase_child_dict:
        relation_dep_option = st.selectbox(
            'Choose the relation to another span',
            st.session_state.relation_counter)
        relation_dep_option = st.session_state.relation_counter[relation_dep_option]
        span_to_add = relation_dep_option
if (st.session_state.is_head_state and placeholder.button("add to sentence")) or (
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
