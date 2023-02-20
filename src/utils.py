from src.topic_clustering import main_clustering as main_clustering
from src.expansions import parse_medical_data
import pickle
import torch
import os
import sys
from transformers import AutoTokenizer, AutoModel
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
sapBert_tokenizer = AutoTokenizer.from_pretrained('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')
sapBert_model = AutoModel.from_pretrained('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')
model = sapBert_model.to(device)
model = model.eval()
print(os.getcwd())


def load_data_dicts(etiology='chest_pain'):
    directory_relative_path = "intermediate_load_files//" + etiology + "//"
    a_file = open(directory_relative_path + "noun_lemma_to_example.pkl", "rb")
    topics_dict = pickle.load(a_file)
    topics_dict = {k: v for k, v in
                   sorted(topics_dict.items(), key=lambda item: len(item[1]),
                          reverse=True)}
    b_file = open(directory_relative_path + "span_counter.pkl", "rb")
    dict_span_to_counter = pickle.load(b_file)
    c_file = open(directory_relative_path + "word_to_lemma.pkl", "rb")
    dict_word_to_lemma = pickle.load(c_file)
    d_file = open(directory_relative_path + "lemma_to_synonyms.pkl", "rb")
    dict_lemma_to_synonyms = pickle.load(d_file)
    e_file = open(directory_relative_path + "longest_span_to_counter.pkl", "rb")
    dict_longest_span_to_counter = pickle.load(e_file)
    f_file = open(directory_relative_path + "noun_lemma_to_synonyms.pkl", "rb")
    dict_noun_lemma_to_synonyms = pickle.load(f_file)
    g_file = open(directory_relative_path + "noun_lemma_to_noun_words.pkl", "rb")
    dict_noun_lemma_to_noun_words = pickle.load(g_file)
    h_file = open(directory_relative_path + "noun_lemma_to_counter.pkl", "rb")
    dict_noun_lemma_to_counter = pickle.load(h_file)
    i_file = open(directory_relative_path + "noun_word_to_counter.pkl", "rb")
    dict_noun_word_to_counter = pickle.load(i_file)

    return topics_dict, dict_span_to_counter, dict_word_to_lemma, dict_lemma_to_synonyms, \
           dict_longest_span_to_counter, dict_noun_lemma_to_synonyms, dict_noun_lemma_to_noun_words, \
           dict_noun_lemma_to_counter, dict_noun_word_to_counter


# dict_of_topics, dict_span_to_counter, dict_word_to_lemma, dict_lemma_to_synonyms, \
# dict_longest_span_to_counter, dict_noun_lemma_to_synonyms, dict_noun_lemma_to_noun_words, dict_noun_lemma_to_counter, \
# dict_noun_word_to_counter = load_data_dicts()
# is_txt_format = True
# if is_txt_format:
#     file_name = 'input_files/text_files/chest_pain_causes.txt'
# else:
#     file_name = 'input_files/input_json_files/jaundice.json'
# with open('input.txt') as f:
#     lines = f.readlines()
#     file_name = lines[0]
#     is_txt_format = lines[1]
#     if is_txt_format == '1':
#         is_txt_format = True
#     else:
#         is_txt_format = False
file_name = sys.argv[1]
is_txt_format = sys.argv[2]
etiology = sys.argv[3]
entries_number_limit = int(sys.argv[4])
if is_txt_format == '1':
    is_txt_format = True
else:
    is_txt_format = False
examples = parse_medical_data.get_examples_from_special_format(file_name, is_txt_format)
dict_noun_lemma_to_examples, dict_span_to_counter, dict_word_to_lemma, dict_lemma_to_synonyms, \
dict_longest_span_to_counter, dict_noun_lemma_to_synonyms, dict_noun_lemma_to_noun_words, dict_noun_lemma_to_counter, \
dict_noun_word_to_counter = main_clustering.convert_examples_to_clustered_data(examples)
topics_dict = {k: v for k, v in
               sorted(dict_noun_lemma_to_examples.items(), key=lambda item: len(item[1]),
                      reverse=True)}
dict_span_to_counter.update(dict_noun_word_to_counter)
dict_span_to_counter.update(dict_noun_lemma_to_counter)
dict_span_to_lemma_lst = {}


# dict_of_span_to_counter = {k: v for k, v in
#                            sorted(dict_of_span_to_counter.items(), key=lambda item: item[1],
#                                   reverse=True)}
