from src.topic_clustering import main_clustering as main_clustering
from src.expansions import parse_medical_data
import pickle

is_txt_format = True
if is_txt_format:
    file_name = 'input_files/text_files/chest_pain_causes.txt'
else:
    file_name = '../../input_files/input_json_files/pneumonia.json'
examples = parse_medical_data.get_examples_from_special_format(file_name, is_txt_format)
dict_noun_lemma_to_examples, dict_span_to_counter, dict_word_to_lemma, dict_lemma_to_synonyms, \
dict_longest_span_to_counter, dict_noun_lemma_to_synonyms, dict_noun_lemma_to_noun_words, dict_noun_lemma_to_counter, \
dict_noun_word_to_counter = main_clustering.convert_examples_to_clustered_data(examples)
directory_relative_path = "intermediate_load_files/chest_pain//"
a_file = open(directory_relative_path + "noun_lemma_to_example.pkl", "wb")
b_file = open(directory_relative_path + "span_counter.pkl", "wb")
c_file = open(directory_relative_path + "word_to_lemma.pkl", "wb")
d_file = open(directory_relative_path + "lemma_to_synonyms.pkl", "wb")
e_file = open(directory_relative_path + "longest_span_to_counter.pkl", "wb")
f_file = open(directory_relative_path + "noun_lemma_to_synonyms.pkl", "wb")
g_file = open(directory_relative_path + "noun_lemma_to_noun_words.pkl", "wb")
h_file = open(directory_relative_path + "noun_lemma_to_counter.pkl", "wb")
i_file = open(directory_relative_path + "noun_word_to_counter.pkl", "wb")
pickle.dump(dict_noun_lemma_to_examples, a_file)
pickle.dump(dict_span_to_counter, b_file)
pickle.dump(dict_word_to_lemma, c_file)
pickle.dump(dict_lemma_to_synonyms, d_file)
pickle.dump(dict_longest_span_to_counter, e_file)
pickle.dump(dict_noun_lemma_to_synonyms, f_file)
pickle.dump(dict_noun_lemma_to_noun_words, g_file)
pickle.dump(dict_noun_lemma_to_counter, h_file)
pickle.dump(dict_noun_word_to_counter, i_file)

i_file.close()
h_file.close()
g_file.close()
f_file.close()
e_file.close()
d_file.close()
c_file.close()
b_file.close()
a_file.close()
