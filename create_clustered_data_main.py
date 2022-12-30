from topic_clustering import main_clustering as main_clustering
import pickle


file_name = "text_files/output_noun_result.txt"
file_name_lemma = "text_files/output_noun_lemma_result.txt"

dict_noun_lemma_to_examples, dict_span_to_counter, dict_word_to_lemma, dict_lemma_to_synonyms, \
dict_longest_span_to_counter, dict_noun_lemma_to_synonyms, dict_noun_lemma_to_noun_words, dict_noun_lemma_to_counter, \
dict_noun_word_to_counter = main_clustering.convert_examples_to_clustered_data()
directory_relative_path = "load_data\\abortion\\"
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
