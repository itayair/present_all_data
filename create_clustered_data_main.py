from topic_clustering import main_clustering as main_clustering
import pickle
file_name = "text_files/output_noun_result.txt"
file_name_lemma = "text_files/output_noun_lemma_result.txt"

dict_noun_lemma_to_example, dict_span_to_counter, dict_word_to_lemma, dict_lemma_to_synonyms, dict_word_to_his_synonym = main_clustering.convert_examples_to_clustered_data()

a_file = open("load_data/data.pkl", "wb")
b_file = open("load_data/span_counter.pkl", "wb")
c_file = open("load_data/word_to_lemma.pkl", "wb")
d_file = open("load_data/word_to_synonyms.pkl", "wb")
e_file = open("load_data/topic_to_his_synonym.pkl", "wb")
pickle.dump(dict_noun_lemma_to_example, a_file)
pickle.dump(dict_span_to_counter, b_file)
pickle.dump(dict_word_to_lemma, c_file)
pickle.dump(dict_lemma_to_synonyms, d_file)
pickle.dump(dict_word_to_his_synonym, e_file)
e_file.close()
d_file.close()
c_file.close()
b_file.close()
a_file.close()