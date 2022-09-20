from nltk.corpus import wordnet
import pickle
from combine_spans import span_comparison


def get_synonyms_by_word(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.add(l.name())
    # aliases = umls_loader.umls_loader.get_term_aliases(word)
    # for syn in aliases:
    #     synonyms.append(syn)
    return synonyms


def from_words_to_lemma_lst(span, dict_word_to_lemma):
    lemmas_lst = []
    for word in span:
        lemma = dict_word_to_lemma.get(word, None)
        if lemma is None:
            lemma = word
        lemmas_lst.append(lemma)
    return lemmas_lst


def load_data_dicts():
    a_file = open("load_data\\data.pkl", "rb")
    topics_dict = pickle.load(a_file)
    topics_dict = {k: v for k, v in
                      sorted(topics_dict.items(), key=lambda item: len(item[1]),
                             reverse=True)}
    b_file = open("load_data\\span_counter.pkl", "rb")
    dict_span_to_counter = pickle.load(b_file)
    c_file = open("load_data\\word_to_lemma.pkl", "rb")
    dict_noun_head_to_lemma = pickle.load(c_file)
    d_file = open("load_data\\word_to_synonyms.pkl", "rb")
    dict_noun_head_to_synonyms = pickle.load(d_file)
    e_file = open("load_data\\topic_to_his_synonym.pkl", "rb")
    dict_topic_to_synonym = pickle.load(e_file)
    return topics_dict, dict_span_to_counter, dict_noun_head_to_lemma, dict_noun_head_to_synonyms, dict_topic_to_synonym


dict_of_topics, dict_of_span_to_counter, dict_word_to_lemma, dict_lemma_to_synonyms, dict_topic_to_his_synonym = load_data_dicts()
dict_of_span_to_counter = {k: v for k, v in
                           sorted(dict_of_span_to_counter.items(), key=lambda item: item[1],
                                  reverse=True)}


def get_average_value(spans_lst, dict_span_to_rank):
    average_val = 0
    for span in spans_lst:
        average_val += dict_span_to_rank[span]
    average_val = average_val / len(spans_lst)
    return int(average_val)


def create_dict_lemma_word2vec_and_edit_distance(dict_lemma_to_synonyms, dict_word_to_lemma):
    words_lst = list(dict_word_to_lemma.keys())
    dict_lemma_to_close_words = {}
    counter = 0
    for word, lemma in dict_word_to_lemma.items():
        dict_lemma_to_close_words[word] = []
        # if word not in vocab_word2vec:
        #     counter += 1
        #     continue
        for word_ref_idx in range(counter + 1, len(words_lst)):
            word_ref = words_lst[word_ref_idx]
            lemma_ref = dict_word_to_lemma[word_ref]
            synonyms = [word, lemma] + dict_lemma_to_synonyms[lemma]
            synonyms = list(set(synonyms))
            if lemma == lemma_ref or lemma_ref in synonyms:
                continue
            # if word_ref not in vocab_word2vec:
            #     continue
            # sim_val = Word2Vec_model.similarity(word, word_ref)
            # if 0.8 < sim_val < 0.9:
            dict_lemma_to_close_words[word].extend(
                span_comparison.compare_edit_distance_of_synonyms(synonyms, word_ref, lemma_ref))
        counter += 1
    return dict_lemma_to_close_words


def get_most_frequent_span(lst_of_spans):
    most_frequent_span_value = -1
    most_frequent_span = None
    for span in lst_of_spans:
        val = dict_of_span_to_counter.get(span, 0)
        if val > most_frequent_span_value:
            most_frequent_span_value = val
            most_frequent_span = span
    return most_frequent_span

