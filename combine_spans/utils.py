from nltk.corpus import wordnet
import pickle
from combine_spans import span_comparison


def get_synonyms_by_word(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.add(l.name())
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
    a_file = open("data.pkl", "rb")
    dict_of_topics = pickle.load(a_file)
    dict_of_topics = {k: v for k, v in
                      sorted(dict_of_topics.items(), key=lambda item: len(item[1]),
                             reverse=True)}
    b_file = open("span_counter.pkl", "rb")
    dict_of_span_to_counter = pickle.load(b_file)
    c_file = open("word_to_lemma.pkl", "rb")
    dict_word_to_lemma = pickle.load(c_file)
    return dict_of_topics, dict_of_span_to_counter, dict_word_to_lemma


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


def get_most_frequent_span(lst_of_spans, dict_of_span_to_counter):
    most_frequent_span_value = -1
    most_frequent_span = None
    for span in lst_of_spans:
        val = dict_of_span_to_counter.get(span, 0)
        if val > most_frequent_span_value:
            most_frequent_span_value = val
            most_frequent_span = span
    return most_frequent_span


def create_dicts_for_words_similarity(dict_word_to_lemma):
    dict_lemma_to_synonyms = {}
    lemma_lst = set()
    for _, lemma in dict_word_to_lemma.items():
        lemma_lst.add(lemma)
    for lemma in lemma_lst:
        synonyms = get_synonyms_by_word(lemma)
        synonyms = [synonym for synonym in synonyms if synonym in lemma_lst]
        dict_lemma_to_synonyms[lemma] = synonyms
    dict_lemma_to_synonyms = {k: v for k, v in
                              sorted(dict_lemma_to_synonyms.items(), key=lambda item: len(item[1]),
                                     reverse=True)}
    return dict_lemma_to_synonyms
