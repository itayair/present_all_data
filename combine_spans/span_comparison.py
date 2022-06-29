import nltk
import torch
from combine_spans import utils as ut
import gensim
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
import gensim.downloader as api
from nltk.stem import PorterStemmer


# ps = PorterStemmer()


# Word2Vec_model = api.load("glove-wiki-gigaword-200")
# vocab_word2vec = list(Word2Vec_model.key_to_index.keys())
# Word2Vec_model = gensim.models.Word2Vec(common_texts, min_count=1, vector_size=100, window=5)
# model = SentenceTransformer('fse/word2vec-google-news-300)

# from transformers import AutoTokenizer, AutoModel


# def is_similar_meaning_between_span(span_1, span_2, dict_word_to_lemma, dict_lemma_to_synonyms):
#     span_1_lemma_lst = from_words_to_lemma_lst(span_1, dict_word_to_lemma)
#     span_2_lemma_lst = from_words_to_lemma_lst(span_2, dict_word_to_lemma)
#     for lemma in span_1_lemma_lst:
#         is_exist = False
#         if lemma in span_2_lemma_lst:
#             is_exist = True
#         else:
#             for synonym_lemma in dict_lemma_to_synonyms.get(lemma, []):
#                 if synonym_lemma in span_2_lemma_lst:
#                     is_exist = True
#                     break
#         if not is_exist:
#             return False
#     return True


def word_contained_in_list_by_edit_distance(word, lst_words_ref):
    for word_ref in lst_words_ref:
        val = nltk.edit_distance(word, word_ref)
        if val / max(len(word), len(word_ref)) <= 0.34:
            return True, word_ref
    return False, None


def compare_edit_distance_of_synonyms(synonyms, token, lemma_ref):
    close_words = set()
    for synonym in synonyms:
        edit_distance = nltk.edit_distance(synonym, token)
        edit_distance_lemma = nltk.edit_distance(synonym, lemma_ref)
        if edit_distance / max(len(token), len(synonym)) <= 0.34:
            close_words.add((synonym, token))
            continue
        if edit_distance_lemma / max(len(lemma_ref), len(synonym)) <= 0.34:
            close_words.add((synonym, lemma_ref))
            continue
    return list(close_words)


def remove_token_if_in_span(token, span, dict_lemma_to_synonyms):
    if token in span:
        span.remove(token)
        return True
    else:
        for synonym_lemma in dict_lemma_to_synonyms.get(token, []):
            if synonym_lemma in span:
                span.remove(synonym_lemma)
                return True
    return False


def is_similar_meaning_between_span(span_1, span_2, dict_word_to_lemma, dict_lemma_to_synonyms):
    span_1_lemma_lst = ut.from_words_to_lemma_lst(span_1, dict_word_to_lemma)
    span_2_lemma_lst = ut.from_words_to_lemma_lst(span_2, dict_word_to_lemma)
    not_satisfied = []
    for lemma in span_1_lemma_lst:
        is_exist = remove_token_if_in_span(lemma, span_2_lemma_lst, dict_lemma_to_synonyms)
        if not is_exist:
            not_satisfied.append(lemma)
    if len(not_satisfied) > 2:
        return False
    for lemma in not_satisfied:
        is_exist, lemma_to_remove = word_contained_in_list_by_edit_distance(lemma, span_2_lemma_lst)
        if is_exist:
            span_2_lemma_lst.remove(lemma_to_remove)
            continue
        return False
    return True


def cluster_phrases_by_similarity_rate(dict_phrase_to_embeds, phrase_list, cos_sim, phrase_embs, cos_sim_rate):
    counter = 0
    already_matched = set()
    dict_clustered_spans = {}
    for span in phrase_list:
        if span in already_matched:
            counter += 1
            continue
        dict_clustered_spans[span] = [span]
        already_matched.add(span)
        for phrase, embedding in dict_phrase_to_embeds.items():
            if phrase in already_matched:
                continue
            # print("The cosine similarity between " + phrase_list[0] + " and " + phrase + "  is " + str(
            #     {cos_sim(torch.tensor(phrase_embs[0]), torch.tensor(embedding))}))
            cos_sim_examples = cos_sim(torch.tensor(dict_phrase_to_embeds[span]), torch.tensor(embedding))
            if cos_sim_examples > cos_sim_rate:
                dict_clustered_spans[span].append(phrase)
                already_matched.add(phrase)
        counter += 1
    return dict_clustered_spans


def find_similarity_in_same_length_group(lst_spans_tuple, dict_word_to_lemma, dict_lemma_to_synonyms):
    black_list = []
    dict_span_to_similar_spans = {}
    for span_tuple in lst_spans_tuple:
        if span_tuple[0] in black_list:
            continue
        dict_span_to_similar_spans[span_tuple[0]] = [span_tuple[0]]
        black_list.append(span_tuple[0])
        for span_tuple_to_compare in lst_spans_tuple:
            if span_tuple_to_compare[0] in black_list:
                continue
            is_similar = is_similar_meaning_between_span(span_tuple[1], span_tuple_to_compare[1], dict_word_to_lemma,
                                                         dict_lemma_to_synonyms)
            if is_similar:
                black_list.append(span_tuple_to_compare[0])
                dict_span_to_similar_spans[span_tuple[0]].append(span_tuple_to_compare[0])
    return dict_span_to_similar_spans


# def is_span_contained_in_other_span(span_source, span_target):
#     if len(span_source) >= span_target:
#         return False
#     for token in span_source:
#         if token in span_target:
#             span_target.remove(token)
#             continue
#
#     if span_target:
#         return False
#     return True

def combine_not_clustered_spans_in_clustered_spans(not_clustered_spans, clustered_spans, dict_word_to_lemma, dict_lemma_to_synonyms, dict_span_to_lst):
    for cluster in clustered_spans.keys():
        span_to_remove = []
        for span, lst_spans in not_clustered_spans.items():
            if len(dict_span_to_lst[cluster]) < len(dict_span_to_lst[span]):
                if is_similar_meaning_between_span(dict_span_to_lst[cluster], dict_span_to_lst[span], dict_word_to_lemma, dict_lemma_to_synonyms):
                    if lst_spans == []:
                        clustered_spans[cluster].append(span)
                    else:
                        clustered_spans[cluster].extend(lst_spans)
                    span_to_remove.append(span)
        for span in span_to_remove:
            not_clustered_spans.pop(span, None)
