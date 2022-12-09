from combine_spans import utils as ut
from sentence_transformers import SentenceTransformer
import torch
from sklearn.cluster import AgglomerativeClustering
# import gensim
# from gensim.models import Word2Vec
# from gensim.test.utils import common_texts
# import gensim.downloader as api
from nltk.stem import PorterStemmer

# ps = PorterStemmer()


# Word2Vec_model = api.load("glove-wiki-gigaword-200")
# vocab_word2vec = list(Word2Vec_model.key_to_index.keys())
# Word2Vec_model = gensim.models.Word2Vec(common_texts, min_count=1, vector_size=100, window=5)
# model = SentenceTransformer('fse/word2vec-google-news-300)

# from transformers import AutoTokenizer, AutoModel

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


# def cluster_phrases_by_similarity_rate(dict_phrase_to_embeds, phrase_list, cos_sim, cos_sim_rate):
#     counter = 0
#     already_matched = set()
#     dict_clustered_spans = {}
#     for span in phrase_list:
#         if span in already_matched:
#             counter += 1
#             continue
#         dict_clustered_spans[span] = [span]
#         already_matched.add(span)
#         for phrase, embedding in dict_phrase_to_embeds.items():
#             if phrase in already_matched:
#                 continue
#             cos_sim_examples = cos_sim(torch.tensor(dict_phrase_to_embeds[span]), torch.tensor(embedding))
#             if cos_sim_examples > cos_sim_rate:
#                 dict_clustered_spans[span].append(phrase)
#                 already_matched.add(phrase)
#         counter += 1
#     return dict_clustered_spans


def find_similarity_in_same_length_group(lst_spans_tuple):
    black_list = []
    dict_span_to_similar_spans = {}
    for span_tuple in lst_spans_tuple:
        if span_tuple[0] in black_list:
            continue
        dict_span_to_similar_spans[span_tuple[0]] = set()
        dict_span_to_similar_spans[span_tuple[0]].add(span_tuple[0])
        black_list.append(span_tuple[0])
        for span_tuple_to_compare in lst_spans_tuple:
            if span_tuple_to_compare[0] in black_list:
                continue
            is_similar = ut.is_similar_meaning_between_span(span_tuple[1], span_tuple_to_compare[1])
            if is_similar:
                black_list.append(span_tuple_to_compare[0])
                dict_span_to_similar_spans[span_tuple[0]].add(span_tuple_to_compare[0])
    return dict_span_to_similar_spans


def combine_similar_longest_np_with_common_sub_nps(common_np_to_group_members_indices,
                                                   dict_longest_span_to_his_synonyms, dict_span_to_similar_spans):
    black_lst = []
    for span, indices_group in common_np_to_group_members_indices.items():
        if span in black_lst:
            continue
        synonyms_span = dict_span_to_similar_spans[span].intersection(set(dict_longest_span_to_his_synonyms.keys()))
        for longest_span_synonym in synonyms_span:
            if longest_span_synonym != span and longest_span_synonym in common_np_to_group_members_indices.keys():
                common_np_to_group_members_indices[span].update(
                    common_np_to_group_members_indices[longest_span_synonym])
                if longest_span_synonym not in black_lst:
                    black_lst.append(longest_span_synonym)
    for span in black_lst:
        del common_np_to_group_members_indices[span]


def create_dict_from_common_np_to_group_members_indices(span_to_group_members, dict_span_to_rank,
                                                        dict_longest_span_to_his_synonyms, dict_span_to_similar_spans):
    common_np_to_group_members_indices = {k: v for k, v in
                                          sorted(span_to_group_members.items(), key=lambda item: len(item[1]),
                                                 reverse=True)}
    common_np_to_group_members_indices = {k: v for k, v in common_np_to_group_members_indices.items() if
                                          len(v) > 1 and dict_span_to_rank[k] >= 2}

    combine_similar_longest_np_with_common_sub_nps(common_np_to_group_members_indices,
                                                   dict_longest_span_to_his_synonyms, dict_span_to_similar_spans)
    return common_np_to_group_members_indices


def create_data_dicts_for_combine_synonyms(label_to_nps_collection, dict_label_to_longest_nps_group, dict_span_to_lst):
    span_to_group_members = {}
    dict_longest_span_to_his_synonyms = {}
    for idx, spans_lst in label_to_nps_collection.items():
        for span in spans_lst:
            span_to_group_members[span] = span_to_group_members.get(span, set())
            span_to_group_members[span].add(idx)
        longest_nps_lst = dict_label_to_longest_nps_group[idx]
        for longest_np in longest_nps_lst:
            dict_longest_span_to_his_synonyms[longest_np] = longest_nps_lst
    span_to_group_members = {k: v for k, v in
                             sorted(span_to_group_members.items(), key=lambda item: len(item[1]),
                                    reverse=True)}
    dict_length_to_span = ut.create_dicts_length_to_span_and_span_to_list(span_to_group_members,
                                                                          dict_span_to_lst)
    return span_to_group_members, dict_longest_span_to_his_synonyms, dict_length_to_span


def union_common_np_by_DL_model(common_np_to_group_members_indices, dict_span_to_similar_spans):
    if len(common_np_to_group_members_indices.keys()) < 2:
        return common_np_to_group_members_indices
    weighted_average_vector_lst = []
    for span in common_np_to_group_members_indices.keys():
        spans_embeddings = model.encode(list(dict_span_to_similar_spans[span]))
        weighted_average_vector = ut.get_weighted_average_vector_of_some_vectors_embeddings(spans_embeddings,
                                                                                            dict_span_to_similar_spans[
                                                                                                span])
        weighted_average_vector_lst.append(weighted_average_vector)
    clustering = AgglomerativeClustering(distance_threshold=0.08, n_clusters=None, linkage="single",
                                         affinity="cosine", compute_full_tree=True).fit(
        torch.stack(weighted_average_vector_lst, dim=0))
    dict_cluster_to_common_spans_lst = {}
    for idx, label in enumerate(clustering.labels_):
        dict_cluster_to_common_spans_lst[label] = dict_cluster_to_common_spans_lst.get(label, [])
        dict_cluster_to_common_spans_lst[label].append(list(common_np_to_group_members_indices.keys())[idx])
    new_common_np_to_group_members_indices = {}
    for label, similar_common_spans_lst in dict_cluster_to_common_spans_lst.items():
        if len(similar_common_spans_lst) == 1:
            continue
        new_common_np_to_group_members_indices[similar_common_spans_lst[0]] = set()
        all_similar_common_spans = set()
        for span in similar_common_spans_lst:
            all_similar_common_spans.update(dict_span_to_similar_spans[span])
        for span in similar_common_spans_lst:
            dict_span_to_similar_spans[span] = all_similar_common_spans
            new_common_np_to_group_members_indices[similar_common_spans_lst[0]].update(
                common_np_to_group_members_indices[span])
    return new_common_np_to_group_members_indices


def create_dict_span_to_similar_spans_for_all(dict_span_to_similar_spans_partially):
    dict_span_to_similar_spans = {}
    for _, synonyms in dict_span_to_similar_spans_partially.items():
        for span in synonyms:
            dict_span_to_similar_spans[span] = synonyms
    return dict_span_to_similar_spans


def combine_similar_spans(span_to_group_members, dict_length_to_span,
                          dict_longest_span_to_his_synonyms):
    dict_span_to_similar_spans = {}
    for idx, spans in dict_length_to_span.items():
        dict_span_to_similar_spans.update(find_similarity_in_same_length_group(spans))
    span_to_group_members_new = {}
    for span, sub_set in dict_span_to_similar_spans.items():
        span_to_group_members_new[span] = set()
        for synonym in sub_set:
            span_to_group_members_new[span].update(span_to_group_members[synonym])
    for span, synonyms in dict_span_to_similar_spans.items():
        synonyms_intersect = synonyms.intersection(set(dict_longest_span_to_his_synonyms.keys()))
        for synonym in synonyms_intersect:
            synonyms.update(dict_longest_span_to_his_synonyms[synonym])
    dict_span_to_similar_spans = create_dict_span_to_similar_spans_for_all(dict_span_to_similar_spans)
    return span_to_group_members_new, dict_span_to_similar_spans


def union_common_np(label_to_nps_collection, dict_span_to_rank, dict_label_to_longest_nps_group, dict_span_to_lst):
    span_to_group_members, dict_longest_span_to_his_synonyms, dict_length_to_span = \
        create_data_dicts_for_combine_synonyms(label_to_nps_collection, dict_label_to_longest_nps_group, dict_span_to_lst)
    span_to_group_members, dict_span_to_similar_spans = combine_similar_spans(span_to_group_members,
                                                                              dict_length_to_span,
                                                                              dict_longest_span_to_his_synonyms)
    common_np_to_group_members_indices = \
        create_dict_from_common_np_to_group_members_indices(span_to_group_members,
                                                            dict_span_to_rank, dict_longest_span_to_his_synonyms,
                                                            dict_span_to_similar_spans)
    # common_np_to_group_members_indices = union_common_np_by_DL_model(common_np_to_group_members_indices,
    #                                                                  dict_span_to_similar_spans)
    return common_np_to_group_members_indices, dict_span_to_similar_spans


def combine_non_clustered_spans_in_clustered_spans(dict_label_to_longest_nps_group,
                                                   clustered_spans,
                                                   common_np_to_group_members_indices, common_span_lst,
                                                   dict_span_to_lst):
    for label, lst_spans in dict_label_to_longest_nps_group.items():
        for common_span in common_span_lst:
            if label in common_np_to_group_members_indices[common_span]:
                continue
            synonym_span_lst = clustered_spans[common_span]
            for span in synonym_span_lst:
                is_contained = False
                for not_clustered_span in lst_spans:
                    if len(dict_span_to_lst[span]) < len(dict_span_to_lst[not_clustered_span]):
                        if ut.is_similar_meaning_between_span(dict_span_to_lst[span], dict_span_to_lst[not_clustered_span]):
                            is_contained = True
                            break
                if is_contained:
                    common_np_to_group_members_indices[common_span].add(label)
                    break


def union_nps(label_to_nps_collection, dict_span_to_rank, dict_label_to_longest_nps_group, dict_span_to_lst):
    common_np_to_group_members_indices, dict_span_to_similar_spans = union_common_np(
        label_to_nps_collection, dict_span_to_rank, dict_label_to_longest_nps_group, dict_span_to_lst)
    # dict_label_to_longest_np_without_common_sub_np, common_span_lst = ut.get_non_clustered_group_numbers(
    #     common_np_to_group_members_indices,
    #     dict_label_to_longest_nps_group)
    common_span_lst = list(common_np_to_group_members_indices.keys())
    dict_label_to_longest_np_without_common_sub_np = dict_label_to_longest_nps_group
    combine_non_clustered_spans_in_clustered_spans(dict_label_to_longest_nps_group,
                                                   dict_span_to_similar_spans,
                                                   common_np_to_group_members_indices, common_span_lst,
                                                   dict_span_to_lst)
    dict_score_to_collection_of_sub_groups = ut.get_dict_spans_group_to_score(common_np_to_group_members_indices,
                                                                              dict_span_to_rank,
                                                                              dict_span_to_similar_spans,
                                                                              dict_label_to_longest_nps_group)
    return dict_score_to_collection_of_sub_groups, dict_span_to_similar_spans


def group_hierarchical_clustering_results(clustering, dict_idx_to_all_valid_expansions, dict_idx_to_longest_np,
                                          global_longest_np_index, global_index_to_similar_longest_np,
                                          longest_NP_to_global_index):
    label_to_nps_collection = {}
    dict_label_to_longest_nps_group = {}
    dict_label_to_global_index = {}
    for label in set(clustering.labels_):
        dict_label_to_global_index[int(label)] = global_longest_np_index[0]
        global_longest_np_index[0] += 1
    for idx, label in enumerate(clustering.labels_):
        global_val_label = dict_label_to_global_index[int(label)]
        #longest np clustering
        global_index_to_similar_longest_np[global_val_label] = global_index_to_similar_longest_np.get(global_val_label,
                                                                                                      set())
        global_index_to_similar_longest_np[global_val_label].add(dict_idx_to_longest_np[idx])
        dict_label_to_longest_nps_group[global_val_label] = dict_label_to_longest_nps_group.get(global_val_label, set())
        dict_label_to_longest_nps_group[global_val_label].add(dict_idx_to_longest_np[idx])
        longest_NP_to_global_index[dict_idx_to_longest_np[idx]] = global_val_label
        #Collection of nps
        label_to_nps_collection[global_val_label] = label_to_nps_collection.get(global_val_label, set())
        label_to_nps_collection[global_val_label].update(dict_idx_to_all_valid_expansions[idx])
    return label_to_nps_collection, dict_label_to_longest_nps_group


def create_clusters_of_longest_nps(longest_np_lst, dict_idx_to_all_valid_expansions, dict_idx_to_longest_np,
                                   global_longest_np_index, global_index_to_similar_longest_np,
                                   longest_NP_to_global_index, dict_uncounted_expansions, dict_counted_longest_answers):
    label_to_nps_collection = {}
    if len(longest_np_lst) == 0:
        dict_label_to_longest_nps_group = {}
    elif len(longest_np_lst) == 1:
        longest_NP_to_global_index[longest_np_lst[0]] = global_longest_np_index[0]
        global_index_to_similar_longest_np[global_longest_np_index[0]] = [longest_np_lst[0]]
        dict_label_to_longest_nps_group = {global_longest_np_index[0]:
                                         [longest_np_lst[0]]}
        global_longest_np_index[0] += 1
        # dict_span_to_similar_spans = {longest_np_lst[0]: longest_np_lst[0]}
    else:
        phrase_embeddings = model.encode(longest_np_lst)
        clustering = AgglomerativeClustering(distance_threshold=0.08, n_clusters=None, linkage="average",
                                             affinity="cosine", compute_full_tree=True).fit(phrase_embeddings)
        label_to_nps_collection, dict_label_to_longest_nps_group = group_hierarchical_clustering_results(
            clustering, dict_idx_to_all_valid_expansions, dict_idx_to_longest_np, global_longest_np_index, global_index_to_similar_longest_np,
            longest_NP_to_global_index)
        dict_label_to_longest_nps_group = {k: v for k, v in
                                     sorted(dict_label_to_longest_nps_group.items(), key=lambda item: len(item[1]),
                                            reverse=True)}
    if dict_uncounted_expansions:
        label_to_nps_collection.update(dict_uncounted_expansions)
        dict_label_to_longest_nps_group.update(dict_counted_longest_answers)
    return label_to_nps_collection, dict_label_to_longest_nps_group
