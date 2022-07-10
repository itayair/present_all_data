from sentence_transformers import SentenceTransformer
import torch
from torch import nn
from sklearn.cluster import AgglomerativeClustering
from combine_spans import utils as combine_spans_utils
from combine_spans import span_comparison
import nltk
from nltk.tokenize import word_tokenize

dict_of_topics, dict_of_span_to_counter, dict_word_to_lemma = combine_spans_utils.load_data_dicts()
dict_lemma_to_synonyms = combine_spans_utils.create_dicts_for_words_similarity(dict_word_to_lemma)


class NP:
    def __init__(self, np, label_lst):
        self.np = np
        self.label_lst = label_lst
        self.children = []

    def add_children(self, children):
        self.children.extend(children)


def create_dicts_length_to_span_and_span_to_list(span_to_group_members, dict_span_to_lst):
    dict_length_to_span = {}
    # dict_span_to_lst = {}
    for span, sub_set in span_to_group_members.items():
        # span_as_lst = span.replace(",", "")
        # span_as_lst = span_as_lst.replace(".", "")
        # span_as_lst = span_as_lst.split()
        span_as_lst = dict_span_to_lst[span]
        dict_length_to_span[len(span_as_lst)] = dict_length_to_span.get(len(span_as_lst), [])
        dict_length_to_span[len(span_as_lst)].append((span, span_as_lst))
        dict_span_to_lst[span] = span_as_lst
    return dict_length_to_span


def combine_similar_spans(span_to_group_members, dict_length_to_span, dict_word_to_lemma, dict_lemma_to_synonyms):
    dict_spans = {}
    for idx, spans in dict_length_to_span.items():
        if idx == 1:
            continue
        dict_spans.update(
            span_comparison.find_similarity_in_same_length_group(spans, dict_word_to_lemma, dict_lemma_to_synonyms))
    span_to_group_members_new = {}
    for span, sub_set in dict_spans.items():
        span_to_group_members_new[span] = span_to_group_members[span]
        for union_span in sub_set:
            span_to_group_members_new[span].extend(span_to_group_members[union_span])
        span_to_group_members_new[span] = list(set(span_to_group_members_new[span]))
    return span_to_group_members_new


def group_agglomerative_clustering_results(clustering, dict_idx_to_all_valid_expansions):
    label_to_cluster = {}
    dict_label_to_spans_group = {}
    for idx, label in enumerate(clustering.labels_):
        label_to_cluster[int(label)] = label_to_cluster.get(int(label), [])
        label_to_cluster[int(label)].extend(dict_idx_to_all_valid_expansions[idx])
        dict_label_to_spans_group[int(label)] = dict_label_to_spans_group.get(int(label), [])
        dict_label_to_spans_group[int(label)].append(dict_idx_to_all_valid_expansions[idx][0])
    span_to_same_meaning_cluster_of_spans = {}
    # for label, cluster in label_to_cluster.items():
    #     most_frequent_span = combine_spans_utils.get_most_frequent_span(cluster, dict_of_span_to_counter)
    #     span_to_same_meaning_cluster_of_spans[most_frequent_span] = cluster
    return label_to_cluster, dict_label_to_spans_group


def initialize_spans_data(example_list, dict_span_to_rank):
    dict_idx_to_all_valid_expansions = {}
    idx = 0
    for phrase in example_list:
        dict_idx_to_all_valid_expansions[idx] = []
        for span in phrase[1]:
            dict_span_to_rank[span[0]] = span[1]
            dict_idx_to_all_valid_expansions[idx].append((span[0], span[2]))
        idx += 1
    return dict_idx_to_all_valid_expansions


def from_dict_to_lst(label_to_cluster):
    for idx, valid_expansions in label_to_cluster.items():
        label_to_cluster[idx] = list(set(valid_expansions))
    return label_to_cluster


def set_cover(universe, subsets):
    """Find a family of subsets that covers the universal set"""
    elements = set(e for s in subsets for e in s)
    # Check the subsets cover the universe
    if elements != universe:
        return None
    covered = set()
    cover = []
    # Greedily add the subsets with the most uncovered points
    while covered != elements:
        subset = max(subsets, key=lambda s: len(s - covered))
        cover.append(subset)
        covered |= subset

    return cover


def create_score_to_group_dict(span_to_group_members, dict_span_to_rank):
    dict_score_to_collection_of_sub_groups = {}
    for key, group in span_to_group_members.items():
        dict_score_to_collection_of_sub_groups[dict_span_to_rank[key]] = dict_score_to_collection_of_sub_groups.get(
            dict_span_to_rank[key], [])
        dict_score_to_collection_of_sub_groups[dict_span_to_rank[key]].append((key, set(group)))
    dict_score_to_collection_of_sub_groups = {k: v for k, v in
                                              sorted(dict_score_to_collection_of_sub_groups.items(),
                                                     key=lambda item: item[0],
                                                     reverse=True)}
    for score, sub_group_lst in dict_score_to_collection_of_sub_groups.items():
        sub_group_lst.sort(key=lambda tup: len(tup[1]), reverse=True)
    return dict_score_to_collection_of_sub_groups


def set_cover_with_priority(dict_score_to_collection_of_sub_groups):
    covered = set()
    span_to_group_members = {}
    for score, sub_group_lst in dict_score_to_collection_of_sub_groups.items():
        while True:
            subset = max(sub_group_lst, key=lambda s: len(s[1] - covered))
            if len(subset[1] - covered) > 1:
                span_to_group_members[subset[0]] = list(subset[1])
                covered.update(subset[1])
            else:
                break
    return span_to_group_members


def add_NP_to_DAG(main_np_object, np_object):
    is_added = False
    for np in main_np_object.np:
        if np_object.np[0][0] == np[0]:
            return True
        if span_comparison.is_similar_meaning_between_span(np_object.np[0][1], np[1], dict_word_to_lemma,
                                                           dict_lemma_to_synonyms):
            for child in main_np_object.children:
                is_added = add_NP_to_DAG(child, np_object)
            if not is_added:
                main_np_object.add_children([np_object])
            return True
    return False


def create_DAG(dict_score_to_collection_of_sub_groups, dict_label_to_spans_group, dict_span_to_lemmas_lst):
    np_object_lst = []
    dict_label_to_np_object = {}
    for label, nps in dict_label_to_spans_group.items():
        # lst_strings = []
        # for np in nps:
        #     lst_strings.append(np[0])
        np_object = NP(nps, [label])
        np_object_lst.append(np_object)
        dict_label_to_np_object[label] = np_object
    for score, np_to_labels_collection in dict_score_to_collection_of_sub_groups.items():
        for np, labels in np_to_labels_collection:
            np_object = NP([(np, dict_span_to_lemmas_lst[np])], labels)
            for label in labels:
                main_np_object = dict_label_to_np_object[label]
                is_added = add_NP_to_DAG(main_np_object, np_object)
                if not is_added:
                    print("main NP:", main_np_object.np, "sub NP", np)
    return np_object_lst


def union_groups(clusters, dict_word_to_lemma, dict_lemma_to_synonyms, dict_span_to_rank, dict_label_to_spans_group):
    span_to_group_members = {}
    dict_span_to_lemmas_lst = {}
    for idx, valid_expansions_idx in clusters.items():
        # valid_expansions_lst = clusters[valid_expansions_idx]
        for span in valid_expansions_idx:
            span_to_group_members[span[0]] = span_to_group_members.get(span[0], [])
            span_to_group_members[span[0]].append(idx)
            dict_span_to_lemmas_lst[span[0]] = span[1]
    span_to_group_members = {k: v for k, v in
                             sorted(span_to_group_members.items(), key=lambda item: len(item[1]),
                                    reverse=True)}
    dict_length_to_span = create_dicts_length_to_span_and_span_to_list(span_to_group_members, dict_span_to_lemmas_lst)
    span_to_group_members = combine_similar_spans(span_to_group_members, dict_length_to_span, dict_word_to_lemma,
                                                  dict_lemma_to_synonyms)
    span_to_group_members = {k: v for k, v in
                             sorted(span_to_group_members.items(), key=lambda item: len(item[1]),
                                    reverse=True)}
    span_to_group_members_more_than_1_element = {k: v for k, v in span_to_group_members.items() if
                                                 len(v) > 1 and dict_span_to_rank[k] >= 2}
    dict_score_to_collection_of_sub_groups = create_score_to_group_dict(span_to_group_members_more_than_1_element,
                                                                        dict_span_to_rank)
    np_object_lst = create_DAG(dict_score_to_collection_of_sub_groups, dict_label_to_spans_group, dict_span_to_lemmas_lst)
    span_to_group_members = set_cover_with_priority(dict_score_to_collection_of_sub_groups)
    return span_to_group_members, dict_span_to_lemmas_lst


def get_non_clustered_group_numbers(label_to_cluster, span_to_group_members, dict_label_to_spans_group):
    dict_sub_string_to_spans = {}
    all_group_numbers = list(range(0, len(label_to_cluster.keys())))
    already_grouped = []
    for sub_string, group_numbers in span_to_group_members.items():
        already_grouped.extend(group_numbers)
        dict_sub_string_to_spans[sub_string] = []
        for num in group_numbers:
            dict_sub_string_to_spans[sub_string].extend(dict_label_to_spans_group[num])
    res_group_numbers = [item for item in all_group_numbers if item not in already_grouped]
    return res_group_numbers, dict_sub_string_to_spans


def combine_clustered_and_non_clustered(label_to_cluster, span_to_group_members, dict_label_to_spans_group):
    res_group_numbers, dict_sub_string_to_spans = get_non_clustered_group_numbers(label_to_cluster,
                                                                                  span_to_group_members,
                                                                                  dict_label_to_spans_group)
    dict_sub_string = {}
    # check if the clustered spans are contained in spans without match
    for num in res_group_numbers:
        if len(dict_label_to_spans_group[num]) > 1:
            dict_sub_string[dict_label_to_spans_group[num][0]] = dict_label_to_spans_group[num]
        else:
            dict_sub_string[dict_label_to_spans_group[num][0]] = []
    return dict_sub_string_to_spans, dict_sub_string


def main():
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    word_to_cluster = {}
    dict_span_to_rank = {}
    for key, example_list in dict_of_topics.items():
        dict_span_to_all_valid_expansions = {}
        phrase_list = []
        for example in example_list:
            dict_span_to_all_valid_expansions[example[0]] = [item[0] for item in example[1]]
            phrase_list.append(example[0])
        dict_idx_to_all_valid_expansions = initialize_spans_data(example_list, dict_span_to_rank)
        if len(phrase_list) > 1:
            phrase_embeddings = model.encode(phrase_list)
            clustering = AgglomerativeClustering(distance_threshold=0.08, n_clusters=None, linkage="average",
                                                 affinity="cosine", compute_full_tree=True).fit(phrase_embeddings)
            label_to_cluster, dict_label_to_spans_group = group_agglomerative_clustering_results(
                clustering, dict_idx_to_all_valid_expansions)
            dict_label_to_spans_group = {k: v for k, v in
                                         sorted(dict_label_to_spans_group.items(), key=lambda item: len(item[1]),
                                                reverse=True)}
        else:
            word_to_cluster[phrase_list[0]] = []
            continue
        span_to_group_members, dict_span_to_lst = union_groups(label_to_cluster, dict_word_to_lemma,
                                                               dict_lemma_to_synonyms,
                                                               dict_span_to_rank, dict_label_to_spans_group)
        dict_sub_string_to_spans, dict_sub_string = combine_clustered_and_non_clustered(label_to_cluster,
                                                                                        span_to_group_members,
                                                                                        dict_label_to_spans_group)
        span_comparison.combine_not_clustered_spans_in_clustered_spans(dict_sub_string, dict_sub_string_to_spans,
                                                                       dict_word_to_lemma,
                                                                       dict_lemma_to_synonyms, dict_span_to_lst)
        dict_sub_string_to_spans.update(dict_sub_string)
        word_to_cluster[key] = dict_sub_string_to_spans
    print(word_to_cluster)


main()
