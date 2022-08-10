from sentence_transformers import SentenceTransformer
import torch
from torch import nn
from sklearn.cluster import AgglomerativeClustering
from combine_spans import utils as combine_spans_utils
from combine_spans import span_comparison
import combine_spans.hierarchical_structure_algorithms as hierarchical_structure_algorithms
import nltk
from nltk.tokenize import word_tokenize
from matplotlib import pyplot as plt
import networkx as nx
import collections
import graphviz

g1 = nx.DiGraph()

dict_of_topics, dict_of_span_to_counter, dict_word_to_lemma = combine_spans_utils.load_data_dicts()
dict_lemma_to_synonyms = combine_spans_utils.create_dicts_for_words_similarity(dict_word_to_lemma)


class NP:
    def __init__(self, np, label_lst):
        self.np_val, self.np = nps_lst_to_string(np)
        self.label_lst = label_lst
        self.children = []
        self.marginal_val = 0.0

    def add_children(self, children):
        self.children.extend(children)

    def __gt__(self, ob2):
        return self.marginal_val < ob2.marginal_val


def nps_lst_to_string(nps_lst):
    new_nps_str = []
    np_lst = []
    for np in nps_lst:
        new_nps_str.append(np[0])
        np_lst.append(np[1])
    return new_nps_str, np_lst


def create_dicts_length_to_span_and_span_to_list(span_to_group_members, dict_span_to_lst):
    dict_length_to_span = {}
    for span, sub_set in span_to_group_members.items():
        span_as_lst = dict_span_to_lst[span]
        if len(span_as_lst) == 1:
            continue
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
            span_to_group_members_new[span].update(span_to_group_members[union_span])
    return span_to_group_members_new


def group_hierarchical_clustering_results(clustering, dict_idx_to_all_valid_expansions):
    cluster_index_to_local_indices = {}
    label_to_cluster = {}
    dict_label_to_spans_group = {}
    for idx, label in enumerate(clustering.labels_):
        label_to_cluster[int(label)] = label_to_cluster.get(int(label), [])
        label_to_cluster[int(label)].extend(dict_idx_to_all_valid_expansions[idx])
        dict_label_to_spans_group[int(label)] = dict_label_to_spans_group.get(int(label), [])
        dict_label_to_spans_group[int(label)].append(dict_idx_to_all_valid_expansions[idx][0])

        cluster_index_to_local_indices[int(label)] = cluster_index_to_local_indices.get(int(label), [])
        cluster_index_to_local_indices[int(label)].append(idx)
    return label_to_cluster, dict_label_to_spans_group, cluster_index_to_local_indices


def initialize_spans_data(all_nps_example_lst, dict_span_to_rank):
    dict_idx_to_all_valid_expansions = {}
    idx = 0
    for phrase in all_nps_example_lst:
        dict_idx_to_all_valid_expansions[idx] = []
        for span in phrase:
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
    if np_object.np_val == main_np_object.np_val:
        return True
    for np in main_np_object.np:
        if span_comparison.is_similar_meaning_between_span(np_object.np[0], np, dict_word_to_lemma,
                                                           dict_lemma_to_synonyms):
            for child in main_np_object.children:
                is_added = add_NP_to_DAG(child, np_object)
            if not is_added:
                main_np_object.add_children([np_object])
            return True
    return False


def get_global_index(cluster_index_to_local_indices, longest_np_to_index, longest_np_lst, label_lst):
    global_indices = []
    for label in label_lst:
        local_indices = cluster_index_to_local_indices[label]
        for index in local_indices:
            longest_np = longest_np_lst[index]
            global_indices.append(longest_np_to_index[longest_np])
    return global_indices


def create_DAG(dict_score_to_collection_of_sub_groups, dict_label_to_spans_group, dict_span_to_lemmas_lst,
               cluster_index_to_local_indices, longest_np_to_index, longest_np_lst, all_object_np_lst, span_to_object):
    np_object_lst = []
    dict_label_to_np_object = {}
    for label, nps in dict_label_to_spans_group.items():
        global_indices = get_global_index(cluster_index_to_local_indices, longest_np_to_index, longest_np_lst, [label])
        np_object = NP(nps, global_indices)
        for np in np_object.np_val:
            span_to_object[np] = np_object
        all_object_np_lst.append(np_object)
        np_object_lst.append(np_object)
        dict_label_to_np_object[label] = np_object
    # for label, np_object in dict_label_to_np_object.items():
    #     np_object.np_val += " leaf"
    for score, np_to_labels_collection in dict_score_to_collection_of_sub_groups.items():
        for np, labels in np_to_labels_collection:
            global_indices = get_global_index(cluster_index_to_local_indices, longest_np_to_index, longest_np_lst,
                                              labels)
            np_object = NP([(np, dict_span_to_lemmas_lst[np])], global_indices)
            all_object_np_lst.append(np_object)
            for label in labels:
                main_np_object = dict_label_to_np_object[label]
                add_NP_to_DAG(main_np_object, np_object)
    return np_object_lst, dict_label_to_np_object


def bfs(root):
    visited, queue = set(), collections.deque(root)
    for np_element in root:
        visited.add(np_element)
    relation_lst = set()
    while queue:
        # Dequeue a vertex from queue
        vertex = queue.popleft()
        print(str(vertex) + " ", end="")
        # If not visited, mark it as visited, and
        # enqueue it
        for neighbour in vertex.children:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)
            edge = (vertex.np_val, neighbour.np_val)
            if edge not in relation_lst:
                relation_lst.add(edge)
    return relation_lst


def visualize_dag(dag):
    relation_lst = bfs(dag)
    g = graphviz.Digraph('G', filename='data_DAG.gv')
    for relation in relation_lst:
        g.edge(relation[1], relation[0])
    g.view()


def combine_similar_longest_np_with_common_sub_nps(common_np_to_group_members_indices,
                                                   dict_longest_span_to_his_synonyms):
    black_lst = []
    for span, indices_group in common_np_to_group_members_indices.items():
        if span in black_lst:
            continue
        if span in dict_longest_span_to_his_synonyms.keys():
            for longest_span_synonym in dict_longest_span_to_his_synonyms[span]:
                if longest_span_synonym != span and longest_span_synonym in common_np_to_group_members_indices.keys():
                    common_np_to_group_members_indices[span].update(
                        common_np_to_group_members_indices[longest_span_synonym])
                    if longest_span_synonym not in black_lst:
                        black_lst.append(longest_span_synonym)
    for span in black_lst:
        del common_np_to_group_members_indices[span]


def create_dict_from_common_np_to_group_members_indices(span_to_group_members, dict_span_to_rank,
                                                        dict_longest_span_to_his_synonyms):
    common_np_to_group_members_indices = {k: v for k, v in
                                          sorted(span_to_group_members.items(), key=lambda item: len(item[1]),
                                                 reverse=True)}
    common_np_to_group_members_indices = {k: v for k, v in common_np_to_group_members_indices.items() if
                                          len(v) > 1 and dict_span_to_rank[k] >= 2}

    combine_similar_longest_np_with_common_sub_nps(common_np_to_group_members_indices,
                                                   dict_longest_span_to_his_synonyms)
    return common_np_to_group_members_indices


def create_date_dicts_for_combine_synonyms(clusters, dict_label_to_spans_group):
    span_to_group_members = {}
    dict_span_to_lemmas_lst = {}
    dict_longest_span_to_his_synonyms = {}
    for idx, span_tuple_lst in clusters.items():
        for span in span_tuple_lst:
            span_to_group_members[span[0]] = span_to_group_members.get(span[0], set())
            span_to_group_members[span[0]].add(idx)
            dict_span_to_lemmas_lst[span[0]] = span[1]
        longest_np_tuple_lst = dict_label_to_spans_group[idx]
        collection_spans = [longest_np_tuple[0] for longest_np_tuple in longest_np_tuple_lst]
        for longest_np_tuple in longest_np_tuple_lst:
            dict_longest_span_to_his_synonyms[longest_np_tuple[0]] = collection_spans
    span_to_group_members = {k: v for k, v in
                             sorted(span_to_group_members.items(), key=lambda item: len(item[1]),
                                    reverse=True)}
    dict_length_to_span = create_dicts_length_to_span_and_span_to_list(span_to_group_members, dict_span_to_lemmas_lst)
    return span_to_group_members, dict_span_to_lemmas_lst, dict_longest_span_to_his_synonyms, dict_length_to_span


def union_common_np(clusters, dict_word_to_lemma, dict_lemma_to_synonyms, dict_span_to_rank, dict_label_to_spans_group):
    span_to_group_members, dict_span_to_lemmas_lst, dict_longest_span_to_his_synonyms, dict_length_to_span = create_date_dicts_for_combine_synonyms(
        clusters, dict_label_to_spans_group)
    span_to_group_members = combine_similar_spans(span_to_group_members, dict_length_to_span, dict_word_to_lemma,
                                                  dict_lemma_to_synonyms)
    common_np_to_group_members_indices = create_dict_from_common_np_to_group_members_indices(span_to_group_members,
                                                                                             dict_span_to_rank,
                                                                                             dict_longest_span_to_his_synonyms)
    return dict_span_to_lemmas_lst, common_np_to_group_members_indices


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
    dict_label_to_longest_np_without_common_sub_np = {}
    for num in res_group_numbers:
        dict_label_to_longest_np_without_common_sub_np[num] = dict_label_to_spans_group[num]
    return dict_label_to_longest_np_without_common_sub_np, dict_sub_string_to_spans


def get_common_sub_np_to_his_synonyms_dict_and_longest_nps_without_common_sub_np(label_to_cluster,
                                                                                 span_to_group_members,
                                                                                 dict_label_to_spans_group):
    dict_label_to_longest_np_without_common_sub_np, dict_sub_string_to_spans = get_non_clustered_group_numbers(
        label_to_cluster,
        span_to_group_members,
        dict_label_to_spans_group)
    return dict_sub_string_to_spans, dict_label_to_longest_np_without_common_sub_np


def add_new_objects_to_global_dict_index_to_object(np_object_lst, global_dict_label_to_object):
    for np_object in np_object_lst:
        for label in np_object.label_lst:
            global_dict_label_to_object[label] = np_object


def get_global_indices_and_objects_from_longest_np_lst(longest_np_to_index, longest_np_lst, dict_label_to_np_object):
    label_lst = set()
    object_lst = set()
    for longest_np in longest_np_lst:
        label = longest_np_to_index[longest_np]
        label_lst.add(label)
        object_lst.add(dict_label_to_np_object[label])
    return label_lst, object_lst


def add_topic_object(longest_np_to_index, longest_np_total_lst, global_dict_label_to_object, topic, topic_object_lst):
    label_lst, longest_np_object_lst = get_global_indices_and_objects_from_longest_np_lst(longest_np_to_index,
                                                                                          longest_np_total_lst,
                                                                                          global_dict_label_to_object)
    topic_object = NP([(topic, [topic])], label_lst)
    for longest_np_object in longest_np_object_lst:
        add_NP_to_DAG(longest_np_object, topic_object)
    topic_object_lst.append(topic_object)


def change_DAG_direction(global_np_object_lst, visited=[]):
    for np_object in global_np_object_lst:
        if np_object in visited:
            continue
        visited.append(np_object)
        child_np_object_remove_lst = []
        for child_np_object in np_object.children:
            child_np_object_remove_lst.append(child_np_object)
        if np_object.children:
            change_DAG_direction(np_object.children, visited)
            for child_np_object_to_remove in child_np_object_remove_lst:
                child_np_object_to_remove.children.append(np_object)
                np_object.children.remove(child_np_object_to_remove)


def union_nps(label_to_cluster, dict_span_to_rank, dict_label_to_spans_group):
    dict_span_to_lst, common_np_to_group_members_indices = union_common_np(
        label_to_cluster, dict_word_to_lemma,
        dict_lemma_to_synonyms,
        dict_span_to_rank, dict_label_to_spans_group)
    dict_sub_string_to_spans, dict_label_to_longest_np_without_common_sub_np = get_common_sub_np_to_his_synonyms_dict_and_longest_nps_without_common_sub_np(
        label_to_cluster,
        common_np_to_group_members_indices,
        dict_label_to_spans_group)
    span_comparison.combine_not_clustered_spans_in_clustered_spans(dict_label_to_longest_np_without_common_sub_np,
                                                                   dict_sub_string_to_spans,
                                                                   dict_word_to_lemma,
                                                                   dict_lemma_to_synonyms,
                                                                   common_np_to_group_members_indices)
    dict_score_to_collection_of_sub_groups = create_score_to_group_dict(common_np_to_group_members_indices,
                                                                        dict_span_to_rank)
    return dict_score_to_collection_of_sub_groups, dict_span_to_lst


def main():
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    word_to_cluster = {}
    dict_span_to_rank = {}
    longest_np_to_index = {}
    global_dict_label_to_object = {}
    global_np_object_lst = []
    topic_object_lst = []
    all_object_np_lst = []
    counter = 0
    span_to_object = {}
    for topic, example_list in dict_of_topics.items():
        dict_span_to_all_valid_expansions = {}
        longest_np_lst = []
        longest_np_total_lst = []
        all_nps_example_lst = []
        for example in example_list:
            longest_np_total_lst.append(example[0])
            if example[0] in longest_np_to_index.keys():
                continue
            dict_span_to_all_valid_expansions[example[0]] = [item[0] for item in example[1]]
            longest_np_lst.append(example[0])
            all_nps_example_lst.append(example[1])
            longest_np_to_index[example[0]] = len(longest_np_to_index.keys())
        if len(longest_np_total_lst) == 0 or len(longest_np_lst) == 0:
            continue
        counter += 1
        dict_idx_to_all_valid_expansions = initialize_spans_data(all_nps_example_lst, dict_span_to_rank)
        if len(longest_np_lst) == 1:
            np_object = NP(dict_idx_to_all_valid_expansions[0], [longest_np_to_index[longest_np_lst[0]]])
            np_object_lst = [np_object]
            all_object_np_lst.append(np_object)
            add_new_objects_to_global_dict_index_to_object(np_object_lst, global_dict_label_to_object)
            add_topic_object(longest_np_to_index, longest_np_total_lst, global_dict_label_to_object, topic,
                             topic_object_lst)
            continue
        # if len(longest_np_lst) > 1:
        phrase_embeddings = model.encode(longest_np_lst)
        clustering = AgglomerativeClustering(distance_threshold=0.08, n_clusters=None, linkage="average",
                                             affinity="cosine", compute_full_tree=True).fit(phrase_embeddings)
        label_to_cluster, dict_label_to_spans_group, cluster_index_to_local_indices = group_hierarchical_clustering_results(
            clustering, dict_idx_to_all_valid_expansions)
        dict_label_to_spans_group = {k: v for k, v in
                                     sorted(dict_label_to_spans_group.items(), key=lambda item: len(item[1]),
                                            reverse=True)}
        # else:
        #     word_to_cluster[longest_np_lst[0]] = []
        #     continue
        dict_score_to_collection_of_sub_groups, dict_span_to_lst = union_nps(label_to_cluster, dict_span_to_rank,
                                                                             dict_label_to_spans_group)
        # span_to_group_members = set_cover_with_priority(dict_score_to_collection_of_sub_groups)
        np_object_lst, dict_label_to_np_object = create_DAG(dict_score_to_collection_of_sub_groups,
                                                            dict_label_to_spans_group,
                                                            dict_span_to_lst, cluster_index_to_local_indices,
                                                            longest_np_to_index, longest_np_lst, all_object_np_lst,
                                                            span_to_object)
        add_new_objects_to_global_dict_index_to_object(np_object_lst, global_dict_label_to_object)
        add_topic_object(longest_np_to_index, longest_np_total_lst, global_dict_label_to_object, topic,
                         topic_object_lst)
        global_np_object_lst.extend(np_object_lst)
        # visualize_dag(np_object_lst)
        # span_to_group_members = set_cover_with_priority(dict_score_to_collection_of_sub_groups)
        # dict_sub_string_to_spans.update(dict_sub_string)
        # word_to_cluster[key] = dict_sub_string_to_spans
    change_DAG_direction(global_np_object_lst)
    hierarchical_structure_algorithms.greedy_algorithm(80, topic_object_lst, all_object_np_lst)
    print(word_to_cluster)


main()
