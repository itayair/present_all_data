from sentence_transformers import SentenceTransformer
import torch
from torch import nn
from sklearn.cluster import AgglomerativeClustering
from transformers import AutoTokenizer, AutoModel

from combine_spans import utils as combine_spans_utils
from combine_spans import span_comparison
import combine_spans.hierarchical_structure_algorithms as hierarchical_structure_algorithms
import nltk
from nltk.tokenize import word_tokenize
from matplotlib import pyplot as plt
import networkx as nx
import collections
import graphviz
from networkx.readwrite import json_graph
import json

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
medical_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
# encoded_input = tokenizer("My name is Itay Yair", return_tensors='pt')
# output = model(**encoded_input)
g1 = nx.DiGraph()

# dict_lemma_to_synonyms = combine_spans_utils.create_dicts_for_words_similarity(combine_spans_utils.dict_word_to_lemma)
dict_lemma_to_synonyms = combine_spans_utils.dict_lemma_to_synonyms


class NP:
    def __init__(self, np, label_lst):
        self.np_val, self.np = nps_lst_to_string(np)
        self.label_lst = set(label_lst)
        self.children = []
        self.parents = set()
        self.frequency = 0
        self.score = 0.0
        self.marginal_val = 0.0
        self.combined_nodes_lst = set()
        self.weighted_average_vector = None

    def add_children(self, children):
        for child in children:
            if child not in self.children:
                self.children.append(child)
        for child in children:
            self.label_lst.update(child.label_lst)

    def add_unique_lst(self, span_as_tokens_lst):
        new_np_lst = []
        for span_as_tokens in span_as_tokens_lst:
            is_already_exist = False
            for np in self.np:
                intersection_np = set(np).intersection(set(span_as_tokens))
                if len(intersection_np) == len(span_as_tokens):
                    is_already_exist = True
                    break
            if not is_already_exist:
                new_np_lst.append(span_as_tokens)
        self.np.extend(new_np_lst)

    def update_parents_label(self, np_object, label_lst, visited):
        for parent_object in np_object.parents:
            if parent_object in visited:
                continue
            parent_object.label_lst.update(label_lst)
            visited.add(parent_object)
            self.update_parents_label(parent_object, label_lst, visited)

    def update_children_with_new_parent(self, children, previous_parent):
        for child in children:
            if previous_parent not in child.parents:
                continue
            child.parents.remove(previous_parent)
            child.parents.add(self)

    def update_parents_with_new_node(self, parents, previous_node):
        for parent in parents:
            if self not in parent.children:
                parent.children.append(self)
            parent.children.remove(previous_node)

    def combine_nodes(self, np_object):
        self.np_val.update(np_object.np_val)
        self.add_unique_lst(np_object.np)
        # self.np.extend(np_object.np)
        self.label_lst.update(np_object.label_lst)
        self.update_parents_label(np_object, self.label_lst, set())
        self.update_parents_with_new_node(np_object.parents, np_object)
        self.update_children_with_new_parent(np_object.children, np_object)
        self.parents.update(np_object.parents)
        for child in np_object.children:
            if child not in self.children:
                self.children.append(child)
        self.combined_nodes_lst.add(np_object)

    def __gt__(self, ob2):
        return self.marginal_val < ob2.marginal_val


def nps_lst_to_string(nps_lst):
    new_nps_str = set()
    np_lst = []
    for np in nps_lst:
        new_nps_str.add(np[0])
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


def combine_similar_spans(span_to_group_members, dict_length_to_span, dict_word_to_lemma,
                          dict_lemma_to_synonyms, dict_longest_span_to_his_synonyms):
    dict_spans = {}
    for idx, spans in dict_length_to_span.items():
        dict_spans.update(
            span_comparison.find_similarity_in_same_length_group(spans, dict_word_to_lemma, dict_lemma_to_synonyms))
    span_to_group_members_new = {}
    for span, sub_set in dict_spans.items():
        span_to_group_members_new[span] = span_to_group_members[span]
        for union_span in sub_set:
            span_to_group_members_new[span].update(span_to_group_members[union_span])
    for span, synonyms in dict_spans.items():
        new_synonyms = set()
        for synonym in synonyms:
            if synonym in dict_longest_span_to_his_synonyms.keys():
                new_synonyms.update(dict_longest_span_to_his_synonyms[synonym])
        synonyms.update(new_synonyms)
    # black_list = set()
    # dict_keys_lst = list(dict_spans.keys())
    # for idx, key in enumerate(dict_keys_lst):
    #     if key in black_list:
    #         continue
    #     for key_ref in dict_keys_lst[idx+1:-1]:
    #         if key_ref in black_list:
    #             continue
    #         if dict_spans[key] == dict_spans[key_ref]:
    #             black_list.add(key_ref)
    # for key in black_list:
    #     dict_spans.pop(key, None)
    return span_to_group_members_new, dict_spans


def group_hierarchical_clustering_results(clustering, dict_idx_to_all_valid_expansions, global_longest_np_index,
                                          global_index_to_similar_longest_np):
    cluster_index_to_local_indices = {}
    label_to_cluster = {}
    dict_label_to_spans_group = {}
    dict_label_to_global_index = {}
    for label in set(clustering.labels_):
        dict_label_to_global_index[int(label)] = global_longest_np_index[0]
        global_longest_np_index[0] += 1

    for idx, label in enumerate(clustering.labels_):
        global_val_label = dict_label_to_global_index[int(label)]
        global_index_to_similar_longest_np[global_val_label] = global_index_to_similar_longest_np.get(global_val_label,
                                                                                                      [])
        global_index_to_similar_longest_np[global_val_label].append(dict_idx_to_all_valid_expansions[idx][0][0])
        label_to_cluster[global_val_label] = label_to_cluster.get(global_val_label, [])
        label_to_cluster[global_val_label].extend(dict_idx_to_all_valid_expansions[idx])
        dict_label_to_spans_group[global_val_label] = dict_label_to_spans_group.get(global_val_label, [])
        dict_label_to_spans_group[global_val_label].append(dict_idx_to_all_valid_expansions[idx][0])

        cluster_index_to_local_indices[global_val_label] = cluster_index_to_local_indices.get(global_val_label, [])
        cluster_index_to_local_indices[global_val_label].append(idx)
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


def get_average_value(spans_lst, dict_span_to_rank):
    average_val = 0
    for span in spans_lst:
        average_val += dict_span_to_rank[span]
    average_val = average_val / len(spans_lst)
    return int(average_val)


def create_score_to_group_dict(span_to_group_members, dict_span_to_rank, dict_span_to_similar_spans):
    dict_score_to_collection_of_sub_groups = {}
    for key, group in span_to_group_members.items():
        average_val = get_average_value(dict_span_to_similar_spans[key], dict_span_to_rank)
        dict_score_to_collection_of_sub_groups[average_val] = dict_score_to_collection_of_sub_groups.get(
            average_val, [])
        dict_score_to_collection_of_sub_groups[average_val].append((key, set(group)))
    dict_score_to_collection_of_sub_groups = {k: v for k, v in
                                              sorted(dict_score_to_collection_of_sub_groups.items(),
                                                     key=lambda item: item[0])}
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


def add_NP_to_DAG_up_to_bottom(np_object_to_add, np_object, similar_np_object):
    is_contained = False
    if np_object == np_object_to_add:
        return True
    for np in np_object_to_add.np:
        for np_ref in np_object.np:
            if span_comparison.is_similar_meaning_between_span(np_ref, np, combine_spans_utils.dict_word_to_lemma,
                                                               dict_lemma_to_synonyms):
                is_contained = True
                if len(np_ref) == len(np):
                    np_object.combine_nodes(np_object_to_add)
                    similar_np_object[0] = np_object
                    return True
    if is_contained:
        is_added = False
        for child in np_object.children:
            is_added |= add_NP_to_DAG_up_to_bottom(np_object_to_add, child, similar_np_object)
            if similar_np_object[0]:
                return True
        if not is_added:
            if np_object_to_add not in np_object.children:
                np_object.add_children([np_object_to_add])
            np_object_to_add.parents.add(np_object)
        return True
    return False


def add_NP_to_DAG_bottom_to_up(np_object_to_add, np_object, visited, similar_np_object):
    is_contained = False
    if np_object in visited:
        return False
    if np_object == np_object_to_add:
        return True
    visited.add(np_object)
    for np in np_object_to_add.np:
        for np_ref in np_object.np:
            if span_comparison.is_similar_meaning_between_span(np, np_ref, combine_spans_utils.dict_word_to_lemma,
                                                               dict_lemma_to_synonyms):
                is_contained = True
                if len(np_ref) == len(np):
                    np_object.combine_nodes(np_object_to_add)
                    similar_np_object[0] = np_object
                    return True
    if is_contained:
        is_added = False
        for parent in np_object.parents:
            is_added |= add_NP_to_DAG_bottom_to_up(np_object_to_add, parent, visited, similar_np_object)
            if similar_np_object[0]:
                return True
        if not is_added:
            if np_object not in np_object_to_add.children:
                np_object_to_add.add_children([np_object])
            np_object.parents.add(np_object_to_add)
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


def create_DAG_from_top_to_bottom(dict_score_to_collection_of_sub_groups, topic_synonym_lst,
                                  dict_span_to_lemmas_lst,
                                  all_object_np_lst, span_to_object,
                                  dict_span_to_similar_spans, dict_label_to_spans_group,
                                  global_dict_label_to_object, topic_object_lst):
    topic_synonyms_tuples = [(synonym, [synonym]) for synonym in topic_synonym_lst]
    topic_object = NP(topic_synonyms_tuples, set(dict_label_to_spans_group.keys()))
    topic_object_lst.append(topic_object)
    for score, np_to_labels_collection in dict_score_to_collection_of_sub_groups.items():
        for np_key, labels in np_to_labels_collection:
            np_collection = dict_span_to_similar_spans[np_key]
            for np in np_collection:
                np_object = span_to_object.get(np, None)
                if np_object:
                    break
            if np_object:
                continue
            tuple_np_lst = []
            for np in np_collection:
                span_to_object[np] = np_object
                tuple_np_lst.append((np, dict_span_to_lemmas_lst[np]))
            np_object = NP(tuple_np_lst, labels)
            similar_np_object = [None]
            add_NP_to_DAG_up_to_bottom(np_object, topic_object, similar_np_object)
            if not similar_np_object[0]:
                all_object_np_lst.append(np_object)
    for label, nps in dict_label_to_spans_group.items():
        for np in nps:
            np_object = span_to_object.get(np[0], None)
            if np_object:
                break
        if np_object:
            global_dict_label_to_object[label] = np_object
            continue
        has_single_token = False
        np_object = NP(nps, [label])
        for np in nps:
            if len(np[1]) == 1:
                has_single_token = True
                break
        if has_single_token:
            topic_object.combine_nodes(np_object)
            global_dict_label_to_object[label] = topic_object
            for np in nps:
                span_to_object[np[0]] = topic_object
            continue
        similar_np_object = [None]
        add_NP_to_DAG_up_to_bottom(np_object, topic_object, similar_np_object)
        if similar_np_object[0]:
            np_object = similar_np_object[0]
        else:
            all_object_np_lst.append(np_object)
        global_dict_label_to_object[label] = np_object
        for np in np_object.np_val:
            span_to_object[np] = np_object
    return topic_object


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


def get_weighted_average_vector_of_some_vectors_embeddings(spans_embeddings, dict_of_span_to_counter, common_np_lst):
    weighted_average_vector = torch.zeros(spans_embeddings[0].shape)
    total_occurrences = 0
    for idx, embedding_vector in enumerate(spans_embeddings):
        # total_occurrences += dict_of_span_to_counter[common_np_lst[idx]]
        # weighted_average_vector += embedding_vector * common_np_lst[idx]
        weighted_average_vector += embedding_vector
    # weighted_average_vector /= total_occurrences
    weighted_average_vector /= len(common_np_lst)
    return weighted_average_vector


def union_common_np_by_DL_model(common_np_to_group_members_indices, dict_span_to_similar_spans, model):
    if len(common_np_to_group_members_indices.keys()) < 2:
        return common_np_to_group_members_indices
    weighted_average_vector_lst = []
    for span in common_np_to_group_members_indices.keys():
        spans_embeddings = model.encode(list(dict_span_to_similar_spans[span]))
        weighted_average_vector = get_weighted_average_vector_of_some_vectors_embeddings(spans_embeddings,
                                                                                         combine_spans_utils.dict_of_span_to_counter,
                                                                                         dict_span_to_similar_spans[
                                                                                             span])
        weighted_average_vector_lst.append(weighted_average_vector)
    clustering = AgglomerativeClustering(distance_threshold=0.08, n_clusters=None, linkage="average",
                                         affinity="cosine", compute_full_tree=True).fit(torch.stack(weighted_average_vector_lst, dim=0))
    dict_cluster_to_common_spans_lst = {}
    for idx, label in enumerate(clustering.labels_):
        dict_cluster_to_common_spans_lst[label] = dict_cluster_to_common_spans_lst.get(label, [])
        dict_cluster_to_common_spans_lst[label].append(list(common_np_to_group_members_indices.keys())[idx])
    new_common_np_to_group_members_indices = {}
    for label, similar_common_spans_lst in dict_cluster_to_common_spans_lst.items():
        if len(similar_common_spans_lst) == 1:
            continue
        new_common_np_to_group_members_indices[similar_common_spans_lst[0]] = set()
        for common_span in similar_common_spans_lst:
            for span in similar_common_spans_lst:
                dict_span_to_similar_spans[common_span].update(dict_span_to_similar_spans[span])
            new_common_np_to_group_members_indices[similar_common_spans_lst[0]].update(common_np_to_group_members_indices[common_span])
    return new_common_np_to_group_members_indices


def union_common_np(clusters, dict_word_to_lemma, dict_lemma_to_synonyms, dict_span_to_rank, dict_label_to_spans_group, model):
    span_to_group_members, dict_span_to_lemmas_lst, dict_longest_span_to_his_synonyms, dict_length_to_span = \
        create_date_dicts_for_combine_synonyms(clusters, dict_label_to_spans_group)
    span_to_group_members, dict_span_to_similar_spans = combine_similar_spans(span_to_group_members,
                                                                              dict_length_to_span, dict_word_to_lemma,
                                                                              dict_lemma_to_synonyms,
                                                                              dict_longest_span_to_his_synonyms)
    common_np_to_group_members_indices = \
        create_dict_from_common_np_to_group_members_indices(span_to_group_members,
                                                            dict_span_to_rank, dict_longest_span_to_his_synonyms)
    common_np_to_group_members_indices = union_common_np_by_DL_model(common_np_to_group_members_indices,
                                                                     dict_span_to_similar_spans, model)
    return dict_span_to_lemmas_lst, common_np_to_group_members_indices, dict_span_to_similar_spans


def get_non_clustered_group_numbers(label_to_cluster, span_to_group_members, dict_label_to_spans_group):
    all_group_numbers = set(label_to_cluster.keys())
    already_grouped = set()
    common_span_lst = []
    for common_span, group_numbers in span_to_group_members.items():
        already_grouped.update(group_numbers)
        common_span_lst.append(common_span)
    res_group_numbers = [item for item in all_group_numbers if item not in already_grouped]
    dict_label_to_longest_np_without_common_sub_np = {}
    for num in res_group_numbers:
        dict_label_to_longest_np_without_common_sub_np[num] = dict_label_to_spans_group[num]
    return dict_label_to_longest_np_without_common_sub_np, common_span_lst


def get_global_indices_and_objects_from_longest_np_lst(span_to_object, longest_np_lst):
    object_lst = set()
    label_lst = set()
    for longest_np in longest_np_lst:
        np_object = span_to_object[longest_np]
        label_lst.update(np_object.label_lst)
        object_lst.add(np_object)
    return label_lst, object_lst


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
                for np in np_object.np:
                    if len(np) == 1:
                        print(np)
                        break
                if np_object not in child_np_object_to_remove:
                    child_np_object_to_remove.add_children([np_object])
                    np_object.children.remove(child_np_object_to_remove)


def union_nps(model, label_to_cluster, dict_span_to_rank, dict_label_to_spans_group):
    dict_span_to_lst, common_np_to_group_members_indices, dict_span_to_similar_spans = union_common_np(
        label_to_cluster, combine_spans_utils.dict_word_to_lemma,
        dict_lemma_to_synonyms, dict_span_to_rank, dict_label_to_spans_group, model)
    dict_label_to_longest_np_without_common_sub_np, common_span_lst = get_non_clustered_group_numbers(
        label_to_cluster,
        common_np_to_group_members_indices,
        dict_label_to_spans_group)
    span_comparison.combine_not_clustered_spans_in_clustered_spans(dict_label_to_longest_np_without_common_sub_np,
                                                                   dict_span_to_similar_spans,
                                                                   combine_spans_utils.dict_word_to_lemma,
                                                                   dict_lemma_to_synonyms,
                                                                   common_np_to_group_members_indices, common_span_lst,
                                                                   dict_span_to_lst)
    dict_score_to_collection_of_sub_groups = create_score_to_group_dict(common_np_to_group_members_indices,
                                                                        dict_span_to_rank, dict_span_to_similar_spans)
    return dict_score_to_collection_of_sub_groups, dict_span_to_lst, dict_span_to_similar_spans


def get_only_relevant_example(example_list, global_longest_np_lst):
    dict_span_to_all_valid_expansions = {}
    longest_np_lst = []
    longest_np_total_lst = []
    all_nps_example_lst = []
    for example in example_list:
        longest_np_total_lst.append(example[0])
        if example[0] in global_longest_np_lst:
            continue
        global_longest_np_lst.add(example[0])
        dict_span_to_all_valid_expansions[example[0]] = [item[0] for item in example[1]]
        longest_np_lst.append(example[0])
        all_nps_example_lst.append(example[1])
    return dict_span_to_all_valid_expansions, longest_np_lst, longest_np_total_lst, all_nps_example_lst


def create_longest_nps_clusters(model, longest_np_lst, dict_idx_to_all_valid_expansions, global_longest_np_index,
                                global_index_to_similar_longest_np):
    phrase_embeddings = model.encode(longest_np_lst)
    clustering = AgglomerativeClustering(distance_threshold=0.08, n_clusters=None, linkage="average",
                                         affinity="cosine", compute_full_tree=True).fit(phrase_embeddings)
    label_to_cluster, dict_label_to_spans_group, cluster_index_to_local_indices = group_hierarchical_clustering_results(
        clustering, dict_idx_to_all_valid_expansions, global_longest_np_index, global_index_to_similar_longest_np)
    dict_label_to_spans_group = {k: v for k, v in
                                 sorted(dict_label_to_spans_group.items(), key=lambda item: len(item[1]),
                                        reverse=True)}
    return label_to_cluster, dict_label_to_spans_group, dict_label_to_spans_group, cluster_index_to_local_indices


def from_DAG_to_JSON(topic_object_lst, global_index_to_similar_longest_np):
    np_val_lst = {}
    topic_object_lst.sort(key=lambda topic_object: topic_object.marginal_val, reverse=True)
    for topic_node in topic_object_lst:
        np_val_lst.update(add_descendants_of_node_to_graph(topic_node, global_index_to_similar_longest_np))
    return np_val_lst


def add_descendants_of_node_to_graph(node, global_index_to_similar_longest_np):
    span_to_present = ""
    first_val = True
    for np_val in node.np_val:
        if not first_val:
            span_to_present += " | "
        first_val = False
        span_to_present += np_val
    label_lst = combine_spans_utils.get_labels_of_children(node.children)
    label_lst = node.label_lst - label_lst
    NP_occurrences = combine_spans_utils.get_frequency_from_labels_lst(global_index_to_similar_longest_np,
                                                                       label_lst)
    span_to_present += " NP " + str(NP_occurrences) + " covered by NP " + str(
        combine_spans_utils.get_frequency_from_labels_lst(global_index_to_similar_longest_np,
                                                          node.label_lst))
    np_val_dict = {span_to_present: {}}
    node.children = sorted(node.children, key=lambda child: combine_spans_utils.get_frequency_from_labels_lst(
        global_index_to_similar_longest_np,
        child.label_lst), reverse=True)
    for child in node.children:
        np_val_dict[span_to_present].update(add_descendants_of_node_to_graph(child, global_index_to_similar_longest_np))
    return np_val_dict


def update_nodes_frequency(topic_object_lst, global_index_to_similar_longest_np, visited=[]):
    for node in topic_object_lst:
        if node in visited:
            continue
        is_first = True
        for np in node.np_val:
            encoded_input = tokenizer(np, return_tensors='pt')
            if is_first:
                weighted_average_vector = medical_model(**encoded_input).last_hidden_state[0,0,:]
                is_first = False
            else:
                weighted_average_vector += medical_model(**encoded_input).last_hidden_state[0,0,:]
        weighted_average_vector /= len(node.np_val)
        node.weighted_average_vector = weighted_average_vector
        node.frequency = combine_spans_utils.get_frequency_from_labels_lst(global_index_to_similar_longest_np,
                                                                           node.label_lst)
        visited.append(node)
        update_nodes_frequency(node.children, global_index_to_similar_longest_np, visited)


def add_dependency_routh_between_longest_np_to_topic(span_to_object, topic_object_lst,
                                                     longest_nps, topic_object):
    for longest_np_span in longest_nps:
        np_object = span_to_object[longest_np_span]
        if np_object in topic_object_lst:
            np_object.combine_nodes(topic_object)
            topic_object_lst.remove(topic_object)
            topic_object = np_object
            continue
        similar_np_object = [None]
        add_NP_to_DAG_bottom_to_up(topic_object, np_object, set(), similar_np_object)
        if similar_np_object[0]:
            if similar_np_object[0] in topic_object_lst:
                topic_object_lst.remove(topic_object)
                topic_object = similar_np_object[0]
            for np in similar_np_object[0].np_val:
                span_to_object[np] = similar_np_object[0]


def update_score(topic_object_lst, dict_span_to_rank, visited=[]):
    for node in topic_object_lst:
        if node in visited:
            continue
        visited.append(node)
        node.score = get_average_value(node.np_val, dict_span_to_rank)
        update_score(node.children, dict_span_to_rank, visited)


def main():
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    dict_span_to_rank = {}
    global_longest_np_lst = set()
    global_dict_label_to_object = {}
    topic_object_lst = []
    all_object_np_lst = []
    span_to_object = {}
    global_longest_np_index = [0]
    global_index_to_similar_longest_np = {}
    counter = 0
    num_of_examples = 0
    print(sum(combine_spans_utils.dict_of_span_to_counter.values()))
    for topic, example_list in combine_spans_utils.dict_of_topics.items():
        counter += 1
        topic_synonym_lst = set()
        for synonym in combine_spans_utils.dict_topic_to_his_synonym[topic]:
            topic_synonym_lst.add(synonym)
            dict_span_to_rank[synonym] = 1
        dict_span_to_all_valid_expansions, longest_np_lst, longest_np_total_lst, all_nps_example_lst = get_only_relevant_example(
            example_list, global_longest_np_lst)
        if len(longest_np_total_lst) == 0 or len(longest_np_lst) == 0:
            continue
        for longest_np in longest_np_lst:
            num_of_examples += combine_spans_utils.dict_of_span_to_counter[longest_np]
        dict_idx_to_all_valid_expansions = initialize_spans_data(all_nps_example_lst, dict_span_to_rank)
        if len(longest_np_lst) == 1:
            global_longest_np_index[0] += 1
            global_index_to_similar_longest_np[global_longest_np_index[0]] = [longest_np_lst[0]]
            dict_label_to_spans_group = {global_longest_np_index[0]:
                                             [(longest_np_lst[0], dict_idx_to_all_valid_expansions[0][0][1])]}
            dict_score_to_collection_of_sub_groups = {}
            dict_span_to_similar_spans = {longest_np_lst[0]: longest_np_lst[0]}
            dict_span_to_lst = {}
        else:
            label_to_cluster, dict_label_to_spans_group, dict_label_to_spans_group, cluster_index_to_local_indices = \
                create_longest_nps_clusters(model, longest_np_lst, dict_idx_to_all_valid_expansions,
                                            global_longest_np_index, global_index_to_similar_longest_np)
            dict_score_to_collection_of_sub_groups, dict_span_to_lst, dict_span_to_similar_spans = union_nps(model,
                                                                                                             label_to_cluster,
                                                                                                             dict_span_to_rank,
                                                                                                             dict_label_to_spans_group)
        # span_to_group_members = set_cover_with_priority(dict_score_to_collection_of_sub_groups)
        topic_object = create_DAG_from_top_to_bottom(dict_score_to_collection_of_sub_groups, topic_synonym_lst,
                                                     dict_span_to_lst, all_object_np_lst,
                                                     span_to_object, dict_span_to_similar_spans,
                                                     dict_label_to_spans_group,
                                                     global_dict_label_to_object, topic_object_lst)
        longest_spans_calculated_in_previous_topics = set(longest_np_total_lst) - set(longest_np_lst)
        add_dependency_routh_between_longest_np_to_topic(span_to_object, topic_object_lst,
                                                         longest_spans_calculated_in_previous_topics, topic_object)
    # uncounted_span = []
    # for span in list(combine_spans_utils.dict_of_span_to_counter.keys()):
    #     if span not in global_longest_np_lst:
    #         uncounted_span.append(span)
    # print(uncounted_span)
    # print(num_of_examples)
    update_nodes_frequency(topic_object_lst, global_index_to_similar_longest_np)
    hierarchical_structure_algorithms.get_k_navigable_DAGS_from_DAG(50, topic_object_lst, global_dict_label_to_object,
                                                                    global_index_to_similar_longest_np)
    combine_spans_utils.check_symmetric_relation_in_DAG(topic_object_lst)
    leaves_lst = set()
    combine_spans_utils.get_leaves(topic_object_lst, leaves_lst, set())
    combine_spans_utils.remove_redundant_nodes(leaves_lst, topic_object_lst)
    update_score(topic_object_lst, dict_span_to_rank)
    np_val_lst = from_DAG_to_JSON(topic_object_lst, global_index_to_similar_longest_np)
    # print(np_val_lst)
    # print("END of the regular form")
    top_k_topics, already_counted_labels, all_labels = \
        hierarchical_structure_algorithms.greedy_algorithm(100, topic_object_lst, global_index_to_similar_longest_np)
    covered_labels = combine_spans_utils.get_frequency_from_labels_lst(global_index_to_similar_longest_np,
                                                                       already_counted_labels)
    total_labels = combine_spans_utils.get_frequency_from_labels_lst(global_index_to_similar_longest_np,
                                                                     all_labels)
    top_k_topics = from_DAG_to_JSON(top_k_topics, global_index_to_similar_longest_np)
    labels_of_topics = set()
    for topic_object in topic_object_lst:
        labels_of_topics.update(topic_object.label_lst)
    total_labels_of_topics = combine_spans_utils.get_frequency_from_labels_lst(global_index_to_similar_longest_np,
                                                                               labels_of_topics)
    print("total labels of topics: ", total_labels_of_topics)
    print(top_k_topics)
    print(covered_labels, total_labels)


main()
