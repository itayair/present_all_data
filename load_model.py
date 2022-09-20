from sentence_transformers import SentenceTransformer
import torch
from sklearn.cluster import AgglomerativeClustering
from combine_spans import utils as combine_spans_utils
from combine_spans import span_comparison
import DAG.hierarchical_structure_algorithms as hierarchical_structure_algorithms
import DAG.DAG_utils as DAG_utils
import networkx as nx
import collections
import graphviz

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
g1 = nx.DiGraph()

dict_lemma_to_synonyms = combine_spans_utils.dict_lemma_to_synonyms


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
    return span_to_group_members_new, dict_spans


def group_hierarchical_clustering_results(clustering, dict_idx_to_all_valid_expansions, global_longest_np_index,
                                          global_index_to_similar_longest_np):
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
    return label_to_cluster, dict_label_to_spans_group


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


def create_score_to_group_dict(span_to_group_members, dict_span_to_rank, dict_span_to_similar_spans):
    dict_score_to_collection_of_sub_groups = {}
    for key, group in span_to_group_members.items():
        average_val = combine_spans_utils.get_average_value(dict_span_to_similar_spans[key], dict_span_to_rank)
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


def get_global_index(cluster_index_to_local_indices, longest_np_to_index, longest_np_lst, label_lst):
    global_indices = []
    for label in label_lst:
        local_indices = cluster_index_to_local_indices[label]
        for index in local_indices:
            longest_np = longest_np_lst[index]
            global_indices.append(longest_np_to_index[longest_np])
    return global_indices


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


def union_common_np_by_DL_model(common_np_to_group_members_indices, dict_span_to_similar_spans):
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
        for common_span in similar_common_spans_lst:
            for span in similar_common_spans_lst:
                dict_span_to_similar_spans[common_span].update(dict_span_to_similar_spans[span])
            new_common_np_to_group_members_indices[similar_common_spans_lst[0]].update(
                common_np_to_group_members_indices[common_span])
    return new_common_np_to_group_members_indices


def union_common_np(clusters, dict_word_to_lemma, dict_lemma_to_synonyms, dict_span_to_rank, dict_label_to_spans_group):
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
                                                                     dict_span_to_similar_spans)
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


def union_nps(label_to_cluster, dict_span_to_rank, dict_label_to_spans_group):
    dict_span_to_lst, common_np_to_group_members_indices, dict_span_to_similar_spans = union_common_np(
        label_to_cluster, combine_spans_utils.dict_word_to_lemma, dict_lemma_to_synonyms,
        dict_span_to_rank, dict_label_to_spans_group)
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


def create_clusters_of_longest_nps(longest_np_lst, dict_idx_to_all_valid_expansions, global_longest_np_index,
                                   global_index_to_similar_longest_np):
    phrase_embeddings = model.encode(longest_np_lst)
    clustering = AgglomerativeClustering(distance_threshold=0.08, n_clusters=None, linkage="average",
                                         affinity="cosine", compute_full_tree=True).fit(phrase_embeddings)
    label_to_cluster, dict_label_to_spans_group = group_hierarchical_clustering_results(
        clustering, dict_idx_to_all_valid_expansions, global_longest_np_index, global_index_to_similar_longest_np)
    dict_label_to_spans_group = {k: v for k, v in
                                 sorted(dict_label_to_spans_group.items(), key=lambda item: len(item[1]),
                                        reverse=True)}
    return label_to_cluster, dict_label_to_spans_group


def main():
    dict_span_to_rank = {}
    global_dict_label_to_object = {}
    span_to_object = {}
    global_index_to_similar_longest_np = {}
    global_longest_np_lst = set()
    topic_object_lst = []
    all_object_np_lst = []
    global_longest_np_index = [0]
    for topic, examples_list in combine_spans_utils.dict_of_topics.items():
        topic_synonym_lst = set()
        dict_score_to_collection_of_sub_groups = {}
        dict_span_to_lst = {}
        for synonym in combine_spans_utils.dict_topic_to_his_synonym[topic]:
            topic_synonym_lst.add(synonym)
            dict_span_to_rank[synonym] = 1
        dict_span_to_all_valid_expansions, longest_np_lst, longest_np_total_lst, all_nps_example_lst = \
            get_only_relevant_example(examples_list, global_longest_np_lst)
        dict_idx_to_all_valid_expansions = initialize_spans_data(all_nps_example_lst, dict_span_to_rank)
        if len(longest_np_lst) == 1:
            global_longest_np_index[0] += 1
            global_index_to_similar_longest_np[global_longest_np_index[0]] = [longest_np_lst[0]]
            dict_label_to_spans_group = {global_longest_np_index[0]:
                                             [(longest_np_lst[0], dict_idx_to_all_valid_expansions[0][0][1])]}
            dict_span_to_similar_spans = {longest_np_lst[0]: longest_np_lst[0]}
        else:
            label_to_cluster, dict_label_to_spans_group = create_clusters_of_longest_nps(
                longest_np_lst, dict_idx_to_all_valid_expansions,
                global_longest_np_index,
                global_index_to_similar_longest_np)
            dict_score_to_collection_of_sub_groups, dict_span_to_lst, dict_span_to_similar_spans = \
                union_nps(label_to_cluster, dict_span_to_rank, dict_label_to_spans_group)
        DAG_utils.insert_examples_of_topic_to_DAG(dict_score_to_collection_of_sub_groups, topic_synonym_lst,
                                                  dict_span_to_lst,
                                                  all_object_np_lst, span_to_object, dict_span_to_similar_spans,
                                                  dict_label_to_spans_group, global_dict_label_to_object,
                                                  topic_object_lst,
                                                  longest_np_total_lst, longest_np_lst, dict_lemma_to_synonyms)
    DAG_utils.update_nodes_frequency(topic_object_lst, global_index_to_similar_longest_np)
    hierarchical_structure_algorithms.DAG_contraction_by_set_cover_algorithm(topic_object_lst,
                                                                             global_dict_label_to_object,
                                                                             global_index_to_similar_longest_np)
    DAG_utils.check_symmetric_relation_in_DAG(topic_object_lst)
    DAG_utils.remove_redundant_nodes(topic_object_lst)
    DAG_utils.update_score(topic_object_lst, dict_span_to_rank)
    top_k_topics, already_counted_labels, all_labels = \
        hierarchical_structure_algorithms.extract_top_k_concept_nodes_greedy_algorithm(
            50, topic_object_lst, global_index_to_similar_longest_np)
    covered_labels = DAG_utils.get_frequency_from_labels_lst(global_index_to_similar_longest_np,
                                                             already_counted_labels)
    top_k_topics = DAG_utils.from_DAG_to_JSON(top_k_topics, global_index_to_similar_longest_np)
    labels_of_topics = set()
    for topic_object in topic_object_lst:
        labels_of_topics.update(topic_object.label_lst)
    total_labels_of_topics = DAG_utils.get_frequency_from_labels_lst(global_index_to_similar_longest_np,
                                                                     labels_of_topics)
    print("total labels of topics:", total_labels_of_topics)
    print("Covered labels by selected nodes:", covered_labels)
    print(top_k_topics)


main()
