from combine_spans import utils as combine_spans_utils
from combine_spans import combineSpans
import DAG.hierarchical_structure_algorithms as hierarchical_structure_algorithms
import DAG.DAG_utils as DAG_utils
from topic_clustering import clustered_data_objects as clustered_data_objects
import networkx as nx
import collections
import graphviz
import json
import pickle

g1 = nx.DiGraph()


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


def extract_dag_relation_by_bfs_algorithm(root):
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
    relation_lst = extract_dag_relation_by_bfs_algorithm(dag)
    g = graphviz.Digraph('G', filename='data_DAG.gv')
    for relation in relation_lst:
        g.edge(relation[1], relation[0])
    g.view()


def initialize_spans_data(all_nps_example_lst):
    dict_idx_to_all_valid_expansions = {}
    dict_idx_to_longest_np = {}
    idx = 0
    for phrase in all_nps_example_lst:
        dict_idx_to_all_valid_expansions[idx] = set()
        dict_idx_to_longest_np[idx] = phrase[0][0]
        for span in phrase:
            dict_idx_to_all_valid_expansions[idx].add(span[0])
        idx += 1
    return dict_idx_to_all_valid_expansions, dict_idx_to_longest_np


def get_uncounted_examples(example_list, global_longest_np_lst, dict_global_longest_np_to_all_counted_expansions,
                           longest_NP_to_global_index, global_index_to_similar_longest_np, dict_span_to_lst,
                           dict_span_to_rank):
    dict_span_to_all_valid_expansions = {}
    longest_np_lst = []
    longest_np_total_lst = []
    all_nps_example_lst = []
    dict_uncounted_expansions = {}
    dict_counted_longest_answers = {}
    duplicate_longest_answers = set()
    for example in example_list:
        longest_np_total_lst.append(example[0])
        if example[0] in duplicate_longest_answers:
            continue
        duplicate_longest_answers.add(example[0])
        if example[0] in global_longest_np_lst:
            label = longest_NP_to_global_index[example[0]]
            dict_counted_longest_answers[label] = set()
            for longest_answer in global_index_to_similar_longest_np[label]:
                dict_counted_longest_answers[label].add(longest_answer)
            uncounted_expansions = set()
            for item in example[1]:
                if item[0] in dict_global_longest_np_to_all_counted_expansions[example[0]]:
                    continue
                dict_span_to_lst[item[0]] = item[2]
                dict_span_to_rank[item[0]] = item[1]
                dict_global_longest_np_to_all_counted_expansions[example[0]].add(item[0])
                uncounted_expansions.add(item[0])
            if uncounted_expansions:
                dict_uncounted_expansions[label] = uncounted_expansions
            continue
        dict_global_longest_np_to_all_counted_expansions[
            example[0]] = dict_global_longest_np_to_all_counted_expansions.get(example[0], set())
        for item in example[1]:
            dict_span_to_lst[item[0]] = item[2]
            dict_span_to_rank[item[0]] = item[1]
            dict_global_longest_np_to_all_counted_expansions[example[0]].add(item[0])
        global_longest_np_lst.add(example[0])
        dict_span_to_all_valid_expansions[example[0]] = [item[0] for item in example[1]]
        longest_np_lst.append(example[0])
        all_nps_example_lst.append(example[1])
    # dict_counted_longest_answers = {}
    # if dict_uncounted_expansions:
    #     for key in dict_uncounted_expansions.keys():
    #         dict_counted_longest_answers[key] = set()
    #         for longest_answer in global_index_to_similar_longest_np[key]:
    #             dict_counted_longest_answers[key].add(longest_answer)
    return dict_span_to_all_valid_expansions, longest_np_lst, longest_np_total_lst, all_nps_example_lst, \
           dict_uncounted_expansions, dict_counted_longest_answers


def main():
    dict_span_to_rank = {}
    dict_span_to_lst = {}
    global_dict_label_to_object = {}
    span_to_object = {}
    global_index_to_similar_longest_np = {}
    global_longest_np_lst = set()
    dict_global_longest_np_to_all_counted_expansions = {}
    topic_object_lst = []
    all_object_np_lst = []
    global_longest_np_index = [0]
    longest_NP_to_global_index = {}
    dict_object_to_global_label = {}
    dict_span_to_similar_spans = {}
    print(combine_spans_utils.dict_noun_lemma_to_synonyms)
    for topic, examples_list in combine_spans_utils.dict_of_topics.items():
        # noun_object = clustered_data_objects.noun_cluster_object(topic,
        #                                                          combine_spans_utils.dict_noun_lemma_to_synonyms[topic],
        #                                                          examples_list)
        topic_synonym_lst = set(combine_spans_utils.dict_noun_lemma_to_synonyms[topic])
        for synonym in topic_synonym_lst:
            dict_span_to_rank[synonym] = 1
        dict_span_to_all_valid_expansions, longest_np_lst, longest_np_total_lst, all_nps_example_lst, \
        dict_uncounted_expansions, dict_counted_longest_answers = get_uncounted_examples(examples_list,
                                                                                         global_longest_np_lst,
                                                                                         dict_global_longest_np_to_all_counted_expansions,
                                                                                         longest_NP_to_global_index,
                                                                                         global_index_to_similar_longest_np,
                                                                                         dict_span_to_lst,
                                                                                         dict_span_to_rank)
        dict_idx_to_all_valid_expansions, dict_idx_to_longest_np = initialize_spans_data(all_nps_example_lst)
        # if len(longest_np_lst) == 1:
        #     global_longest_np_index[0] += 1s
        #     longest_NP_to_global_index[longest_np_lst[0]] = global_longest_np_index[0]
        #     global_index_to_similar_longest_np[global_longest_np_index[0]] = [longest_np_lst[0]]
        #     dict_label_to_spans_group = {global_longest_np_index[0]:
        #                                      [(longest_np_lst[0], dict_idx_to_all_valid_expansions[0][0][1])]}
        #     dict_span_to_similar_spans = {longest_np_lst[0]: longest_np_lst[0]}
        # else:
        label_to_nps_collection, dict_label_to_longest_nps_group = combineSpans.create_clusters_of_longest_nps(
            longest_np_lst, dict_idx_to_all_valid_expansions, dict_idx_to_longest_np,
            global_longest_np_index,
            global_index_to_similar_longest_np, longest_NP_to_global_index, dict_uncounted_expansions,
            dict_counted_longest_answers)
        dict_score_to_collection_of_sub_groups, dict_span_to_similar_spans = \
            combineSpans.union_nps(label_to_nps_collection, dict_span_to_rank, dict_label_to_longest_nps_group,
                                   dict_span_to_lst)
        DAG_utils.insert_examples_of_topic_to_DAG(dict_score_to_collection_of_sub_groups, topic_synonym_lst,
                                                  dict_span_to_lst,
                                                  all_object_np_lst, span_to_object, dict_span_to_similar_spans,
                                                  dict_label_to_longest_nps_group, global_dict_label_to_object,
                                                  topic_object_lst, longest_np_total_lst, longest_np_lst,
                                                  longest_NP_to_global_index, dict_object_to_global_label)
        DAG_utils.check_symmetric_relation_in_DAG(topic_object_lst)
    DAG_utils.update_nodes_frequency(topic_object_lst, global_index_to_similar_longest_np)
    hierarchical_structure_algorithms.DAG_contraction_by_set_cover_algorithm(topic_object_lst,
                                                                             global_dict_label_to_object,
                                                                             global_index_to_similar_longest_np)

    DAG_utils.check_symmetric_relation_in_DAG(topic_object_lst)
    DAG_utils.remove_redundant_nodes(topic_object_lst)
    DAG_utils.update_score(topic_object_lst, dict_span_to_rank)
    top_k_topics, already_counted_labels, all_labels = \
        hierarchical_structure_algorithms.extract_top_k_concept_nodes_greedy_algorithm(
            100, topic_object_lst, global_index_to_similar_longest_np)
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
    # result_file = open("diabetes_output.txt", "wb")
    with open('meningitis_output.txt', 'w') as result_file:
        result_file.write(json.dumps(top_k_topics))


main()
