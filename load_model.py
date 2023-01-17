from combine_spans import utils as combine_spans_utils
from combine_spans import combineSpans
import DAG.hierarchical_structure_algorithms as hierarchical_structure_algorithms
import DAG.DAG_utils as DAG_utils
from topic_clustering import clustered_data_objects as clustered_data_objects
import networkx as nx
import collections
import graphviz
import json
import matplotlib.pyplot as plt
import pickle
import sys

sys.setrecursionlimit(10000)
G = nx.DiGraph()


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
                           longest_NP_to_global_index, global_index_to_similar_longest_np, dict_span_to_lemma_lst,
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
                dict_span_to_lemma_lst[item[0]] = item[2]
                dict_span_to_rank[item[0]] = item[1]
                dict_global_longest_np_to_all_counted_expansions[example[0]].add(item[0])
                uncounted_expansions.add(item[0])
            if uncounted_expansions:
                dict_uncounted_expansions[label] = uncounted_expansions
            continue
        dict_global_longest_np_to_all_counted_expansions[
            example[0]] = dict_global_longest_np_to_all_counted_expansions.get(example[0], set())
        for item in example[1]:
            dict_span_to_lemma_lst[item[0]] = item[2]
            dict_span_to_rank[item[0]] = item[1]
            dict_global_longest_np_to_all_counted_expansions[example[0]].add(item[0])
        global_longest_np_lst.add(example[0])
        dict_span_to_all_valid_expansions[example[0]] = [item[0] for item in example[1]]
        longest_np_lst.append(example[0])
        all_nps_example_lst.append(example[1])
    return dict_span_to_all_valid_expansions, longest_np_lst, longest_np_total_lst, all_nps_example_lst, \
           dict_uncounted_expansions, dict_counted_longest_answers


def convert_node_to_symbol(spans_lst):
    spans_as_string = ""
    idx = 0
    for span in spans_lst:
        if idx != 0:
            spans_as_string += "\n"
        spans_as_string += span
        idx += 1
        break
    return spans_as_string


def plot_graph(nodes_lst):
    # for node in nodes_lst:
    #     label = convert_node_to_symbol(node.span_lst)
    #     G.add_node(label)
    g = graphviz.Digraph('G', filename='contracted_DAG.gv')
    # for relation in relation_lst:
    #     g.edge(relation[1], relation[0])
    for node in nodes_lst:
        node_label = convert_node_to_symbol(node.span_lst)
        g.node(node_label)
        for child in node.children:
            child_label = convert_node_to_symbol(child.span_lst)
            g.edge(node_label, child_label)
    g.view()
    g.clear()
    # plt.show()
    # G.clear()


def dfs(visited, node):  # function for dfs
    if node not in visited:
        visited.append(node)
        for neighbour in node.children:
            dfs(visited, neighbour)


def get_all_nodes_from_roots(topic_lst):
    all_object_np_lst = []
    for node in topic_lst:
        dfs(all_object_np_lst, node)
    return all_object_np_lst


def write_to_file_topic_to_collection():
    file_name = "topic_to_collection.txt"
    with open(file_name, 'w', encoding='utf-8') as f:
        for topic, collections in combine_spans_utils.dict_of_topics.items():
            f.write(topic + ": ")
            f.write('[')
            for collection in collections:
                longest_span = collection[0]
                f.write(longest_span + ': ')
                idx = 0
                f.write('[')
                for span_data in collection[1]:
                    span = span_data[0]
                    if idx != 0:
                        f.write(', ')
                    f.write(span)
                    idx += 1
                f.write(']')
            f.write(']')


def write_to_file_group_of_similar_concepts(nodes_lst):
    file_name = "group_of_similar_concepts.txt"
    with open(file_name, 'w', encoding='utf-8') as f:
        idx_group = 0
        for node in nodes_lst:
            if idx_group != 0:
                f.write(', ')
            f.write('[')
            idx = 0
            for span in node.span_lst:
                if idx != 0:
                    f.write(', ')
                f.write(span)
                idx += 1
            f.write(']')
            idx_group += 1


def dfs_for_cyclic(visited, helper, node):
    visited.append(node)
    helper.append(node)
    children = node.children
    for child in children:
        if child not in visited:
            ans = dfs_for_cyclic(visited, helper, child)
            if ans == True:
                print(child.span_lst)
                return True
        elif child in helper:
            print(child.span_lst)
            return True
    helper.remove(node)
    return False


def isCyclic(nodes_lst):
    visited = []
    helper = []
    for i in nodes_lst:
        if i not in visited:
            ans = dfs_for_cyclic(visited, helper, i)
            if ans == True:
                print(i.span_lst)
                return True
    return False


def main():
    dict_span_to_rank = {}
    dict_span_to_lemma_lst = combine_spans_utils.dict_span_to_lemma_lst
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
    counter = 0
    print(len(combine_spans_utils.dict_of_topics.keys()))
    for topic, examples_list in combine_spans_utils.dict_of_topics.items():
        print(counter)
        print(topic)
        topic_synonym_lst = set(combine_spans_utils.dict_noun_lemma_to_synonyms[topic])
        for synonym in topic_synonym_lst:
            dict_span_to_rank[synonym] = 1
        dict_span_to_all_valid_expansions, longest_np_lst, longest_np_total_lst, all_nps_example_lst, \
        dict_uncounted_expansions, dict_counted_longest_answers = get_uncounted_examples(examples_list,
                                                                                         global_longest_np_lst,
                                                                                         dict_global_longest_np_to_all_counted_expansions,
                                                                                         longest_NP_to_global_index,
                                                                                         global_index_to_similar_longest_np,
                                                                                         dict_span_to_lemma_lst,
                                                                                         dict_span_to_rank)
        label_to_nps_collection, dict_label_to_longest_nps_group = \
            combineSpans.create_index_and_collection_for_longest_nps(longest_np_lst, all_nps_example_lst,
                                                                     global_longest_np_index,
                                                                     global_index_to_similar_longest_np,
                                                                     longest_NP_to_global_index,
                                                                     dict_uncounted_expansions,
                                                                     dict_counted_longest_answers)
        dict_score_to_collection_of_sub_groups, dict_span_to_similar_spans = \
            combineSpans.union_nps(label_to_nps_collection, dict_span_to_rank, dict_label_to_longest_nps_group,
                                   dict_span_to_lemma_lst, span_to_object)
        DAG_utils.insert_examples_of_topic_to_DAG(dict_score_to_collection_of_sub_groups, topic_synonym_lst,
                                                  dict_span_to_lemma_lst,
                                                  all_object_np_lst, span_to_object, dict_span_to_similar_spans,
                                                  global_dict_label_to_object, topic_object_lst, longest_np_total_lst,
                                                  longest_np_lst, longest_NP_to_global_index,
                                                  dict_object_to_global_label)
        # DAG_utils.check_symmetric_relation_in_DAG(topic_object_lst)
        # isCyclic_val = isCyclic(topic_object_lst)
        # if isCyclic_val:
        #     print("There is a cyclic in the DAG")
        counter += 1
    # nodes_lst = get_all_nodes_from_roots(topic_object_lst)
    # write_to_file_group_of_similar_concepts(nodes_lst)
    # plot_graph(nodes_lst)
    DAG_utils.initialize_nodes_weighted_average_vector(topic_object_lst, global_index_to_similar_longest_np)
    all_data_dict = {
        'topic_object_lst': topic_object_lst,
        'global_index_to_similar_longest_np': global_index_to_similar_longest_np,
        'dict_span_to_rank': dict_span_to_rank,
        'global_dict_label_to_object': global_dict_label_to_object
    }
    pickle.dump(all_data_dict, open("results_disease/diabetes/all_data_dict.p", "wb"))
    # pickle.dump(topic_object_lst, open("results_disease/diabetes/topic_object_lst.p", "wb"))
    # pickle.dump(global_index_to_similar_longest_np,
    #             open("results_disease/diabetes/global_index_to_similar_longest_np.p", "wb"))
    # pickle.dump(dict_span_to_rank, open("results_disease/diabetes/dict_span_to_rank.p", "wb"))
    # pickle.dump(global_dict_label_to_object, open("results_disease/diabetes/global_dict_label_to_object.p", "wb"))
    # hierarchical_structure_algorithms.DAG_contraction_by_set_cover_algorithm(topic_object_lst,
    #                                                                          global_dict_label_to_object,
    #                                                                          global_index_to_similar_longest_np)
    # # nodes_lst = get_all_nodes_from_roots(topic_object_lst)
    # # plot_graph(nodes_lst)
    # DAG_utils.check_symmetric_relation_in_DAG(topic_object_lst)
    # DAG_utils.remove_redundant_nodes(topic_object_lst)
    # DAG_utils.update_score(topic_object_lst, dict_span_to_rank)
    # top_k_topics, already_counted_labels, all_labels = \
    #     hierarchical_structure_algorithms.extract_top_k_concept_nodes_greedy_algorithm(
    #         100, topic_object_lst, global_index_to_similar_longest_np)
    # covered_labels = DAG_utils.get_frequency_from_labels_lst(global_index_to_similar_longest_np,
    #                                                          already_counted_labels)
    #
    # # nodes_lst = get_all_nodes_from_roots(top_k_topics)
    # # plot_graph(nodes_lst)
    # top_k_topics = DAG_utils.from_DAG_to_JSON(top_k_topics, global_index_to_similar_longest_np)
    # labels_of_topics = set()
    # for topic_object in topic_object_lst:
    #     labels_of_topics.update(topic_object.label_lst)
    # total_labels_of_topics = DAG_utils.get_frequency_from_labels_lst(global_index_to_similar_longest_np,
    #                                                                  labels_of_topics)
    # print("total labels of topics:", total_labels_of_topics)
    # print("Covered labels by selected nodes:", covered_labels)
    # # result_file = open("diabetes_output.txt", "wb")
    # with open('abortion_output.txt.txt', 'w') as result_file:
    #     result_file.write(json.dumps(top_k_topics))


main()
