from combine_spans import utils as combine_spans_utils
from combine_spans import combineSpans
import DAG.hierarchical_structure_algorithms as hierarchical_structure_algorithms
import DAG.DAG_utils as DAG_utils
import networkx as nx
import collections
import graphviz

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
        for synonym in combine_spans_utils.dict_noun_lemma_to_synonyms[topic]:
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
            label_to_cluster, dict_label_to_spans_group = combineSpans.create_clusters_of_longest_nps(
                longest_np_lst, dict_idx_to_all_valid_expansions,
                global_longest_np_index,
                global_index_to_similar_longest_np)
            dict_score_to_collection_of_sub_groups, dict_span_to_lst, dict_span_to_similar_spans = \
                combineSpans.union_nps(label_to_cluster, dict_span_to_rank, dict_label_to_spans_group)
        DAG_utils.insert_examples_of_topic_to_DAG(dict_score_to_collection_of_sub_groups, topic_synonym_lst,
                                                  dict_span_to_lst,
                                                  all_object_np_lst, span_to_object, dict_span_to_similar_spans,
                                                  dict_label_to_spans_group, global_dict_label_to_object,
                                                  topic_object_lst,
                                                  longest_np_total_lst, longest_np_lst)
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
