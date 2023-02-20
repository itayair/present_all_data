from src.combine_spans import combineSpans, paraphrase_detection_SAP_BERT
from src import utils as ut
import src.DAG.hierarchical_structure_algorithms as hierarchical_structure_algorithms
import src.visualization.json_dag_visualization as json_dag_visualization
import src.DAG.DAG_utils as DAG_utils
import collections
from src.taxonomy import taxonomies_from_UMLS
import sys

# from qa_nom import qa_nom_in_DAG as qa_nom_in_DAG

sys.setrecursionlimit(10000)


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
                dict_span_to_rank[item[0]] = len(item[2])
                dict_global_longest_np_to_all_counted_expansions[example[0]].add(item[0])
                uncounted_expansions.add(item[0])
            if uncounted_expansions:
                dict_uncounted_expansions[label] = uncounted_expansions
            continue
        dict_global_longest_np_to_all_counted_expansions[
            example[0]] = dict_global_longest_np_to_all_counted_expansions.get(example[0], set())
        for item in example[1]:
            dict_span_to_lemma_lst[item[0]] = item[2]
            dict_span_to_rank[item[0]] = len(item[2])
            dict_global_longest_np_to_all_counted_expansions[example[0]].add(item[0])
        global_longest_np_lst.add(example[0])
        dict_span_to_all_valid_expansions[example[0]] = [item[0] for item in example[1]]
        longest_np_lst.append(example[0])
        all_nps_example_lst.append(example[1])
    return dict_span_to_all_valid_expansions, longest_np_lst, longest_np_total_lst, all_nps_example_lst, \
           dict_uncounted_expansions, dict_counted_longest_answers


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


def update_nodes_labels(nodes_lst, visited=set()):
    labels_lst = set()
    for node in nodes_lst:
        desc_labels = update_nodes_labels(node.children, visited)
        node.label_lst.update(desc_labels)
        labels_lst.update(node.label_lst)
        visited.add(node)
    return labels_lst


def get_all_spans(np_object_lst, all_spans, visited=set()):
    for np_object in np_object_lst:
        if np_object in visited:
            continue
        all_spans.update(np_object.span_lst)
        get_all_spans(np_object.children, all_spans)


def main():
    dict_span_to_rank = {}
    dict_span_to_lemma_lst = ut.dict_span_to_lemma_lst
    global_dict_label_to_object = {}
    span_to_object = {}
    global_index_to_similar_longest_np = {}
    all_spans = set()
    span_to_vector = {}
    global_longest_np_lst = set()
    dict_global_longest_np_to_all_counted_expansions = {}
    topic_object_lst = []
    all_object_np_lst = []
    global_longest_np_index = [0]
    longest_NP_to_global_index = {}
    dict_object_to_global_label = {}
    counter = 0
    print(len(ut.topics_dict.keys()))
    for topic, examples_list in ut.topics_dict.items():
        print(counter)
        print(topic)
        topic_synonym_lst = set(ut.dict_noun_lemma_to_synonyms[topic])
        for synonym in topic_synonym_lst:
            dict_span_to_rank[synonym] = 1
        dict_span_to_all_valid_expansions, longest_np_lst, longest_np_total_lst, all_nps_example_lst, \
        dict_uncounted_expansions, dict_counted_longest_answers = \
            get_uncounted_examples(examples_list, global_longest_np_lst,
                                   dict_global_longest_np_to_all_counted_expansions, longest_NP_to_global_index,
                                   global_index_to_similar_longest_np, dict_span_to_lemma_lst, dict_span_to_rank)
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
        counter += 1
    print(global_longest_np_index)
    get_all_spans(topic_object_lst, all_spans)
    all_spans = list(all_spans)
    DAG_utils.initialize_all_spans_vectors(all_spans, span_to_vector)
    DAG_utils.initialize_nodes_weighted_average_vector(topic_object_lst, global_index_to_similar_longest_np,
                                                       span_to_vector)
    paraphrase_detection_SAP_BERT.combine_equivalent_nodes_by_semantic_DL_model(topic_object_lst, span_to_object,
                                                                                dict_object_to_global_label,
                                                                                global_dict_label_to_object,
                                                                                span_to_vector)
    print("END combine equivalent nodes by DL model")
    DAG_utils.initialize_nodes_weighted_average_vector(topic_object_lst, global_index_to_similar_longest_np,
                                                       span_to_vector)
    # qa_nom_in_DAG.add_qanom_to_DAG(topic_object_lst, span_to_object)
    taxonomic_np_objects = taxonomies_from_UMLS.add_taxonomies_to_DAG_by_UMLS(topic_object_lst, dict_span_to_rank,
                                                                              span_to_object,
                                                                              dict_object_to_global_label,
                                                                              global_dict_label_to_object,
                                                                              span_to_vector)
    DAG_utils.initialize_nodes_weighted_average_vector(topic_object_lst, global_index_to_similar_longest_np,
                                                       span_to_vector)
    # DAG_utils.check_symmetric_relation_in_DAG(topic_object_lst)
    # DAG contraction
    print("before combine parent and children nodes by DL model")
    paraphrase_detection_SAP_BERT.combine_equivalent_parent_and_children_nodes_by_semantic_DL_model(
        topic_object_lst.copy(), span_to_object, dict_object_to_global_label, global_dict_label_to_object,
        topic_object_lst, span_to_vector)
    update_nodes_labels(topic_object_lst)
    print("finish combine parent children nodes by DL models")
    hierarchical_structure_algorithms.DAG_contraction_by_set_cover_algorithm(topic_object_lst,
                                                                             global_dict_label_to_object,
                                                                             global_index_to_similar_longest_np)
    print("finish DAG contraction")
    DAG_utils.remove_redundant_nodes(topic_object_lst)
    update_nodes_labels(topic_object_lst)
    DAG_utils.initialize_nodes_weighted_average_vector(topic_object_lst, global_index_to_similar_longest_np,
                                                       span_to_vector)
    top_k_topics, already_counted_labels, all_labels = \
        hierarchical_structure_algorithms.extract_top_k_concept_nodes_greedy_algorithm(
            ut.entries_number_limit, topic_object_lst, global_index_to_similar_longest_np)
    json_dag_visualization.json_dag_visualization(top_k_topics, global_index_to_similar_longest_np,
                                                  taxonomic_np_objects, topic_object_lst)


main()
