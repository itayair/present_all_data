import src.DAG.DAG_utils as DAG_utils
import src.utils as ut
import json
import networkx as nx
import graphviz


def write_to_file_topic_to_collection():
    file_name = "topic_to_collection.txt"
    with open(file_name, 'w', encoding='utf-8') as f:
        for topic, collections in ut.dict_of_topics.items():
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


def print_flat_list_to_file(concept_to_occurrences):
    concept_to_occurrences = {k: v for k, v in
                              sorted(concept_to_occurrences.items(), key=lambda item: item[1], reverse=True)}
    file_name = "flat_list_for_UI_chest_pain.txt"
    concept_lst = []
    with open(file_name, 'w', encoding='utf-8') as f:
        idx = 0
        for longest_span, number in concept_to_occurrences.items():
            concept = longest_span + ': ' + str(number)
            concept_lst.append(concept)
            concept = str(idx) + ") " + concept
            f.write(concept)
            idx += 1
            f.write('\n')
    with open('flat_list_for_UI_as_json_chest_pain.txt', 'w') as result_file:
        result_file.write(json.dumps(concept_lst))


def visualize_dag(dag):
    relation_lst = extract_dag_relation_by_bfs_algorithm(dag)
    g = graphviz.Digraph('G', filename='data_DAG.gv')
    for relation in relation_lst:
        g.edge(relation[1], relation[0])
    g.view()


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
    g = graphviz.Digraph('G', filename='contracted_DAG.gv')
    for node in nodes_lst:
        node_label = convert_node_to_symbol(node.span_lst)
        g.node(node_label)
        for child in node.children:
            child_label = convert_node_to_symbol(child.span_lst)
            g.edge(node_label, child_label)
    g.view()
    g.clear()


def get_all_labels(nodes, labels, visited=set()):
    if visited is None:
        visited = set()
    for node in nodes:
        if node in visited:
            continue
        labels.update(node.label_lst)
        get_all_labels(node.children, labels, visited)


def json_dag_visualization(top_k_topics, global_index_to_similar_longest_np, taxonomic_np_objects, topic_object_lst):
    different_concepts = set()
    concept_to_occurrences = {}
    top_k_topics_as_json = DAG_utils.from_DAG_to_JSON(top_k_topics, global_index_to_similar_longest_np,
                                                      taxonomic_np_objects, different_concepts,
                                                      concept_to_occurrences)
    # print_flat_list_to_file(concept_to_occurrences)
    print(len(different_concepts))
    top_k_labels = set()
    get_all_labels(top_k_topics, top_k_labels, visited=set())
    print("Number of different results covered by the k topics:")
    print(len(top_k_labels))
    covered_labels = DAG_utils.get_frequency_from_labels_lst(global_index_to_similar_longest_np,
                                                             top_k_labels)
    labels_of_topics = set()
    get_all_labels(topic_object_lst, labels_of_topics, visited=set())
    total_labels_of_topics = DAG_utils.get_frequency_from_labels_lst(global_index_to_similar_longest_np,
                                                                     labels_of_topics)
    print("total labels of topics:", total_labels_of_topics)
    print("Covered labels by selected nodes:", covered_labels)  # result_file = open("diabetes_output.txt", "wb")
    with open('results/results_disease/' + ut.etiology + '_' + str(ut.entries_number_limit) + '.txt', 'w') as result_file:
        result_file.write(json.dumps(top_k_topics_as_json))
    print("Done")
