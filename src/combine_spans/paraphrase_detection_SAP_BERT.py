from transformers import AutoTokenizer, AutoModel
import torch
import pickle
from sklearn.cluster import AgglomerativeClustering
from src.combine_spans import combineSpans as combineSpans
from src.combine_spans import utils as combine_spans_utils

cos = torch.nn.CosineSimilarity(dim=0, eps=1e-08)


# def combine_equivalent_nodes(np_object_lst, span_to_object, dict_object_to_global_label, global_dict_label_to_object,
#                              span_to_vector):
#     if len(np_object_lst) <= 1:
#         return
#     span_to_node_dict = {}
#     span_lst = []
#     for node in np_object_lst:
#         most_frequent_span = combine_spans_utils.get_most_frequent_span(node.span_lst)
#         if most_frequent_span in span_to_node_dict:
#             np_object = span_to_node_dict[most_frequent_span]
#             np_object.combine_nodes(node)
#             continue
#         span_to_node_dict[most_frequent_span] = node
#         span_lst.append(most_frequent_span)
#     with torch.no_grad():
#         encoded_input = \
#             combineSpans.sapBert_tokenizer(span_lst, return_tensors='pt', padding=True).to(combineSpans.device)
#         phrase_embeddings = combineSpans.model(**encoded_input).last_hidden_state.cpu()
#         phrase_embeddings = torch.transpose(phrase_embeddings, 0, 1)
#         phrase_embeddings = phrase_embeddings[0, :]
#         del encoded_input
#         torch.cuda.empty_cache()
#     clustering = AgglomerativeClustering(distance_threshold=0.1, n_clusters=None, linkage="average",
#                                          affinity="cosine", compute_full_tree=True).fit(phrase_embeddings)
#     dict_cluster_to_common_spans_lst = {}
#     for idx, label in enumerate(clustering.labels_):
#         dict_cluster_to_common_spans_lst[label] = dict_cluster_to_common_spans_lst.get(label, [])
#         np_object = span_to_node_dict.get(span_lst[idx], None)
#         dict_cluster_to_common_spans_lst[label].append(np_object)
#     for label, equivalent_nodes in dict_cluster_to_common_spans_lst.items():
#         if len(equivalent_nodes) == 1:
#             continue
#         combine_spans_utils.combine_nodes_lst(equivalent_nodes, span_to_object, dict_object_to_global_label,
#                                               global_dict_label_to_object)


def combine_equivalent_nodes(np_object_lst, span_to_object, dict_object_to_global_label, global_dict_label_to_object,
                             span_to_vector):
    if len(np_object_lst) <= 1:
        return
    node_vector_lst = []
    for np_object in np_object_lst:
        node_vector_lst.append(np_object.weighted_average_vector)
    node_vector_lst = torch.stack(node_vector_lst).reshape(len(np_object_lst), -1)
    clustering = AgglomerativeClustering(distance_threshold=0.1, n_clusters=None, linkage="average",
                                         affinity="cosine", compute_full_tree=True).fit(node_vector_lst)
    dict_cluster_to_common_spans_lst = {}
    for idx, label in enumerate(clustering.labels_):
        dict_cluster_to_common_spans_lst[label] = dict_cluster_to_common_spans_lst.get(label, [])
        dict_cluster_to_common_spans_lst[label].append(np_object_lst[idx])
    for label, equivalent_nodes in dict_cluster_to_common_spans_lst.items():
        if len(equivalent_nodes) == 1:
            continue
        combine_spans_utils.combine_nodes_lst(equivalent_nodes, span_to_object, dict_object_to_global_label,
                                              global_dict_label_to_object, span_to_vector)


# def combine_equivalent_node_with_its_equivalent_children(parent, children, span_to_object, dict_object_to_global_label,
#                                                          global_dict_label_to_object, topic_object_lst, span_to_vector,
#                                                          visited):
#     if len(children) == 0:
#         return
#     span_to_node_dict = {}
#     span_lst = []
#     for node in children:
#         most_frequent_span = combine_spans_utils.get_most_frequent_span(node.span_lst)
#         if most_frequent_span in span_to_node_dict:
#             np_object = span_to_node_dict[most_frequent_span]
#             np_object.combine_nodes(node)
#             continue
#         span_to_node_dict[most_frequent_span] = node
#         span_lst.append(most_frequent_span)
#     most_frequent_span = combine_spans_utils.get_most_frequent_span(parent.span_lst)
#     with torch.no_grad():
#         encoded_input_node = \
#             combineSpans.sapBert_tokenizer(most_frequent_span, return_tensors='pt', padding=True).to(
#                 combineSpans.device)
#         phrase_embeddings_node = combineSpans.model(**encoded_input_node).last_hidden_state.cpu()
#         phrase_embeddings_node = torch.transpose(phrase_embeddings_node, 0, 1)
#         phrase_embeddings_node = phrase_embeddings_node[0, :]
#         encoded_input = \
#             combineSpans.sapBert_tokenizer(span_lst, return_tensors='pt', padding=True).to(combineSpans.device)
#         phrase_embeddings = combineSpans.model(**encoded_input).last_hidden_state.cpu()
#         phrase_embeddings = torch.transpose(phrase_embeddings, 0, 1)
#         phrase_embeddings = phrase_embeddings[0, :]
#         del encoded_input, encoded_input_node
#         torch.cuda.empty_cache()
#     equivalents_children = set()
#     for j in range(len(span_lst)):
#         # cos_sim(u, v) = 1 - cos_dist(u, v)
#         res = cos(phrase_embeddings_node.reshape(-1, 1), phrase_embeddings[j].reshape(-1, 1))
#         if res > 0.90:
#             equivalents_children.add(span_to_node_dict[span_lst[j]])
#             parent.children.remove(span_to_node_dict[span_lst[j]])
#     if equivalents_children:
#         combine_spans_utils.combine_node_with_equivalent_children(parent, equivalents_children, span_to_object,
#                                                                   dict_object_to_global_label,
#                                                                   global_dict_label_to_object, topic_object_lst,
#                                                                   visited)


def combine_equivalent_node_with_its_equivalent_children(parent, children, span_to_object, dict_object_to_global_label,
                                                         global_dict_label_to_object, topic_object_lst, span_to_vector,
                                                         visited):
    if len(children) == 0:
        return
    equivalents_children = set()
    for child in children:
        res = cos(parent.weighted_average_vector, child.weighted_average_vector)
        if res > 0.95:
            equivalents_children.add(child)
    if equivalents_children:
        combine_spans_utils.combine_node_with_equivalent_children(parent, equivalents_children, span_to_object,
                                                                  dict_object_to_global_label,
                                                                  global_dict_label_to_object, topic_object_lst,
                                                                  span_to_vector, visited)


def combine_equivalent_parent_and_children_nodes_by_semantic_DL_model(np_object_lst, span_to_object,
                                                                      dict_object_to_global_label,
                                                                      global_dict_label_to_object,
                                                                      topic_object_lst, span_to_vector, visited=set()):
    for np_object in np_object_lst:
        if np_object in visited:
            continue
        visited.add(np_object)
        combine_equivalent_parent_and_children_nodes_by_semantic_DL_model(np_object.children, span_to_object,
                                                                          dict_object_to_global_label,
                                                                          global_dict_label_to_object,
                                                                          topic_object_lst, span_to_vector, visited)
        combine_equivalent_node_with_its_equivalent_children(np_object, np_object.children, span_to_object,
                                                             dict_object_to_global_label,
                                                             global_dict_label_to_object, topic_object_lst,
                                                             span_to_vector, visited)


def combine_equivalent_nodes_by_semantic_DL_model(np_object_lst, span_to_object, dict_object_to_global_label,
                                                  global_dict_label_to_object, span_to_vector, visited=set()):
    for np_object in np_object_lst:
        if np_object in visited:
            continue
        visited.add(np_object)
        combine_equivalent_nodes(np_object.children, span_to_object, dict_object_to_global_label,
                                 global_dict_label_to_object, span_to_vector)
        combine_equivalent_nodes_by_semantic_DL_model(np_object.children, span_to_object, dict_object_to_global_label,
                                                      global_dict_label_to_object, span_to_vector, visited)
