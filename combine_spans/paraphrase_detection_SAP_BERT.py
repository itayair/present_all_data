from transformers import AutoTokenizer, AutoModel
import torch
import pickle
from sklearn.cluster import AgglomerativeClustering
from combine_spans import utils as combine_spans_utils
from combine_spans import combineSpans as combineSpans


def combine_equivalent_nodes(np_object_lst, span_to_object, dict_object_to_global_label, global_dict_label_to_object):
    if len(np_object_lst) <= 1:
        return
    span_to_node_dict = {}
    span_lst = []
    for node in np_object_lst:
        most_frequent_span = combine_spans_utils.get_most_frequent_span(node.span_lst)
        if most_frequent_span in span_to_node_dict:
            np_object = span_to_node_dict[most_frequent_span]
            np_object.combine_nodes(node)
            continue
        span_to_node_dict[most_frequent_span] = node
        span_lst.append(most_frequent_span)
    with torch.no_grad():
        encoded_input = \
            combineSpans.sapBert_tokenizer(span_lst, return_tensors='pt', padding=True).to(combineSpans.device)
        phrase_embeddings = combineSpans.model(**encoded_input).last_hidden_state.cpu()
        phrase_embeddings = torch.transpose(phrase_embeddings, 0, 1)
        phrase_embeddings = phrase_embeddings[0, :]
        del encoded_input
        torch.cuda.empty_cache()
    clustering = AgglomerativeClustering(distance_threshold=0.1, n_clusters=None, linkage="average",
                                         affinity="cosine", compute_full_tree=True).fit(phrase_embeddings)
    dict_cluster_to_common_spans_lst = {}
    for idx, label in enumerate(clustering.labels_):
        dict_cluster_to_common_spans_lst[label] = dict_cluster_to_common_spans_lst.get(label, [])
        np_object = span_to_node_dict.get(span_lst[idx], None)
        dict_cluster_to_common_spans_lst[label].append(np_object)
    for label, equivalent_nodes in dict_cluster_to_common_spans_lst.items():
        if len(equivalent_nodes) == 1:
            continue
        combine_spans_utils.combine_nodes_lst(equivalent_nodes, span_to_object, dict_object_to_global_label,
                                              global_dict_label_to_object)


def combine_equivalent_nodes_by_semantic_DL_model(np_object_lst, span_to_object, dict_object_to_global_label,
                                                  global_dict_label_to_object, visited=set()):
    for np_object in np_object_lst:
        if np_object in visited:
            continue
        visited.add(np_object)
        combine_equivalent_nodes(np_object.children, span_to_object, dict_object_to_global_label,
                                 global_dict_label_to_object)
        combine_equivalent_nodes_by_semantic_DL_model(np_object.children, span_to_object, dict_object_to_global_label,
                                                      global_dict_label_to_object, visited)

