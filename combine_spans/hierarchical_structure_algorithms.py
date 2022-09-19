import heapq
from itertools import combinations
import torch
from combine_spans import utils as combine_spans_utils

cos = torch.nn.CosineSimilarity(dim=0, eps=1e-08)

def dfs_update_marginal_gain(visited, node, dist_matrix, k, dist=1):  # function for dfs
    if node not in visited:
        visited.append(node)
        for neighbour in node.children:
            if dist < dist_matrix.get(hash(k) - hash(neighbour), float('inf')):
                dist_matrix[hash(k) - hash(neighbour)] = dist
            dfs_update_marginal_gain(visited, neighbour, dist_matrix, k, dist + 1)


def dfs(visited, node):  # function for dfs
    if node not in visited:
        visited.append(node)
        for neighbour in node.children:
            dfs(visited, neighbour)


def calculate_dist_from_set_to_vertex(S, v, dist_matrix):
    min_dist = float('inf')
    for u in S:
        dist_u_v = dist_matrix.get(hash(str(hash(u))) - hash(str(hash(v))), float('inf'))
        if dist_u_v < min_dist:
            min_dist = dist_u_v
    return min_dist


def get_rep_from_group(S, y, dist_matrix, global_index_to_similar_longest_np):
    dist = calculate_dist_from_set_to_vertex(S, y, dist_matrix)
    return get_rep(y, dist, global_index_to_similar_longest_np)


def get_rep(y, dist, global_index_to_similar_longest_np, x):
    rep = get_value_by_cosineSimilarity_format(dist, global_index_to_similar_longest_np, y, x)
    return rep


def calculate_marginal_gain(x, dist_matrix, S_rep, k, dict_object_to_desc,
                            global_index_to_similar_longest_np):
    marginal_val = 0
    S_rep_new = {}
    for y in dict_object_to_desc[hash(x)]:
        if dist_matrix.get(hash(k) - hash(y), 1) == 0:
            continue
        else:
            marginal_val_y = S_rep.get(hash(y), 0)
            if x == y:
                gain_x = get_value_by_cosineSimilarity_format(0, global_index_to_similar_longest_np, x, x)
                S_rep_new[hash(y)] = gain_x
                marginal_val += (gain_x - marginal_val_y)
            else:
                dist = dist_matrix[hash(str(hash(x))) - hash(str(hash(y)))]
                S_rep_new[hash(y)] = get_rep(y, dist, global_index_to_similar_longest_np, x)
                marginal_val += (S_rep_new[hash(y)] - marginal_val_y)
    return marginal_val, S_rep_new


def get_value_in_score_format(global_index_to_similar_longest_np, current_node, main_node):
    label_lst = combine_spans_utils.get_labels_of_children(current_node.children)
    label_lst_minus_children_labels = current_node.label_lst - label_lst
    cos_similarity_val = cos(current_node.weighted_average_vector, main_node.weighted_average_vector)
    marginal_gain = combine_spans_utils.get_frequency_from_labels_lst(global_index_to_similar_longest_np,
                                                                      label_lst_minus_children_labels) * cos_similarity_val
    return marginal_gain


def get_value_by_cosineSimilarity_format(dist, global_index_to_similar_longest_np, node):
    label_lst = combine_spans_utils.get_labels_of_children(node.children)
    label_lst_minus_children_labels = node.label_lst - label_lst
    # labels_children = label_lst - label_lst_minus_children_labels
    leaves_labels = combine_spans_utils.get_labels_of_leaves(node.children)
    marginal_gain = combine_spans_utils.get_frequency_from_labels_lst(global_index_to_similar_longest_np,
                                                                      label_lst_minus_children_labels) ** 2
    marginal_gain += combine_spans_utils.get_frequency_from_labels_lst(global_index_to_similar_longest_np,
                                                                       leaves_labels) * (node.score - 1)
    marginal_gain += combine_spans_utils.get_frequency_from_labels_lst(global_index_to_similar_longest_np,
                                                                       label_lst - leaves_labels) * (node.score - 1)
    marginal_gain = marginal_gain / (dist + 1)
    return marginal_gain




def get_value_in_pow_format(dist, global_index_to_similar_longest_np, label_lst_node, children):
    label_lst = combine_spans_utils.get_labels_of_children(children)
    label_lst = label_lst_node - label_lst
    marginal_gain = (combine_spans_utils.get_frequency_from_labels_lst
                     (global_index_to_similar_longest_np, label_lst) ** 2) / (dist + 1)
    return marginal_gain


def compute_value_for_each_node(x, dist_matrix, dict_object_to_desc, dict_node_to_rep,
                                global_index_to_similar_longest_np):
    Q = [x]
    visited = [x]
    dist_matrix[hash(x)] = 0
    dict_object_to_desc[hash(x)] = []
    rep_matrix = {}
    counter = 0
    total_gain = 0
    while Q:
        v = Q.pop()
        dict_object_to_desc[hash(x)].append(v)
        if x == v:
            x_v = hash(x)
        else:
            x_v = hash(str(hash(x))) - hash(str(hash(v)))
        dist_matrix[x_v] = dist_matrix.get(x_v, 0)
        for u in v.children:
            if u not in visited:
                x_u = hash(str(hash(x))) - hash(str(hash(u)))
                dist_matrix[x_u] = dist_matrix[x_v] + 1
                x_u_marginal_gain = get_value_in_score_format(dist_matrix[x_u], global_index_to_similar_longest_np, u, x)
                rep_matrix[x_u] = x_u_marginal_gain
                total_gain += x_u_marginal_gain
                counter += 1
                Q.append(u)
                visited.append(u)
    total_gain += get_value_by_cosineSimilarity_format(0, global_index_to_similar_longest_np, x, x)
    rep_matrix[hash(x)] = total_gain
    dict_node_to_rep[list(x.np_val)[0]] = rep_matrix
    return total_gain


def get_all_group_with_intersection_greater_than_X(selected_np_objects, threshold_intersection=0.7):
    objects_set_more_than_threshold_intersection = []
    for comboSize in range(2, len(selected_np_objects)):
        for combo in combinations(range(len(selected_np_objects)), comboSize):
            intersection = selected_np_objects[combo[0]].label_lst
            intersection_set = [selected_np_objects[combo[0]]]
            max_object_idx = max(combo[1:], key=lambda idx: len(selected_np_objects[idx].label_lst))
            max_labels_val = max(len(selected_np_objects[max_object_idx].label_lst), len(intersection))
            for i in combo[1:]:
                intersection = intersection & selected_np_objects[i].label_lst
                intersection_set.append(selected_np_objects[i])
            if len(intersection) > threshold_intersection * max_labels_val:
                objects_set_more_than_threshold_intersection.append(intersection_set)
    return objects_set_more_than_threshold_intersection


common_span_remove_lst = []


def remove_unselected_np_objects(parent_np_object, selected_np_objects, visited_nodes):
    # list of unselected children to remove from parent
    remove_lst = []
    for child in parent_np_object.children:
        if child not in selected_np_objects and child not in visited_nodes:
            remove_lst.append(child)
            try:
                child.parents.remove(parent_np_object)
            except:
                print(parent_np_object)
    for node in remove_lst:
        if node.frequency > 1:
            common_span_remove_lst.append(node)
    for np_object in remove_lst:
        parent_np_object.children.remove(np_object)


def set_cover(children, visited_labels, np_object_parent, global_index_to_similar_longest_np):
    covered = set()
    covered.update(visited_labels)
    selected_np_objects = []
    counted_labels = set()
    while True:
        # np_object = max(children, key=lambda np_object: len(
        #     np_object_parent.label_lst.intersection(np_object.label_lst - covered)), default=None)
        np_object = max(children, key=lambda np_object: combine_spans_utils.get_frequency_from_labels_lst(
            global_index_to_similar_longest_np, np_object.label_lst - covered), default=None)
        if not np_object:
            break
        if combine_spans_utils.get_frequency_from_labels_lst(
                global_index_to_similar_longest_np, np_object.label_lst - covered) > 1:
            counted_labels.update(np_object_parent.label_lst.intersection(np_object.label_lst - visited_labels))
            selected_np_objects.append(np_object)
            covered.update(np_object.label_lst - visited_labels)
        else:
            break
    return selected_np_objects, counted_labels


def add_longest_nps_to_np_object_children(topic_object, labels, global_dict_label_to_object):
    longest_nps_lst = set()
    for label in labels:
        longest_nps_lst.add(global_dict_label_to_object[label])
    topic_object.add_children(longest_nps_lst)
    for longest_np in longest_nps_lst:
        longest_np.parents.add(topic_object)


def sort_DAG_by_frequency(topic_object_lst, visited=[]):
    for topic_object in topic_object_lst:
        if topic_object in visited:
            continue
        visited.append(topic_object)
        topic_object.children = sorted(topic_object.children, key=lambda item: item.frequency, reverse=True)
        sort_DAG_by_frequency(topic_object.children, visited)


def get_k_navigable_DAGS_from_DAG(k, topic_object_lst, global_dict_label_to_object, global_index_to_similar_longest_np):
    topic_object_lst = sorted(topic_object_lst, key=lambda item: item.frequency, reverse=True)
    visited_nodes = set()
    for topic_object in topic_object_lst:
        build_tree_from_DAG(topic_object, global_dict_label_to_object, k, visited_nodes,
                            global_index_to_similar_longest_np, set())
    sort_DAG_by_frequency(topic_object_lst)


def get_labels_from_visited_children(children, visited_nodes):
    visited_labels = set()
    for child in children:
        if child in visited_nodes:
            visited_labels.update(child.label_lst)
    return visited_labels


def build_tree_from_DAG(np_object, global_dict_label_to_object, k, visited_nodes,
                        global_index_to_similar_longest_np, visited_labels):
    if not np_object.children:
        return
    labels_covered_by_children = combine_spans_utils.get_labels_of_children(np_object.children)
    labels_covered_by_parent = np_object.label_lst - labels_covered_by_children
    visited_labels.update(labels_covered_by_parent)
    visited_labels.update(get_labels_from_visited_children(np_object.children, visited_nodes))
    all_labels = np_object.label_lst - visited_labels
    unvisited_nodes = set(np_object.children) - visited_nodes
    selected_np_objects, counted_labels = set_cover(unvisited_nodes, visited_labels, np_object,
                                                    global_index_to_similar_longest_np)
    remove_unselected_np_objects(np_object, selected_np_objects, visited_nodes)
    uncounted_labels = all_labels - counted_labels
    visited_labels.update(uncounted_labels)
    add_longest_nps_to_np_object_children(np_object, uncounted_labels, global_dict_label_to_object)
    if selected_np_objects:
        for np_object_child in selected_np_objects:
            visited_nodes.add(np_object_child)
            build_tree_from_DAG(np_object_child, global_dict_label_to_object, k, visited_nodes,
                                global_index_to_similar_longest_np, visited_labels)


def is_ancestor_in_S(v, S, visited):
    visited.add(v)
    for parent in v.parents:
        if parent in visited:
            continue
        if parent in S:
            return True
        is_ancestor = is_ancestor_in_S(parent, S, visited)
        if is_ancestor:
            return True
    return False



def greedy_algorithm(k, topic_lst, global_index_to_similar_longest_np):
    dist_matrix = {}
    dict_object_to_desc = {}
    dict_node_to_rep = {}
    all_object_np_lst = []
    for node in topic_lst:
        dfs(all_object_np_lst, node)
    all_labels = set()
    for node in all_object_np_lst:
        all_labels.update(node.label_lst)
        node.marginal_val = compute_value_for_each_node(node, dist_matrix, dict_object_to_desc, dict_node_to_rep,
                                                        global_index_to_similar_longest_np)
    S = []
    heap_data_structure = all_object_np_lst
    heapq.heapify(heap_data_structure)
    already_counted_labels = set()
    S_rep = {}
    counter = 0
    while len(S) < k and heap_data_structure:
        x = heapq.heappop(heap_data_structure)
        is_ancestor_already_in_S = is_ancestor_in_S(x, S, set())
        if is_ancestor_already_in_S:
            continue
        # is_fully_contained = False
        # for np_object in S:
        #     if len(np_object.label_lst.intersection(x.label_lst)) == len(np_object.label_lst) or \
        #             len(x.label_lst.intersection(np_object.label_lst)) == len(x.label_lst):
        #         is_fully_contained = True
        #         break
        # if is_fully_contained:
        #     continue
        marginal_val_x, S_rep_new = calculate_marginal_gain(x, dist_matrix, S_rep, k,
                                                            dict_object_to_desc, global_index_to_similar_longest_np)
        if x.marginal_val > marginal_val_x + 0.1:
            x.marginal_val = marginal_val_x
            heapq.heappush(heap_data_structure, x)
            continue
        uncounted_labels = []
        for label in x.label_lst:
            if label not in already_counted_labels:
                uncounted_labels.append(label)
        # uncounted_labels_frequency = combine_spans_utils.get_frequency_from_labels_lst(
        #     global_index_to_similar_longest_np, uncounted_labels)
        # if uncounted_labels_frequency < 5:
        #     continue
        already_counted_labels.update(x.label_lst)
        for key, value in S_rep_new.items():
            S_rep[key] = value
        S.append(x)
        dist_matrix[hash(k) - hash(x)] = 0
        counter += 1
        dfs_update_marginal_gain([], x, dist_matrix, k)
    return S, already_counted_labels, all_labels
