import heapq
from itertools import combinations
from combine_spans import utils as combine_spans_utils


# def marginal_gain():


def dfs_update_marginal_gain(visited, node, dist_matrix, k, dist=1):  # function for dfs
    if node not in visited:
        visited.append(node)
        for neighbour in node.children:
            # if dist_matrix[hash(str(hash(node))) - hash(str(hash(neighbour)))] + dist < dist_matrix.get(hash(k) - hash(neighbour),
            #                                                                       float('inf')):
            if dist < dist_matrix.get(hash(k) - hash(neighbour), float('inf')):
                # dist_matrix[hash(k) - hash(neighbour)] = dist_matrix[hash(str(hash(node))) - hash(str(hash(neighbour)))] + dist
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


def get_rep_from_group(S, y, dist_matrix, already_counted_labels):
    dist = calculate_dist_from_set_to_vertex(S, y, dist_matrix)
    return get_rep(y, dist, already_counted_labels)


def get_rep(y, dist, already_counted_labels):
    # non_counted_labels = []
    # for label in y.label_lst:
    #     non_counted_labels.append(label)
    rep = (len(y.label_lst) ** 2) / (dist + 1)
    return rep


def calculate_marginal_gain(x, dist_matrix, S_rep, already_counted_labels, k, dict_object_to_desc, topic_lst):
    marginal_val = 0
    S_rep_new = {}
    for y in dict_object_to_desc[hash(x)]:
        if dist_matrix.get(hash(k) - hash(y), 1) == 0:
            continue
            # marginal_val_y = S_rep[hash(y)]
            # marginal_val_y_with_x = get_rep_from_group(S_temp, y, dist_matrix, already_counted_labels)
            # S_rep_new[hash(y)] = marginal_val_y_with_x
            # marginal_val += (marginal_val_y_with_x - marginal_val_y)
        else:
            marginal_val_y = S_rep.get(hash(y), 0)
            if x == y:
                if x in topic_lst:
                    label_lst = combine_spans_utils.get_labels_of_children(x.children)
                    label_lst = label_lst - x.label_lst
                    gain_x = len(label_lst)**2
                    # distance_x_given_S += 1
                else:
                    gain_x = len(x.label_lst)**2
                S_rep_new[hash(y)] = gain_x
                marginal_val += (gain_x - marginal_val_y)
            else:
                dist = dist_matrix[hash(str(hash(x))) - hash(str(hash(y)))]
                S_rep_new[hash(y)] = get_rep(y, dist, already_counted_labels)
                marginal_val += (S_rep_new[hash(y)] - marginal_val_y)
    return marginal_val, S_rep_new


def compute_value_for_each_node(x, dist_matrix, dict_object_to_desc, dict_node_to_rep, topic_lst):
    Q = [x]
    visited = [x]
    distance_x_given_S = 0
    dist_matrix[hash(x)] = 0
    dict_object_to_desc[hash(x)] = []
    rep_matrix = {}
    counter = 0
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
                distance_x_given_S = distance_x_given_S + (len(u.label_lst) ** 2) / (dist_matrix[x_u] + 1)
                rep_matrix[x_u] = (len(u.label_lst) - 1) / (dist_matrix[x_u] + 1)
                # (1 + dist_matrix.get(hash(v) + hash(u), float('inf')))
                counter += 1
                Q.append(u)
                visited.append(u)
    if x in topic_lst:
        label_lst = combine_spans_utils.get_labels_of_children(x.children)
        label_lst = label_lst - x.label_lst
        distance_x_given_S += len(label_lst) **2
        # distance_x_given_S += 1
    else:
        distance_x_given_S += len(x.label_lst)**2
    dict_node_to_rep[list(x.np_val)[0]] = rep_matrix
    return distance_x_given_S


# def compute_value_for_each_node(x):
#     distance_x_given_S = 0
#     for node in x.children:
#         distance_x_given_S = distance_x_given_S + (len(node.label_lst) -1) / 2
#     distance_x_given_S += len(x.np) * 0.5
#     return distance_x_given_S


def get_all_group_with_intersection_greater_than_X(selected_np_objects, threshold_intersection=0.7):
    objects_set_more_than_threshold_intersection = []
    for comboSize in range(2, len(selected_np_objects)):
        for combo in combinations(range(len(selected_np_objects)), comboSize):
            intersection = selected_np_objects[combo[0]].label_lst
            intersection_set = [selected_np_objects[combo[0]]]
            max_object_idx = max(combo[1:], key=lambda idx: len(selected_np_objects[idx].label_lst))
            max_labels_val = max(len(selected_np_objects[max_object_idx].label_lst), len(intersection))
            for i in combo[1:]:
                # if len(selected_np_objects[i].label_lst) > max_labels_val:
                #     max_labels_val = len(selected_np_objects[i].label_lst)
                intersection = intersection & selected_np_objects[i].label_lst
                intersection_set.append(selected_np_objects[i])
            if len(intersection) > threshold_intersection * max_labels_val:
                objects_set_more_than_threshold_intersection.append(intersection_set)
    return objects_set_more_than_threshold_intersection


def remove_unselected_np_objects(parent_np_object, selected_np_objects, visited_nodes):
    # list of unselected children to remove from parent
    remove_lst = []
    for child in parent_np_object.children:
        if child not in selected_np_objects and child not in visited_nodes:
            remove_lst.append(child)
            child.parents.remove(parent_np_object)
    for np_object in remove_lst:
        parent_np_object.children.remove(np_object)


def set_cover(children, visited_labels, np_object_parent):
    covered = set()
    covered.update(visited_labels)
    selected_np_objects = []
    counted_labels = set()
    while True:
        np_object = max(children, key=lambda np_object: len(
            np_object_parent.label_lst.intersection(np_object.label_lst - covered)), default=None)
        if np_object == None:
            break
        if len(np_object.label_lst - covered) > 1:
            counted_labels.update(np_object_parent.label_lst.intersection(np_object.label_lst - visited_labels))
            selected_np_objects.append(np_object)
            covered.update(np_object.label_lst - visited_labels)
        else:
            break
        # if len(np_object_parent.label_lst - covered) == 0:
        #     break
    return selected_np_objects, counted_labels


def add_longest_nps_to_np_object_children(topic_object, labels, global_dict_label_to_object):
    longest_nps_lst = set()
    for label in labels:
        longest_nps_lst.add(global_dict_label_to_object[label])
    topic_object.children.update(longest_nps_lst)
    for longest_np in longest_nps_lst:
        longest_np.parents.add(topic_object)


def get_k_trees_from_DAG(k, topic_object_lst, global_dict_label_to_object):
    # if len(topic_object_lst) == 1:
    #     if topic_object_lst.children:
    #         topic_object_lst.children = []
    #         add_longest_nps_to_np_object_children(topic_object_lst, topic_object_lst.label_lst,
    #                                               global_dict_label_to_object)
    #     return
    topic_object_lst = sorted(topic_object_lst, key=lambda item: len(item.label_lst), reverse=True)
    visited_nodes = set()
    # visited_labels = set()
    for topic_object in topic_object_lst:
        build_tree_from_DAG(topic_object, global_dict_label_to_object, k, visited_nodes, set())


def get_labels_from_visited_children(children, visited_nodes):
    visited_labels = set()
    for child in children:
        if child in visited_nodes:
            visited_labels.update(child.label_lst)
    return visited_labels


def build_tree_from_DAG(np_object, global_dict_label_to_object, k, visited_nodes, visited_labels):
    if not np_object.children:
        return
    labels_covered_by_children = combine_spans_utils.get_labels_of_children(np_object.children)
    labels_covered_by_parent = np_object.label_lst - labels_covered_by_children
    visited_labels.update(labels_covered_by_parent)
    visited_labels.update(get_labels_from_visited_children(np_object.children, visited_nodes))
    all_labels = np_object.label_lst - visited_labels
    unvisited_nodes = set(np_object.children) - visited_nodes
    selected_np_objects, counted_labels = set_cover(unvisited_nodes, visited_labels, np_object)
    # list of groups of objects with intersection greater than a threshold
    # objects_set_intersection = get_all_group_with_intersection_greater_than_X(
    #     selected_np_objects)
    remove_unselected_np_objects(np_object, selected_np_objects, visited_nodes)
    uncounted_labels = all_labels - counted_labels
    visited_labels.update(uncounted_labels)
    add_longest_nps_to_np_object_children(np_object, uncounted_labels, global_dict_label_to_object)
    if selected_np_objects:
        for np_object_child in selected_np_objects:
            visited_nodes.add(np_object_child)
            build_tree_from_DAG(np_object_child, global_dict_label_to_object, k, visited_nodes, visited_labels)


def greedy_algorithm(k, topic_lst):
    dist_matrix = {}
    dict_object_to_desc = {}
    dict_node_to_rep = {}
    # all_object_np_lst = topic_lst.copy()
    all_object_np_lst = []
    for node in topic_lst:
        dfs(all_object_np_lst, node)
    all_labels = set()
    for node in all_object_np_lst:
        all_labels.update(node.label_lst)
        node.marginal_val = compute_value_for_each_node(node, dist_matrix, dict_object_to_desc, dict_node_to_rep,
                                                        topic_lst)
    # for node in all_object_np_lst:
    #     node.marginal_val = len(node.label_lst)
    #     heap_data_structure.append(node)
    S = []
    heap_data_structure = all_object_np_lst
    heapq.heapify(heap_data_structure)
    already_counted_labels = []
    S_rep = {}
    counter = 0
    while len(S) < k and heap_data_structure:
        x = heapq.heappop(heap_data_structure)
        is_fully_contained = False
        for np_object in S:
            if len(np_object.label_lst.intersection(x.label_lst)) == len(np_object.label_lst) or \
                    len(x.label_lst.intersection(np_object.label_lst)) == len(x.label_lst):
                is_fully_contained = True
                break
        if is_fully_contained:
            continue
        marginal_val_x, S_rep_new = calculate_marginal_gain(x, dist_matrix, S_rep, already_counted_labels,
                                                            k, dict_object_to_desc, topic_lst)
        if x.marginal_val > marginal_val_x + 0.1:
            x.marginal_val = marginal_val_x
            heapq.heappush(heap_data_structure, x)
            continue
        uncounted_labels_counter = 0
        for label in x.label_lst:
            if label not in already_counted_labels:
                uncounted_labels_counter += 1
        if uncounted_labels_counter < 5:
            continue
        already_counted_labels.extend(x.label_lst)
        for key, value in S_rep_new.items():
            S_rep[key] = value
        S.append(x)
        dist_matrix[hash(k) - hash(x)] = 0
        counter += 1
        dfs_update_marginal_gain([], x, dist_matrix, k)
    return S, already_counted_labels, all_labels
