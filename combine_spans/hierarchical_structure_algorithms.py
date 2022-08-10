import heapq


# def marginal_gain():


def dfs(visited, node, dist_matrix, k, dist=1):  # function for dfs
    if node not in visited:
        visited.append(node)
        for neighbour in node.children:
            # if dist_matrix[hash(str(hash(node))) - hash(str(hash(neighbour)))] + dist < dist_matrix.get(hash(k) - hash(neighbour),
            #                                                                       float('inf')):
            if dist < dist_matrix.get(hash(k) - hash(neighbour), float('inf')):
                # dist_matrix[hash(k) - hash(neighbour)] = dist_matrix[hash(str(hash(node))) - hash(str(hash(neighbour)))] + dist
                dist_matrix[hash(k) - hash(neighbour)] = dist
            dfs(visited, neighbour, dist_matrix, k, dist + 1)


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
    rep = max((len(y.label_lst) - 2) / (dist + 1), 0)
    return rep


def calculate_marginal_gain(S, x, dist_matrix, S_rep, nodes_lst, already_counted_labels, k, dict_object_to_desc,
                            dict_node_to_rep):
    marginal_val = 0
    S_rep_new = {}
    # if not S:
    #     S_rep_new[hash(x)] = x.marginal_val
    #     return x.marginal_val, S_rep_new
    rep_matrix = dict_node_to_rep[x.np_val]
    is_root = True
    # if x not in nodes_lst:
    #     is_root = False
    #     for label in x.label_lst:
    #         if label not in already_counted_labels:
    #             marginal_val += 1
    S_temp = S + [x]
    for y in dict_object_to_desc[hash(x)]:
        if dist_matrix.get(hash(k) - hash(y), 0):
            marginal_val_y = S_rep[hash(y)]
            if is_root:
                if marginal_val_y:
                    marginal_val_y_with_x = get_rep_from_group(S_temp, y, dist_matrix, already_counted_labels)
                    S_rep_new[hash(y)] = marginal_val_y_with_x
                    marginal_val += (marginal_val_y_with_x - marginal_val_y)
            else:
                marginal_val -= marginal_val_y
                S_rep_new[hash(y)] = 0
        else:
            if is_root:
                if x == y:
                    S_rep_new[hash(y)] = len(x.np)
                    marginal_val += len(x.np)
                else:
                    dist = dist_matrix[hash(str(hash(x))) - hash(str(hash(y)))]
                    # rep_original = rep_matrix[hash(str(hash(x))) - hash(str(hash(y)))]
                    S_rep_new[hash(y)] = get_rep(y, dist, already_counted_labels)
                    # if rep_original != S_rep_new[hash(y)]:
                    #     print(y.np_val)
                    marginal_val += S_rep_new[hash(y)]
            else:
                S_rep_new[hash(y)] = 0
    if not is_root:
        S_rep_new[hash(x)] = marginal_val
    return marginal_val, S_rep_new


def compute_value_for_each_node(x, dist_matrix, dict_object_to_desc, compute_value_for_each_node):
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
                distance_x_given_S = distance_x_given_S + max((len(u.label_lst) - 2) / (dist_matrix[x_u] + 1), 0)
                rep_matrix[x_u] = (len(u.label_lst) - 1) / (dist_matrix[x_u] + 1)
                # (1 + dist_matrix.get(hash(v) + hash(u), float('inf')))
                counter += 1
                Q.append(u)
                visited.append(u)
    distance_x_given_S += len(x.np)
    compute_value_for_each_node[x.np_val] = rep_matrix
    return distance_x_given_S


# def compute_value_for_each_node(x):
#     distance_x_given_S = 0
#     for node in x.children:
#         distance_x_given_S = distance_x_given_S + (len(node.label_lst) -1) / 2
#     distance_x_given_S += len(x.np) * 0.5
#     return distance_x_given_S


def greedy_algorithm(k, nodes_lst, all_object_np_lst):
    dist_matrix = {}
    dict_object_to_desc = {}
    # heap_data_structure = nodes_lst + all_object_np_lst
    heap_data_structure = nodes_lst
    dict_node_to_rep = {}
    for node in heap_data_structure:
        node.marginal_val = compute_value_for_each_node(node, dist_matrix, dict_object_to_desc, dict_node_to_rep)
    # for node in all_object_np_lst:
    #     node.marginal_val = len(node.label_lst)
    #     heap_data_structure.append(node)
    S = []
    heapq.heapify(heap_data_structure)
    already_counted_labels = []
    S_rep = {}
    counter = 0
    while len(S) < k:
        x = heapq.heappop(heap_data_structure)
        marginal_val_x, S_rep_new = calculate_marginal_gain(S, x, dist_matrix, S_rep, nodes_lst, already_counted_labels,
                                                            k, dict_object_to_desc, dict_node_to_rep)
        if x.marginal_val > marginal_val_x + 0.1:
            x.marginal_val = marginal_val_x
            heapq.heappush(heap_data_structure, x)
            continue
        for key, value in S_rep_new.items():
            S_rep[key] = value
        already_counted_labels.extend(x.label_lst)
        S.append(x)
        dist_matrix[hash(k) - hash(x)] = 0
        counter += 1
        dfs([], x, dist_matrix, k)
    x = 0
