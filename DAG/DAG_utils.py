from combine_spans import utils as combine_spans_utils
from transformers import AutoTokenizer, AutoModel
import DAG.NounPhraseObject as NounPhrase

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
medical_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')


def add_node_between_nodes(parent, child, new_node):
    parent.children.remove(child)
    child.parents.remove(parent)
    parent.children.append(new_node)
    child.parents.add(new_node)
    new_node.children.append(child)
    new_node.parents.add(parent)
    new_node.label_lst.update(child.label_lst)


def is_node_contained_in_another_node(node, ref_node):
    # for span_as_lemmas in node.list_of_span_as_lemmas_lst:
    #     for span_as_lemmas_ref in ref_node.list_of_span_as_lemmas_lst:
    if combine_spans_utils.is_similar_meaning_between_span(node.common_lemmas_in_spans, ref_node.common_lemmas_in_spans):
        return True
    return False


def locate_node_in_DAG(parent, new_node):
    update_lst = set()
    for child in parent.children:
        if is_node_contained_in_another_node(new_node, child):
            update_lst.add(child)
    if update_lst:
        for node in update_lst:
            add_node_between_nodes(parent, node, new_node)
        return
    parent.add_children([new_node])
    new_node.parents.add(parent)


# def add_NP_to_DAG_up_to_bottom(np_object_to_add, np_object, similar_np_object):
#     is_contained = False
#     if np_object == np_object_to_add:
#         return True
#     for span_as_lemmas in np_object_to_add.list_of_span_as_lemmas_lst:
#         for span_as_lemmas_ref in np_object.list_of_span_as_lemmas_lst:
#             if combine_spans_utils.is_similar_meaning_between_span(span_as_lemmas_ref, span_as_lemmas):
#                 is_contained = True
#                 if len(span_as_lemmas_ref) == len(span_as_lemmas):
#                     np_object.combine_nodes(np_object_to_add)
#                     similar_np_object[0] = np_object
#                     return True
#     if is_contained:
#         is_added = False
#         for child in np_object.children:
#             is_added |= add_NP_to_DAG_up_to_bottom(np_object_to_add, child, similar_np_object)
#             if similar_np_object[0]:
#                 return True
#         if not is_added:
#             if np_object_to_add in np_object.children:
#                 raise Exception("the node already inserted")
#             locate_node_in_DAG(np_object, np_object_to_add)
#         return True
#     return False


def add_NP_to_DAG_bottom_to_up(np_object_to_add, np_object, visited, similar_np_object, visited_and_added=set()):
    is_contained = False
    if np_object in visited_and_added:
        return True
    if np_object in visited:
        return False
    if np_object == np_object_to_add:
        return True
    visited.add(np_object)
    # for span_as_lemmas in np_object_to_add.list_of_span_as_lemmas_lst:
    #     for span_as_lemmas_ref in np_object.list_of_span_as_lemmas_lst:
    if combine_spans_utils.is_similar_meaning_between_span(np_object_to_add.common_lemmas_in_spans, np_object.common_lemmas_in_spans):
        is_contained = True
        if len(np_object_to_add.common_lemmas_in_spans) == len(np_object.common_lemmas_in_spans):
            np_object.combine_nodes(np_object_to_add)
            similar_np_object[0] = np_object
            return True
    if is_contained:
        is_added = False
        for parent in np_object.parents:
            is_added |= add_NP_to_DAG_bottom_to_up(np_object_to_add, parent, visited, similar_np_object, visited_and_added)
            if similar_np_object[0]:
                return True
        if not is_added:
            np_object_to_add.add_children([np_object])
            parents_remove_lst = set()
            for parent in np_object.parents:
                if parent in np_object_to_add.parents:
                    parents_remove_lst.add(parent)
                    continue
                if is_node_contained_in_another_node(parent, np_object_to_add):
                    np_object_to_add.parents.add(parent)
                    if np_object_to_add not in parent.children:
                        parent.children.append(np_object_to_add)
                    else:
                        print("this is wrong insertion")
                    parents_remove_lst.add(parent)
            for parent_object in parents_remove_lst:
                np_object.parents.remove(parent_object)
                parent_object.children.remove(np_object)
            np_object.parents.add(np_object_to_add)
        visited_and_added.add(np_object)
        return True
    return False


def create_np_object_from_np_collection(np_collection, dict_span_to_lemmas_lst, labels, span_to_object):
    tuple_np_lst = []
    for np in np_collection:
        tuple_np_lst.append((np, dict_span_to_lemmas_lst[np]))
    np_object = NounPhrase.NP(tuple_np_lst, labels)
    for np in np_collection:
        span_to_object[np] = np_object
    return np_object


def update_np_object(collection_np_object, np_collection, span_to_object, dict_span_to_lemmas_lst, labels):
    nps_to_update = set()
    for np in np_collection:
        np_object = span_to_object.get(np, None)
        if np_object == collection_np_object:
            continue
        if np_object:
            raise Exception("2 different objects to synonyms")
        nps_to_update.add(np)
    if nps_to_update:
        for np in nps_to_update:
            span_to_object[np] = collection_np_object
            collection_np_object.label_lst.update(labels)
            collection_np_object.span_lst.add(np)
            collection_np_object.list_of_span_as_lemmas_lst.append(dict_span_to_lemmas_lst[np])


def get_longest_np_nodes_contain_labels(label_lst, global_dict_label_to_object, np_object, dict_object_to_global_label):
    nodes_lst = set()
    uncounted_labels = set()
    for label in label_lst:
        node = global_dict_label_to_object.get(label, None)
        if node:
            nodes_lst.add(node)
        else:
            uncounted_labels.add(label)
    for label in uncounted_labels:
        global_dict_label_to_object[label] = np_object
    dict_object_to_global_label[hash(np_object)] = uncounted_labels
    return nodes_lst


def create_and_insert_nodes_from_sub_groups_of_spans(dict_score_to_collection_of_sub_groups,
                                                     dict_span_to_lemmas_lst,
                                                     all_object_np_lst, span_to_object,
                                                     dict_span_to_similar_spans, dict_label_to_spans_group,
                                                     global_dict_label_to_object, dict_object_to_global_label):
    np_objet_lst = []
    for score, np_to_labels_collection in dict_score_to_collection_of_sub_groups.items():
        for np_key, labels in np_to_labels_collection:
            np_collection = dict_span_to_similar_spans[np_key]
            for np in np_collection:
                np_object = span_to_object.get(np, None)
                if np_object:
                    # np_object.label_lst.update(labels)
                    update_np_object(np_object, np_collection, span_to_object, dict_span_to_lemmas_lst, labels)
                    break
            if np_object:
                continue
            np_object = create_np_object_from_np_collection(np_collection, dict_span_to_lemmas_lst, labels,
                                                            span_to_object)
            np_objet_lst.append(np_object)
    np_objet_lst = sorted(np_objet_lst, key=lambda np_object: np_object.length, reverse=True)
    for np_object in np_objet_lst:
        longest_nps_nodes_lst = get_longest_np_nodes_contain_labels(np_object.label_lst, global_dict_label_to_object,
                                                                    np_object, dict_object_to_global_label)
        if not longest_nps_nodes_lst:
            continue
        similar_np_object = [None]
        visited = set()
        visited_and_added = set()
        is_combined_with_exist_node = False
        np_object_temp = np_object
        for longest_nps_node in longest_nps_nodes_lst:
            add_NP_to_DAG_bottom_to_up(np_object_temp, longest_nps_node, visited, similar_np_object, visited_and_added)
            if similar_np_object[0]:
                if hash(np_object) in dict_object_to_global_label:
                    for label in dict_object_to_global_label[hash(np_object)]:
                        global_dict_label_to_object[label] = similar_np_object[0]
                    dict_object_to_global_label[hash(similar_np_object[0])] = \
                        dict_object_to_global_label.get(hash(similar_np_object[0]), set())
                    dict_object_to_global_label[hash(similar_np_object[0])].update(dict_object_to_global_label[hash(np_object)])
                    dict_object_to_global_label.pop(hash(np_object), None)
                is_combined_with_exist_node = True
                for span in similar_np_object[0].span_lst:
                    span_to_object[span] = similar_np_object[0]
                np_object_temp = similar_np_object[0]
                similar_np_object[0] = None
        if not is_combined_with_exist_node:
            all_object_np_lst.append(np_object)
    # longest_nps_nodes_lst = get_longest_np_nodes_contain_labels(topic_object.label_lst, global_dict_label_to_object)
    # visited = set()
    # is_combined_with_exist_node = False
    # for longest_nps_node in longest_nps_nodes_lst:
    #     add_NP_to_DAG_bottom_to_up(topic_object, longest_nps_node, visited, similar_np_object)
    #     if similar_np_object[0]:
    #         is_combined_with_exist_node = True
    #         for span in similar_np_object[0].span_lst:
    #             span_to_object[span] = similar_np_object[0]
    #         topic_object = similar_np_object[0]
    # if not is_combined_with_exist_node:
    #     all_object_np_lst.append(topic_object)

# def create_DAG_from_top_to_bottom(dict_score_to_collection_of_sub_groups,
#                                   dict_span_to_lemmas_lst,
#                                   all_object_np_lst, span_to_object,
#                                   dict_span_to_similar_spans, dict_label_to_spans_group,
#                                   global_dict_label_to_object, topic_object):
#     for score, np_to_labels_collection in dict_score_to_collection_of_sub_groups.items():
#         for np_key, labels in np_to_labels_collection:
#             np_collection = dict_span_to_similar_spans[np_key]
#             for np in np_collection:
#                 np_object = span_to_object.get(np, None)
#                 if np_object:
#                     update_np_object(np_object, np_collection, span_to_object, dict_span_to_lemmas_lst, labels)
#                     break
#             if np_object:
#                 continue
#             np_object = create_np_object_from_np_collection(np_collection, dict_span_to_lemmas_lst, labels,
#                                                             span_to_object)
#             similar_np_object = [None]
#             add_NP_to_DAG_up_to_bottom(np_object, topic_object, similar_np_object)
#             if not similar_np_object[0]:
#                 all_object_np_lst.append(np_object)
#             else:
#                 for span in similar_np_object[0].span_lst:
#                     span_to_object[span] = similar_np_object[0]
#     for label, longest_answers_tuple in dict_label_to_spans_group.items():
#         global_dict_label_to_object[label] = span_to_object[longest_answers_tuple[0][0]]
# for label, nps in dict_label_to_spans_group.items():
#     for np in nps:
#         np_object = span_to_object.get(np[0], None)
#         if np_object:
#             break
#     if np_object:
#         global_dict_label_to_object[label] = np_object
#         continue
#     has_single_token = False
#     np_object = NounPhrase.NP(nps, [label])
#     for np in nps:
#         if len(np[1]) == 1:
#             has_single_token = True
#             break
#     if has_single_token:
#         topic_object.combine_nodes(np_object)
#         global_dict_label_to_object[label] = topic_object
#         for np in nps:
#             span_to_object[np[0]] = topic_object
#         continue
#     similar_np_object = [None]
#     add_NP_to_DAG_up_to_bottom(np_object, topic_object, similar_np_object)
#     if similar_np_object[0]:
#         np_object = similar_np_object[0]
#     else:
#         all_object_np_lst.append(np_object)
#     global_dict_label_to_object[label] = np_object
#     for np in np_object.np_val:
#         span_to_object[np] = np_object
# return topic_object


def remove_topic_np_from_np_object(np_object, topic_np):
    lemmas_to_remove = None
    for span_as_lemmas_lst in np_object.list_of_span_as_lemmas_lst:
        if len(span_as_lemmas_lst) == 1 and topic_np in span_as_lemmas_lst:
            lemmas_to_remove = span_as_lemmas_lst
            break
    if lemmas_to_remove:
        np_object.list_of_span_as_lemmas_lst.remove(lemmas_to_remove)


def combine_topic_object(np_object, topic_object):
    if combine_spans_utils.is_similar_meaning_between_span(topic_object.common_lemmas_in_spans, np_object.common_lemmas_in_spans):
        if len(topic_object.common_lemmas_in_spans) == len(np_object.common_lemmas_in_spans):
            np_object.combine_nodes(topic_object)
            return True
    return False


def create_and_update_topic_object(topic_synonym_lst, span_to_object, longest_NP_to_global_index):
    topic_synonyms_tuples = [(synonym, [synonym]) for synonym in topic_synonym_lst]
    topic_object = NounPhrase.NP(topic_synonyms_tuples, set())
    for np in topic_object.span_lst:
        np_object = span_to_object.get(np, None)
        if np_object:
            is_combined = combine_topic_object(np_object, topic_object)
            if is_combined:
                topic_object = np_object
            else:
                remove_topic_np_from_np_object(np_object, np)
    label_lst = set()
    for np in topic_object.span_lst:
        span_to_object[np] = topic_object
        label = longest_NP_to_global_index.get(np, None)
        if label:
            label_lst.add(label)
    topic_object.label_lst = label_lst
    return topic_object

def insert_examples_of_topic_to_DAG(dict_score_to_collection_of_sub_groups, topic_synonym_lst, dict_span_to_lemmas_lst,
                                    all_object_np_lst, span_to_object, dict_span_to_similar_spans,
                                    dict_label_to_spans_group, global_dict_label_to_object, topic_object_lst,
                                    longest_np_total_lst, longest_np_lst, longest_NP_to_global_index, dict_object_to_global_label):
    topic_label_lst = set()
    for longest_np in longest_np_total_lst:
        topic_label_lst.add(longest_NP_to_global_index[longest_np])
    # longest_spans_calculated_in_previous_topics = set(longest_np_total_lst) - set(longest_np_lst)
    # add_dependency_routh_between_longest_np_to_topic(span_to_object, topic_object_lst,
    #                                                  longest_spans_calculated_in_previous_topics, topic_object)
    create_and_insert_nodes_from_sub_groups_of_spans(dict_score_to_collection_of_sub_groups,
                                                     dict_span_to_lemmas_lst, all_object_np_lst,
                                                     span_to_object, dict_span_to_similar_spans,
                                                     dict_label_to_spans_group,
                                                     global_dict_label_to_object, dict_object_to_global_label)
    topic_object = create_and_update_topic_object(topic_synonym_lst, span_to_object, longest_NP_to_global_index)
    add_dependency_routh_between_longest_np_to_topic(span_to_object, topic_object_lst,
                                                     longest_np_total_lst, topic_object)


def change_DAG_direction(global_np_object_lst, visited=[]):
    for np_object in global_np_object_lst:
        if np_object in visited:
            continue
        visited.append(np_object)
        child_np_object_remove_lst = []
        for child_np_object in np_object.children:
            child_np_object_remove_lst.append(child_np_object)
        if np_object.children:
            change_DAG_direction(np_object.children, visited)
            for child_np_object_to_remove in child_np_object_remove_lst:
                for span_as_lemmas in np_object.list_of_span_as_lemmas_lst:
                    if len(span_as_lemmas) == 1:
                        print(span_as_lemmas)
                        break
                if np_object not in child_np_object_to_remove:
                    child_np_object_to_remove.add_children([np_object])
                    np_object.children.remove(child_np_object_to_remove)


def add_descendants_of_node_to_graph(node, global_index_to_similar_longest_np):
    span_to_present = ""
    first_val = True
    node.span_lst = list(node.span_lst)
    # node.np_val = sorted(node.np_val, key=lambda np_val: combine_spans_utils.dict_of_span_to_counter[np_val], reverse=True)
    node.span_lst.sort(key=lambda span_lst: combine_spans_utils.dict_span_to_counter.get(span_lst, 0), reverse=True)
    for span in node.span_lst:
        if not first_val:
            span_to_present += " | "
        first_val = False
        span_to_present += span
    label_lst = get_labels_of_children(node.children)
    label_lst = node.label_lst - label_lst
    NP_occurrences = get_frequency_from_labels_lst(global_index_to_similar_longest_np,
                                                   label_lst)
    span_to_present += " NP " + str(NP_occurrences) + " covered by NP " + str(
        get_frequency_from_labels_lst(global_index_to_similar_longest_np,
                                      node.label_lst))
    np_val_dict = {span_to_present: {}}
    node.children = sorted(node.children, key=lambda child: get_frequency_from_labels_lst(
        global_index_to_similar_longest_np,
        child.label_lst), reverse=True)
    for child in node.children:
        np_val_dict[span_to_present].update(add_descendants_of_node_to_graph(child, global_index_to_similar_longest_np))
    return np_val_dict


def from_DAG_to_JSON(topic_object_lst, global_index_to_similar_longest_np):
    np_val_lst = {}
    topic_object_lst.sort(key=lambda topic_object: topic_object.marginal_val, reverse=True)
    for topic_node in topic_object_lst:
        np_val_lst.update(add_descendants_of_node_to_graph(topic_node, global_index_to_similar_longest_np))
    return np_val_lst


def add_dependency_routh_between_longest_np_to_topic(span_to_object, topic_object_lst,
                                                     longest_nps, topic_object):
    visited_and_added = set()
    visited = set()
    for longest_np_span in longest_nps:
        np_object = span_to_object[longest_np_span]
        if np_object in visited:
            continue
        if np_object in topic_object_lst:
            # np_object.combine_nodes(topic_object)
            # topic_object_lst.remove(topic_object)
            # topic_object = np_object
            continue
        similar_np_object = [None]
        add_NP_to_DAG_bottom_to_up(topic_object, np_object, visited, similar_np_object, visited_and_added)
        if similar_np_object[0]:
            # raise Exception("There are 2 synonyms topics which aren't detected")
            if similar_np_object[0] in topic_object_lst:
                if topic_object in topic_object_lst:
                    topic_object_lst.remove(topic_object)
            topic_object = similar_np_object[0]
            for span in similar_np_object[0].span_lst:
                span_to_object[span] = similar_np_object[0]
    for np in topic_object.span_lst:
        span_to_object[np] = topic_object
    if topic_object not in topic_object_lst:
        topic_object_lst.append(topic_object)


def update_score(topic_object_lst, dict_span_to_rank, visited=[]):
    for node in topic_object_lst:
        if node in visited:
            continue
        visited.append(node)
        node.score = combine_spans_utils.get_average_value(node.span_lst, dict_span_to_rank)
        update_score(node.children, dict_span_to_rank, visited)


def check_symmetric_relation_in_DAG(nodes_lst, visited=set()):
    for node in nodes_lst:
        if node in visited:
            continue
        if node in node.parents:
            raise Exception("node can't be itself parent")
        if node in node.children:
            raise Exception("node can't be itself child")
        visited.add(node)
        for child in node.children:
            if node not in child.parents:
                raise Exception("Parent and child isn't consistent")
        for parent in node.parents:
            if node not in parent.children:
                print(parent.span_lst)
                print(node.span_lst)
                raise Exception("Parent and child isn't consistent")
        check_symmetric_relation_in_DAG(node.children, visited)


def get_frequency_from_labels_lst(global_index_to_similar_longest_np, label_lst):
    num_of_labels = 0
    for label in label_lst:
        for span in global_index_to_similar_longest_np[label]:
            num_of_labels += combine_spans_utils.dict_longest_span_to_counter[span]
        # num_of_labels += len(global_index_to_similar_longest_np[label])
    return num_of_labels


def update_nodes_frequency(topic_object_lst, global_index_to_similar_longest_np, visited=[]):
    for node in topic_object_lst:
        if node in visited:
            continue
        is_first = True
        for np in node.span_lst:
            encoded_input = tokenizer(np, return_tensors='pt')
            if is_first:
                weighted_average_vector = medical_model(**encoded_input).last_hidden_state[0, 0, :]
                is_first = False
            else:
                weighted_average_vector += medical_model(**encoded_input).last_hidden_state[0, 0, :]
        weighted_average_vector /= len(node.span_lst)
        node.weighted_average_vector = weighted_average_vector
        node.frequency = get_frequency_from_labels_lst(global_index_to_similar_longest_np,
                                                       node.label_lst)
        visited.append(node)
        update_nodes_frequency(node.children, global_index_to_similar_longest_np, visited)


def get_labels_of_children(children):
    label_lst = set()
    for child in children:
        label_lst.update(child.label_lst)
    return label_lst


def get_labels_of_leaves(children):
    label_lst = set()
    for child in children:
        if not child.children:
            label_lst.update(child.label_lst)
    return label_lst


counter_error_node = 0


def get_leaves_from_DAG(nodes_lst, leaves_lst=set(), visited=set()):  # function for dfs
    global counter_error_node
    counter_error_node += 1
    for node in nodes_lst:
        if node not in visited:
            if len(node.children) == 0:
                leaves_lst.add(node)
            visited.add(node)
        get_leaves_from_DAG(node.children, leaves_lst, visited)
    return leaves_lst


def remove_redundant_nodes(nodes_lst):
    leaves_lst = get_leaves_from_DAG(nodes_lst)
    queue = []
    visited = []
    visited.extend(leaves_lst)
    queue.extend(leaves_lst)
    counter = 0
    ignore_lst = []
    while queue:
        counter += 1
        check_symmetric_relation_in_DAG(nodes_lst, set())
        s = queue.pop(0)
        parents = s.parents.copy()
        for parent in parents:
            if parent in ignore_lst:
                continue
            if len(parent.children) == 1 and parent.label_lst == s.label_lst:
                # labels_children = get_labels_of_children(parent.children)
                # parent.label_lst - labels_children
                if parent.parents:
                    for ancestor in parent.parents:
                        ancestor.add_children([s])
                        if parent in ancestor.children:
                            ancestor.children.remove(parent)
                        else:
                            raise Exception("Parent and child isn't consistent", parent.span_lst, s.span_lst)
                        s.parents.add(ancestor)
                    ignore_lst.append(parent)
                    if parent in s.parents:
                        s.parents.remove(parent)
                    else:
                        raise Exception("unclear error")
                    if s in parent.children:
                        parent.children.remove(s)
                    else:
                        raise Exception("Parent and child isn't consistent", s.span_lst, parent.span_lst)
                else:
                    ignore_lst.append(parent)
                    s.parents.remove(parent)
                    if parent in nodes_lst:
                        nodes_lst.remove(parent)
                    else:
                        continue
                    if not s.parents:
                        nodes_lst.append(s)
                        break
            else:
                if parent not in visited:
                    visited.append(parent)
                    queue.append(parent)
