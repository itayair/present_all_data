from combine_spans import utils as combine_spans_utils
from transformers import AutoTokenizer, AutoModel
import torch
import DAG.NounPhraseObject as NounPhrase
import gc

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
medical_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
medical_model = medical_model.to(device)
medical_model = medical_model.eval()


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
    if combine_spans_utils.is_similar_meaning_between_span(node.common_lemmas_in_spans,
                                                           ref_node.common_lemmas_in_spans):
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


def add_NP_to_DAG_bottom_to_up(np_object_to_add, np_object, visited, similar_np_object, visited_and_added=set()):
    is_contained = False
    if np_object == np_object_to_add:
        return True
    if np_object in visited_and_added:
        return True
    if np_object in visited:
        return False
    visited.add(np_object)
    if not np_object_to_add.common_lemmas_in_spans:
        print("Empty node")
        print(np_object_to_add.span_lst)
        print(np_object_to_add.lemma_to_occurrences_dict)
        return False
    # for span_as_lemmas in np_object_to_add.list_of_span_as_lemmas_lst:
    #     for span_as_lemmas_ref in np_object.list_of_span_as_lemmas_lst:
    if combine_spans_utils.is_similar_meaning_between_span(np_object_to_add.common_lemmas_in_spans,
                                                           np_object.common_lemmas_in_spans):
        is_contained = True
        if len(np_object_to_add.common_lemmas_in_spans) == len(np_object.common_lemmas_in_spans):
            np_object.combine_nodes(np_object_to_add)
            similar_np_object[0] = np_object
            return True
    if is_contained:
        is_added = False
        for parent in np_object.parents:
            is_added |= add_NP_to_DAG_bottom_to_up(np_object_to_add, parent, visited, similar_np_object,
                                                   visited_and_added)
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
                if np_object in parent_object.children:
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
    collection_np_object.label_lst.update(labels)
    nps_to_update = set()
    for np in np_collection:
        np_object = span_to_object.get(np, None)
        if np_object == collection_np_object:
            continue
        if np_object:
            np_object.combine_nodes(collection_np_object)
            for span in np_object.span_lst:
                span_to_object[span] = np_object
            collection_np_object = np_object
        nps_to_update.add(np)
    for np in nps_to_update:
        span_to_object[np] = collection_np_object
        collection_np_object.span_lst.add(np)
        collection_np_object.add_unique_lst([dict_span_to_lemmas_lst[np]])


#
# def get_longest_np_nodes_contain_labels(label_lst, global_dict_label_to_object, np_object, dict_object_to_global_label,
#                                         longest_nps_total_lst):
#     nodes_lst = set()
#     uncounted_labels = set()
#     for label in label_lst:
#         node = global_dict_label_to_object.get(label, None)
#         if node:
#             nodes_lst.add(node)
#         else:
#             uncounted_labels.add(label)
#     for label in uncounted_labels:
#         global_dict_label_to_object[label] = np_object
#     dict_object_to_global_label[hash(np_object)] = uncounted_labels
#     return nodes_lst


def update_global_label_with_its_object(global_dict_label_to_object, np_object, dict_object_to_global_label,
                                        longest_np_lst, longest_NP_to_global_index):
    uncounted_labels = set()
    for label in np_object.label_lst:
        node = global_dict_label_to_object.get(label, None)
        if node:
            continue
        uncounted_labels.add(label)
    global_indices_in_np_object = set()
    for span in np_object.span_lst:
        if span in longest_np_lst:
            label = longest_NP_to_global_index.get(span, -1)
            if label == -1:
                print("not consistant global label with longest np")
                continue
            global_indices_in_np_object.add(label)
    uncounted_global_labels = uncounted_labels.intersection(global_indices_in_np_object)
    for label in uncounted_global_labels:
        global_dict_label_to_object[label] = np_object

    dict_object_to_global_label[hash(np_object)] = uncounted_global_labels


def create_and_insert_nodes_from_sub_groups_of_spans(dict_score_to_collection_of_sub_groups,
                                                     dict_span_to_lemmas_lst,
                                                     all_object_np_lst, span_to_object,
                                                     dict_span_to_similar_spans, global_dict_label_to_object,
                                                     dict_object_to_global_label, longest_np_lst,
                                                     objects_longest_nps_from_previous_clusters,
                                                     longest_NP_to_global_index):
    longest_nps_total_lst = objects_longest_nps_from_previous_clusters.copy()
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
        update_global_label_with_its_object(global_dict_label_to_object, np_object, dict_object_to_global_label,
                                            longest_np_lst, longest_NP_to_global_index)
        # longest_nps_nodes_lst = get_longest_np_nodes_contain_labels(np_object.label_lst.intersection(topic_label_lst),
        #                                                             global_dict_label_to_object,
        #                                                             np_object, dict_object_to_global_label)
        # if not longest_nps_nodes_lst:
        #     continue
        similar_np_object = [None]
        visited = set()
        visited_and_added = set()
        is_combined_with_exist_node = False
        np_object_temp = np_object
        for longest_nps_node in longest_nps_total_lst:
            if longest_nps_node in visited:
                continue
            add_NP_to_DAG_bottom_to_up(np_object_temp, longest_nps_node, visited, similar_np_object, visited_and_added)
            if similar_np_object[0]:
                if hash(np_object) in dict_object_to_global_label:
                    for label in dict_object_to_global_label[hash(np_object)]:
                        global_dict_label_to_object[label] = similar_np_object[0]
                    dict_object_to_global_label[hash(similar_np_object[0])] = \
                        dict_object_to_global_label.get(hash(similar_np_object[0]), set())
                    dict_object_to_global_label[hash(similar_np_object[0])].update(
                        dict_object_to_global_label[hash(np_object)])
                    dict_object_to_global_label.pop(hash(np_object), None)
                is_combined_with_exist_node = True
                for span in np_object.span_lst:
                    span_to_object[span] = similar_np_object[0]
                np_object_temp = similar_np_object[0]
                if np_object_temp:
                    break
                similar_np_object[0] = None
        if not is_combined_with_exist_node:
            all_object_np_lst.append(np_object)
        if np_object_temp not in longest_nps_total_lst and np_object_temp.span_lst.intersection(longest_np_lst):
            longest_nps_total_lst.add(np_object)


def remove_topic_np_from_np_object(np_object, topic_np):
    lemmas_to_remove = None
    for span_as_lemmas_lst in np_object.list_of_span_as_lemmas_lst:
        if len(span_as_lemmas_lst) == 1 and topic_np in span_as_lemmas_lst:
            lemmas_to_remove = span_as_lemmas_lst
            break
    if lemmas_to_remove:
        np_object.list_of_span_as_lemmas_lst.remove(lemmas_to_remove)


def combine_topic_object(np_object, topic_object):
    if combine_spans_utils.is_similar_meaning_between_span(topic_object.common_lemmas_in_spans,
                                                           np_object.common_lemmas_in_spans):
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
            if np_object == topic_object:
                continue
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
                                    global_dict_label_to_object, topic_object_lst,
                                    longest_np_total_lst, longest_np_lst, longest_NP_to_global_index,
                                    dict_object_to_global_label):
    longest_nps_from_previous_clusters = set(longest_np_total_lst) - set(longest_np_lst)
    objects_longest_nps_from_previous_clusters = set()
    for longest_np in longest_nps_from_previous_clusters:
        objects_longest_nps_from_previous_clusters.add(span_to_object[longest_np])
    create_and_insert_nodes_from_sub_groups_of_spans(dict_score_to_collection_of_sub_groups,
                                                     dict_span_to_lemmas_lst, all_object_np_lst,
                                                     span_to_object, dict_span_to_similar_spans,
                                                     global_dict_label_to_object, dict_object_to_global_label,
                                                     longest_np_lst, objects_longest_nps_from_previous_clusters,
                                                     longest_NP_to_global_index)
    topic_object = create_and_update_topic_object(topic_synonym_lst, span_to_object, longest_NP_to_global_index)
    add_dependency_routh_between_longest_np_to_topic(span_to_object, topic_object_lst,
                                                     longest_np_total_lst, topic_object)


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
            print(node.span_lst)
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


def update_symmetric_relation_in_DAG(nodes_lst, visited=set()):
    for node in nodes_lst:
        if node in visited:
            continue
        if node in node.parents:
            node.parents.remove(node)
        if node in node.children:
            node.children.remove(node)
        visited.add(node)
        for child in node.children:
            if node not in child.parents:
                child.parents.add(node)
        for parent in node.parents:
            if node not in parent.children:
                parent.children.append(node)
        update_symmetric_relation_in_DAG(node.children, visited)


def get_frequency_from_labels_lst(global_index_to_similar_longest_np, label_lst):
    num_of_labels = 0
    for label in label_lst:
        for span in global_index_to_similar_longest_np[label]:
            num_of_labels += combine_spans_utils.dict_longest_span_to_counter[span]
        # num_of_labels += len(global_index_to_similar_longest_np[label])
    return num_of_labels


def initialize_node_weighted_vector(node):
    # is_first = True
    with torch.no_grad():
        most_frequent_span = combine_spans_utils.get_most_frequent_span(node.span_lst)
        encoded_input = tokenizer(most_frequent_span, return_tensors='pt').to(device)
        temp = medical_model(**encoded_input)
        weighted_average_vector = temp.last_hidden_state[0, 0, :].cpu()
    #     for np in node.span_lst:
    #         encoded_input = tokenizer(np, return_tensors='pt').to(device)
    #         allocation_counter += 1
    #         if is_first:
    #             temp = medical_model(**encoded_input)
    #             weighted_average_vector = temp.last_hidden_state[0, 0, :].cpu()
    #             is_first = False
    #         else:
    #             temp = medical_model(**encoded_input)
    #             weighted_average_vector += temp.last_hidden_state[0, 0, :].cpu()
    #         del temp, encoded_input
    #         torch.cuda.empty_cache()
    # weighted_average_vector /= len(node.span_lst)
    node.weighted_average_vector = weighted_average_vector


def get_represented_vector(span):
    encoded_input = tokenizer(span, return_tensors='pt')
    represented_vector = medical_model(**encoded_input).last_hidden_state[0, 0, :]
    return represented_vector


def initialize_nodes_weighted_average_vector(topic_object_lst, global_index_to_similar_longest_np, visited=[]):
    for node in topic_object_lst:
        if node in visited:
            continue
        initialize_node_weighted_vector(node)
        node.frequency = get_frequency_from_labels_lst(global_index_to_similar_longest_np,
                                                       node.label_lst)
        visited.append(node)
        initialize_nodes_weighted_average_vector(node.children, global_index_to_similar_longest_np, visited)


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
                    if parent in nodes_lst:
                        nodes_lst.remove(parent)
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
                    parent.children.remove(s)
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
