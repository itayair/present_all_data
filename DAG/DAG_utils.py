from combine_spans import utils as combine_spans_utils
from transformers import AutoTokenizer, AutoModel
import DAG.NounPhraseObject as NounPhrase

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
medical_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')


def add_NP_to_DAG_up_to_bottom(np_object_to_add, np_object, similar_np_object):
    is_contained = False
    if np_object == np_object_to_add:
        return True
    for np in np_object_to_add.np:
        for np_ref in np_object.np:
            if combine_spans_utils.is_similar_meaning_between_span(np_ref, np):
                is_contained = True
                if len(np_ref) == len(np):
                    np_object.combine_nodes(np_object_to_add)
                    similar_np_object[0] = np_object
                    return True
    if is_contained:
        is_added = False
        for child in np_object.children:
            is_added |= add_NP_to_DAG_up_to_bottom(np_object_to_add, child, similar_np_object)
            if similar_np_object[0]:
                return True
        if not is_added:
            if np_object_to_add not in np_object.children:
                np_object.add_children([np_object_to_add])
            np_object_to_add.parents.add(np_object)
        return True
    return False


def add_NP_to_DAG_bottom_to_up(np_object_to_add, np_object, visited, similar_np_object):
    is_contained = False
    if np_object in visited:
        return False
    if np_object == np_object_to_add:
        return True
    visited.add(np_object)
    for np in np_object_to_add.np:
        for np_ref in np_object.np:
            if combine_spans_utils.is_similar_meaning_between_span(np, np_ref):
                is_contained = True
                if len(np_ref) == len(np):
                    np_object.combine_nodes(np_object_to_add)
                    similar_np_object[0] = np_object
                    return True
    if is_contained:
        is_added = False
        for parent in np_object.parents:
            is_added |= add_NP_to_DAG_bottom_to_up(np_object_to_add, parent, visited, similar_np_object)
            if similar_np_object[0]:
                return True
        if not is_added:
            if np_object not in np_object_to_add.children:
                np_object_to_add.add_children([np_object])
            np_object.parents.add(np_object_to_add)
        return True
    return False


def create_DAG_from_top_to_bottom(dict_score_to_collection_of_sub_groups, topic_synonym_lst,
                                  dict_span_to_lemmas_lst,
                                  all_object_np_lst, span_to_object,
                                  dict_span_to_similar_spans, dict_label_to_spans_group,
                                  global_dict_label_to_object, topic_object_lst):
    topic_synonyms_tuples = [(synonym, [synonym]) for synonym in topic_synonym_lst]
    topic_object = NounPhrase.NP(topic_synonyms_tuples, set(dict_label_to_spans_group.keys()))
    topic_object_lst.append(topic_object)
    for score, np_to_labels_collection in dict_score_to_collection_of_sub_groups.items():
        for np_key, labels in np_to_labels_collection:
            np_collection = dict_span_to_similar_spans[np_key]
            for np in np_collection:
                np_object = span_to_object.get(np, None)
                if np_object:
                    break
            if np_object:
                continue
            tuple_np_lst = []
            for np in np_collection:
                span_to_object[np] = np_object
                tuple_np_lst.append((np, dict_span_to_lemmas_lst[np]))
            np_object = NounPhrase.NP(tuple_np_lst, labels)
            similar_np_object = [None]
            add_NP_to_DAG_up_to_bottom(np_object, topic_object, similar_np_object)
            if not similar_np_object[0]:
                all_object_np_lst.append(np_object)
    for label, nps in dict_label_to_spans_group.items():
        for np in nps:
            np_object = span_to_object.get(np[0], None)
            if np_object:
                break
        if np_object:
            global_dict_label_to_object[label] = np_object
            continue
        has_single_token = False
        np_object = NounPhrase.NP(nps, [label])
        for np in nps:
            if len(np[1]) == 1:
                has_single_token = True
                break
        if has_single_token:
            topic_object.combine_nodes(np_object)
            global_dict_label_to_object[label] = topic_object
            for np in nps:
                span_to_object[np[0]] = topic_object
            continue
        similar_np_object = [None]
        add_NP_to_DAG_up_to_bottom(np_object, topic_object, similar_np_object)
        if similar_np_object[0]:
            np_object = similar_np_object[0]
        else:
            all_object_np_lst.append(np_object)
        global_dict_label_to_object[label] = np_object
        for np in np_object.np_val:
            span_to_object[np] = np_object
    return topic_object


def insert_examples_of_topic_to_DAG(dict_score_to_collection_of_sub_groups, topic_synonym_lst, dict_span_to_lst,
                                    all_object_np_lst, span_to_object, dict_span_to_similar_spans,
                                    dict_label_to_spans_group, global_dict_label_to_object, topic_object_lst,
                                    longest_np_total_lst, longest_np_lst):
    topic_object = create_DAG_from_top_to_bottom(dict_score_to_collection_of_sub_groups, topic_synonym_lst,
                                                 dict_span_to_lst, all_object_np_lst,
                                                 span_to_object, dict_span_to_similar_spans,
                                                 dict_label_to_spans_group,
                                                 global_dict_label_to_object, topic_object_lst)
    longest_spans_calculated_in_previous_topics = set(longest_np_total_lst) - set(longest_np_lst)
    add_dependency_routh_between_longest_np_to_topic(span_to_object, topic_object_lst,
                                                     longest_spans_calculated_in_previous_topics, topic_object)


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
                for np in np_object.np:
                    if len(np) == 1:
                        print(np)
                        break
                if np_object not in child_np_object_to_remove:
                    child_np_object_to_remove.add_children([np_object])
                    np_object.children.remove(child_np_object_to_remove)


def add_descendants_of_node_to_graph(node, global_index_to_similar_longest_np):
    span_to_present = ""
    first_val = True
    for np_val in node.np_val:
        if not first_val:
            span_to_present += " | "
        first_val = False
        span_to_present += np_val
    label_lst = get_labels_of_children(node.children)
    label_lst = node.label_lst - label_lst
    NP_occurrences = get_frequency_from_labels_lst(global_index_to_similar_longest_np,
                                                                       label_lst)
    span_to_present += " NP " + str(NP_occurrences) + " covered by NP " + str(get_frequency_from_labels_lst(global_index_to_similar_longest_np,
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
    for longest_np_span in longest_nps:
        np_object = span_to_object[longest_np_span]
        if np_object in topic_object_lst:
            np_object.combine_nodes(topic_object)
            topic_object_lst.remove(topic_object)
            topic_object = np_object
            continue
        similar_np_object = [None]
        add_NP_to_DAG_bottom_to_up(topic_object, np_object, set(), similar_np_object)
        if similar_np_object[0]:
            if similar_np_object[0] in topic_object_lst:
                topic_object_lst.remove(topic_object)
                topic_object = similar_np_object[0]
            for np in similar_np_object[0].np_val:
                span_to_object[np] = similar_np_object[0]


def update_score(topic_object_lst, dict_span_to_rank, visited=[]):
    for node in topic_object_lst:
        if node in visited:
            continue
        visited.append(node)
        node.score = combine_spans_utils.get_average_value(node.np_val, dict_span_to_rank)
        update_score(node.children, dict_span_to_rank, visited)


def check_symmetric_relation_in_DAG(nodes_lst, visited=set()):
    for node in nodes_lst:
        if node in visited:
            continue
        visited.add(node)
        for child in node.children:
            if node not in child.parents:
                raise Exception("Parent and child isn't consistent")
        for parent in node.parents:
            if node not in parent.children:
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
        for np in node.np_val:
            encoded_input = tokenizer(np, return_tensors='pt')
            if is_first:
                weighted_average_vector = medical_model(**encoded_input).last_hidden_state[0, 0, :]
                is_first = False
            else:
                weighted_average_vector += medical_model(**encoded_input).last_hidden_state[0, 0, :]
        weighted_average_vector /= len(node.np_val)
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


def get_leaves_from_DAG(nodes_lst, leaves_lst=set(), visited=set()):  # function for dfs
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
                            raise Exception("Parent and child isn't consistent", parent.np_val, s.np_val)
                        s.parents.add(ancestor)
                    ignore_lst.append(parent)
                    if parent in s.parents:
                        s.parents.remove(parent)
                    else:
                        raise Exception("unclear error")
                    if s in parent.children:
                        parent.children.remove(s)
                    else:
                        raise Exception("Parent and child isn't consistent", s.np_val, parent.np_val)
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
