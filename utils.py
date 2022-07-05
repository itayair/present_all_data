import itertools
import sentence_representation as sent_rep

noun_tags_lst = ['NN', 'NNS', 'WP', 'PRP', 'NNP', 'NNPS']
pro_noun_tags_lst = ['WP', 'PRP', 'DET', 'NN', 'NNS']
noun_tags_lst_to_expand = ['NNP', 'NNPS']


def from_children_to_list(children):
    lst_children = []
    for token in children:
        lst_children.append(token)
    return lst_children


def get_head_of_span(span):
    head = None
    for child in span:
        if child.head == child:
            return child
        if child.head not in span:
            head = child
    return head


def get_token_by_dep(lst_children, dep_type):
    lst_tokens = []
    for child in lst_children:
        if child.dep_ == dep_type:
            lst_tokens.append(child)
    return lst_tokens


def powerset(iterable):
    # s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(iterable, r) for r in range(len(iterable) + 1))


# def get_all_combination(sub_np_of_child_lst, sub_np_of_child_lst_final):
#     result_list = list(powerset(sub_np_of_child_lst))
#     for item in result_list:
#         for element in itertools.product(*item):
#             if item == ():
#                 continue
#             lst_temp = []
#             for token in list(element):
#                 lst_temp.extend(token)
#             sub_np_of_child_lst_final.append(lst_temp)


def get_all_children_recursively(sub_np_lst, root):
    for child in sub_np_lst:
        from_lst_to_sequence(child, root)


def from_lst_to_sequence(sub_np_lst, root):
    if isinstance(sub_np_lst[0], list):
        if len(sub_np_lst) == 1:
            return from_lst_to_sequence(sub_np_lst[0], root)
        get_all_children_recursively(sub_np_lst, root)
    else:
        collect_to_lst = []
        slice_index = 0
        for item in sub_np_lst:
            if isinstance(item, list):
                break
            collect_to_lst.append(item)
            slice_index += 1
        node_in_sentence_representation = sent_rep.Node(collect_to_lst)
        if len(sub_np_lst) == 1:
            if root is None:
                root = node_in_sentence_representation
            else:
                root.add_children(node_in_sentence_representation)
            return root
        get_all_children_recursively(sub_np_lst[slice_index:], node_in_sentence_representation)
        if root is None:
            root = node_in_sentence_representation
        else:
            root.add_children(node_in_sentence_representation)
    return root


def is_np_child_head(head_word, word):
    if word.head == head_word:
        return True
    if word == word.head:
        return False
    return is_np_child_head(head_word, word.head)


def get_np_boundary(head_word_index, sentence_dep_graph):
    head_word = sentence_dep_graph[head_word_index]
    if head_word.tag_ not in noun_tags_lst:
        return None, None, None
    boundary_np_to_the_left = head_word_index
    # np's boundary to the left
    for i in range(1, head_word_index + 1):
        current_index = head_word_index - i
        word = sentence_dep_graph[current_index]
        if is_np_child_head(head_word, word):
            boundary_np_to_the_left = current_index
            continue
        break
    # np's boundary to the right
    boundary_np_to_the_right = head_word_index
    for i in range(head_word_index + 1, len(sentence_dep_graph)):
        word = sentence_dep_graph[i]
        if is_np_child_head(head_word, word):
            boundary_np_to_the_right = i
            continue
        break
    return sentence_dep_graph[
           boundary_np_to_the_left:boundary_np_to_the_right + 1], head_word_index - boundary_np_to_the_left, boundary_np_to_the_right - boundary_np_to_the_left


def get_tokens_as_span(tokens):
    span = ""
    idx = 0
    for token in tokens:
        if idx == 0 and token.tag_ in ['IN', 'TO']:
            continue
        if idx != 0 and token.text != ',':
            span += ' '
        span += token.text
        idx += 1
    return span


def get_tokens_as_span_special(tokens, tokens_with_special_format):
    span = ""
    idx = 0
    for token in tokens:
        if idx == 0 and token.tag_ in ['IN', 'TO']:
            continue
        if idx != 0 and token.text != ',':
            span += ' '
        if token in tokens_with_special_format:
            span += "EXPAND(" + token.text + ")"
        else:
            span += token.text
        idx += 1
    return span


def get_tokens_as_span_simple(tokens):
    span = ""
    idx = 0
    for token in tokens:
        if idx != 0 and token.text != ',':
            span += ' '
        span += token.text
        idx += 1
    return span


def write_to_file_dict_counter(sub_np_final_lst_collection, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        counter = 0
        for sub_np_final_lst in sub_np_final_lst_collection:
            lst_to_check_duplication = []
            for tokens in sub_np_final_lst:
                span = get_tokens_as_span(tokens)
                if span in lst_to_check_duplication:
                    print(span)
                    counter += 1
                else:
                    lst_to_check_duplication.append(span)
                f.write(span)
                f.write('\n')
            f.write('\n')
        print("Num of duplication:", counter)


def get_noun_in_sentence(sentence_dep_graph):
    noun_lst = []
    for token in sentence_dep_graph:
        if token.tag_ in noun_tags_lst and token.dep_ != 'compound':
            noun_lst.append(token)
    return noun_lst


################################################################
def list_of_nodes_to_span(list_of_nodes, head_noun):
    valid_span = []
    expand_format = []
    for node in list_of_nodes:
        valid_span.extend(node.basic_span_as_tokens)
        if head_noun not in node.basic_span_as_tokens:
            head_noun_token = None
            for token in node.basic_span_as_tokens:
                if token != head_noun and token.tag_ in noun_tags_lst_to_expand and (
                        node.children_to_the_left or node.children_to_the_right):
                    head_noun_token = token
            if head_noun_token:
                expand_format.append(head_noun_token)
    valid_span = set(valid_span)
    valid_span = list(valid_span)
    valid_span.sort(key=lambda x: x.i)
    return get_tokens_as_span_special(valid_span, expand_format)


def get_all_options(node, is_the_first_node=False):
    sub_tree = []
    for child in node.children_to_the_left:
        has_noun_token = False
        for token in child.span:
            if token.tag_ in noun_tags_lst:
                has_noun_token = True
        if has_noun_token:
            sub_tree.append([child])
        else:
            sub_tree_child = get_all_options(child)
            sub_tree.append(sub_tree_child)
    if is_the_first_node:
        for child in node.children_to_the_right:
            has_noun_token = False
            for token in child.span:
                if token.tag_ in noun_tags_lst:
                    has_noun_token = True
            if has_noun_token:
                sub_tree.append([child])
            else:
                sub_tree_child = get_all_options(child)
                sub_tree.append(sub_tree_child)
    sub_tree = [node] + sub_tree
    return sub_tree


def get_all_options_without_shortcut(node, is_the_first_node=False):
    sub_tree = []
    for child in node.children_to_the_left:
        sub_tree_child = get_all_options_without_shortcut(child)
        sub_tree.append(sub_tree_child)
    if not is_the_first_node:
        for child in node.children_to_the_right:
            sub_tree_child = get_all_options_without_shortcut(child)
            sub_tree.append(sub_tree_child)
    sub_tree = [node] + sub_tree
    return sub_tree


def list_of_nodes_to_span_without_shortcut(list_of_nodes):
    valid_span = []
    for node in list_of_nodes:
        valid_span.extend(node.basic_span_as_tokens)
    valid_span = set(valid_span)
    valid_span = list(valid_span)
    valid_span.sort(key=lambda x: x.i)
    return get_tokens_as_span_simple(valid_span)


def from_lst_to_sequence_special(sub_np_lst, current_lst):
    sub_np_of_child_lst_final = []
    if isinstance(sub_np_lst[0], list):
        if len(sub_np_lst) == 1:
            return from_lst_to_sequence_special(sub_np_lst[0], current_lst)
        sub_np_of_child_lst = []
        for child in sub_np_lst:
            new_lst_for_child = current_lst.copy()
            sub_np_of_child = from_lst_to_sequence_special(child, new_lst_for_child)
            sub_np_of_child_lst.append(sub_np_of_child)
    else:
        collect_to_lst = []
        slice_index = 0
        for item in sub_np_lst:
            if isinstance(item, list):
                break
            collect_to_lst.append(item)
            slice_index += 1
        current_lst.extend(collect_to_lst)
        sub_np_of_child_lst_final.append(current_lst)
        if len(sub_np_lst) == 1:
            return [current_lst]
        sub_np_of_child_lst = []
        for child in sub_np_lst[slice_index:]:
            new_lst_for_child = current_lst.copy()
            sub_np_of_child = from_lst_to_sequence_special(child, new_lst_for_child)
            sub_np_of_child_lst.append(sub_np_of_child)
    get_all_combination(sub_np_of_child_lst, sub_np_of_child_lst_final)
    return sub_np_of_child_lst_final
