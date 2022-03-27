import utils as ut

tied_deps = ['det', 'neg', 'nmod:poss', 'compound', 'mwe', 'case', 'mark', 'auxpass', 'name', 'aux']
tied_couples = [['auxpass', 'nsubjpass']]

dep_type_optional = ['advmod', 'dobj', 'npadvmod', 'nmod', 'nummod', 'conj', 'poss', 'nmod:poss',
                     'xcomp']  # , 'conj', 'nsubj', 'appos'

acl_to_seq = ['acomp', 'dobj', 'nmod']  # acl and relcl + [[['xcomp'], ['aux']], 'dobj']
others_to_seq = ['quantmod', 'cop']  # 'cc',
combined_with = ['acl', 'relcl', 'acl:relcl', 'ccomp', 'advcl']  # +, 'cc'
couple_to_seq = {'quantmod': ['amod'], 'cop': ['nsubjpass', 'nsubj']}  # 'nsubjpass': ['amod'] ,'cc': ['conj', 'nmod']
pro_noun_tags_lst = ['WP', 'PRP', 'DET', 'NN', 'NNS']


def get_tied_couple_by_deps(first_dep, second_dep, children):
    first_token = None
    second_token = None
    for child in children:
        if child.dep_ == first_dep:
            first_token = child
        if child.dep_ == second_dep:
            second_token = child
    return first_token, second_token


def get_tied_couples(children):
    tied_couples_to_add = []
    for dep_couples in tied_couples:
        first_token, second_token = get_tied_couple_by_deps(dep_couples[0], dep_couples[1], children)
        if not first_token or not second_token:
            continue
        tied_couples_to_add.append(first_token)
        tied_couples_to_add.append(second_token)
    return tied_couples_to_add


def combine_tied_deps_recursively_and_combine_their_children(head, head_word_index=-1, head_token_type=1):
    combined_children_lst = []
    combined_tied_tokens = [(head, head_token_type)]
    tied_couples_to_add = get_tied_couples(head.children)
    for child in head.children:
        if head_word_index != -1:
            if head.dep_ == 'nmod':
                if child.dep_ in ['case', 'mark']:
                    continue
        if child.dep_ in tied_deps or child in tied_couples_to_add:
            if child.dep_ in ['case', 'mark']:
                temp_tokens, temp_children = combine_tied_deps_recursively_and_combine_their_children(child, -1, 2)
                temp_tokens = [(token_couple[0], 2) for token_couple in temp_tokens]
            else:
                temp_tokens, temp_children = combine_tied_deps_recursively_and_combine_their_children(child)
            combined_tied_tokens.extend(temp_tokens)
            combined_children_lst.extend(temp_children)
        else:
            combined_children_lst.append(child)
    return combined_tied_tokens, combined_children_lst


def initialize_couple_lst(others, couple_lst, lst_children):
    for other in others:
        dep_type = couple_to_seq[other.dep_]
        for token in lst_children:
            if token.dep_ in dep_type:
                if other.dep_ == 'cop':
                    if token.tag_ in pro_noun_tags_lst:
                        continue
                couple_lst.append([other, token])


def remove_conj_if_cc_exist(lst_children):
    cc_is_exist = False
    cc_child_lst = []
    for child in lst_children:
        if child.dep_ == 'cc' or (child.dep_ == 'punct' and child.text == ','):
            cc_child_lst.append(child)
            cc_is_exist = True
    if cc_is_exist:
        children_dep = ut.get_token_by_dep(lst_children, 'conj')
        if children_dep is []:
            children_dep = ut.get_token_by_dep(lst_children, 'nmod')
        children_dep.sort(key=lambda x: x.i)
        cc_child_lst.sort(key=lambda x: x.i)
        tokens_to_skip = cc_child_lst.copy()
        tokens_to_add = []
        for cc_child in cc_child_lst:
            for child in children_dep:
                if child.i > cc_child.i:
                    tokens_to_skip.append(child)
                    tokens_to_add.append([cc_child, child])
                    children_dep.remove(child)
                    break
        return tokens_to_skip, tokens_to_add
    return [], []


def set_couple_deps(couple_lst, sub_np_lst, head):
    for couple in couple_lst:
        sub_np_lst_couple, lst_children_first = combine_tied_deps_recursively_and_combine_their_children(couple[0], -1, 2)
        sub_np_lst_couple_second, lst_children_second = combine_tied_deps_recursively_and_combine_their_children(
            couple[1], -1, 5)
        sub_np_lst_couple.extend(sub_np_lst_couple_second)
        all_sub_of_sub = []
        get_children_expansion(all_sub_of_sub, lst_children_first, head)
        get_children_expansion(all_sub_of_sub, lst_children_second, head)
        if all_sub_of_sub:
            sub_np_lst_couple.append(all_sub_of_sub)
        sub_np_lst.append(sub_np_lst_couple)


dep_type_in_sequential = set()


def get_all_valid_sub_special(token):
    sub_np_lst, lst_children = combine_tied_deps_recursively_and_combine_their_children(token, -1, 2)
    sub_np = []
    complete_children = []
    lst_to_skip, tokens_to_add = remove_conj_if_cc_exist(lst_children)
    for child in lst_children:
        if child in lst_to_skip or child.text in ['-', '(', ')', '"']:
            continue
        if child.dep_ == 'poss':
            print(child.text)
        if child.dep_ in ['dobj', 'advcl', 'nmod']:
            dep_type_in_sequential.add(child.dep_)
            all_sub_of_sub = get_all_valid_sub_np(child)
            all_sub_of_sub = sub_np_lst + all_sub_of_sub
            sub_np.append(all_sub_of_sub)
        else:
            complete_children.append(child)
    couple_lst = []
    couple_lst.extend(tokens_to_add)
    sub_np_lst_couples = []
    set_couple_deps(couple_lst, sub_np_lst_couples, [])
    if sub_np_lst_couples:
        for sub_sub_np_lst in sub_np:
            sub_sub_np_lst.append(sub_np_lst_couples)
    for child in complete_children:
        all_sub_of_sub = get_all_valid_sub_np(child)
        for sub_sub_np_lst in sub_np:
            sub_sub_np_lst.append(all_sub_of_sub)
    if not sub_np_lst:
        return []
    return sub_np

def get_all_children(head, head_token_type = 4):
    combined_tied_tokens = [(head, head_token_type)]
    for child in head.children:
        temp_tokens = get_all_children(child, 2)
        combined_tied_tokens.extend(temp_tokens)
    return combined_tied_tokens

def get_children_expansion(sub_np_lst, lst_children, head):
    others = []
    lst_to_skip, tokens_to_add = remove_conj_if_cc_exist(lst_children)
    for child in lst_children:
        if child in lst_to_skip or child.text in ['-', '(', ')']:
            continue
        sub_np = []
        if child.dep_ in dep_type_optional:
            all_sub_of_sub = get_all_valid_sub_np(child)
            sub_np.append(all_sub_of_sub)
            if sub_np:
                sub_np_lst.extend(sub_np)
        elif child.dep_ in combined_with:
            all_sub_of_sub = get_all_valid_sub_special(child)
            if all_sub_of_sub:
                sub_np.append(all_sub_of_sub)
            if sub_np:
                sub_np_lst.extend(sub_np)
        elif child.dep_ in others_to_seq:
            others.append(child)
        elif child.dep_ == 'amod':
            all_sub_of_sub = get_all_children(child, 4)
            sub_np.append(all_sub_of_sub)
            if sub_np:
                sub_np_lst.extend(sub_np)
        # else:
        #     if child.dep_ not in ['nsubj']:
        #         print(child.dep_)
    couple_lst = []
    if others:
        initialize_couple_lst(others, couple_lst, lst_children)
    couple_lst.extend(tokens_to_add)
    set_couple_deps(couple_lst, sub_np_lst, head)


def get_all_valid_sub_np(head, head_word_index=-1):
    sub_np_lst, lst_children = combine_tied_deps_recursively_and_combine_their_children(head,
                                                                                        head_word_index, 5)
    get_children_expansion(sub_np_lst, lst_children, head)
    return sub_np_lst
