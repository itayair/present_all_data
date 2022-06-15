import valid_expansion_utils

tied_deps = ['neg', 'nmod:poss', 'compound', 'mwe', 'case', 'mark', 'auxpass', 'name', 'aux']
tied_couples = [['auxpass', 'nsubjpass']]

dep_type_optional = ['advmod', 'dobj', 'npadvmod', 'nmod', 'nummod', 'conj', 'aux', 'poss', 'nmod:poss',
                     'xcomp']  # , 'conj', 'nsubj', 'appos'

acl_to_seq = ['acomp', 'dobj', 'nmod']  # acl and relcl + [[['xcomp'], ['aux']], 'dobj']
others_to_seq = ['quantmod', 'cop']  # 'cc',
combined_with = ['acl', 'relcl', 'acl:relcl', 'ccomp', 'advcl', 'amod']  # +, 'cc'
couple_to_seq = {'quantmod': ['amod'], 'cop': ['nsubjpass']}  # 'nsubjpass': ['amod'] ,'cc': ['conj', 'nmod']


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




def combine_tied_deps_recursively_and_combine_their_children(head, boundary_np_to_the_left):
    combined_children_lst = []
    combined_tied_tokens = [head]
    tied_couples_to_add = get_tied_couples(head.children)
    for child in head.children:
        if child.dep_ in tied_deps or child in tied_couples_to_add:
            # if child.dep_ == 'case' and child.i == boundary_np_to_the_left:
            #     continue
            temp_tokens, temp_children = combine_tied_deps_recursively_and_combine_their_children(child,
                                                                                                  boundary_np_to_the_left)
            combined_tied_tokens.extend(temp_tokens)
            combined_children_lst.extend(temp_children)
        else:
            combined_children_lst.append(child)
    return combined_tied_tokens, combined_children_lst


def get_if_has_preposition_child_between(token, head):
    prep_lst = []
    for child in token.children:
        if child.dep_ == 'case' and child.tag_ == 'IN':
            prep_lst.append(child)
    if prep_lst:
        first_prep_child_index = 100
        prep_child = None
        for child in prep_lst:
            if child.i < first_prep_child_index:
                prep_child = child
        if token.i > head.i:
            first_val = head.i
            second_val = token.i
        else:
            first_val = token.i
            second_val = head.i
        if first_val < prep_child.i < second_val:
            return prep_child
        else:
            return None
    return None


def initialize_couple_lst(others, couple_lst, lst_children):
    for other in others:
        dep_type = couple_to_seq[other.dep_]
        for token in lst_children:
            if token.dep_ in dep_type:
                couple_lst.append([other, token])


def remove_conj_if_cc_exist(lst_children):
    cc_is_exist = False
    cc_child_lst = []
    for child in lst_children:
        if child.dep_ == 'cc' or child.dep_ == 'punct' and child.text == ',':
            cc_child_lst.append(child)
            cc_is_exist = True
    if cc_is_exist:
        children_dep = valid_expansion_utils.get_token_by_dep(lst_children, 'conj')
        if children_dep is []:
            children_dep = valid_expansion_utils.get_token_by_dep(lst_children, 'nmod')
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


# def get_det_token_from_children(lst_children):
#     for child in lst_children:
#         if child.dep_ == 'det':
#             lst_children.remove(child)
#             return child
#     return None


def set_couple_deps(couple_lst, boundary_np_to_the_left, sub_np_lst, head):
    for couple in couple_lst:
        sub_np_lst_couple, lst_children_first = combine_tied_deps_recursively_and_combine_their_children(couple[0],
                                                                                                         boundary_np_to_the_left)
        sub_np_lst_couple_second, lst_children_second = combine_tied_deps_recursively_and_combine_their_children(
            couple[1], boundary_np_to_the_left)
        sub_np_lst_couple.extend(sub_np_lst_couple_second)
        all_sub_of_sub = []
        get_children_expansion(all_sub_of_sub, lst_children_first, boundary_np_to_the_left, head)
        get_children_expansion(all_sub_of_sub, lst_children_second, boundary_np_to_the_left, head)
        if all_sub_of_sub:
            sub_np_lst_couple.append(all_sub_of_sub)
        sub_np_lst.append(sub_np_lst_couple)


dep_type_in_sequencial = set()


def get_all_valid_sub_special(token, boundary_np_to_the_left):
    sub_np_lst, lst_children = combine_tied_deps_recursively_and_combine_their_children(token, boundary_np_to_the_left)
    sub_np = []
    complete_children = []
    lst_to_skip, tokens_to_add = remove_conj_if_cc_exist(lst_children)
    for child in lst_children:
        # if child.i >= next_catch_word_index:
        #     continue
        # if child.dep_ in special_to_seq:
        if child in lst_to_skip:
            continue
        if child.dep_ in ['dobj', 'advcl', 'nmod']:  # 'cc', 'conj', 'aux', 'auxpass', 'cop', 'nsubjpass'
            dep_type_in_sequencial.add(child.dep_)
            all_sub_of_sub = get_all_valid_sub_np(child, boundary_np_to_the_left)
            all_sub_of_sub = sub_np_lst + all_sub_of_sub
            sub_np.append(all_sub_of_sub)
        else:
            complete_children.append(child)
    if sub_np == [] and token.dep_ == 'amod':
        sub_np.append(sub_np_lst)
    couple_lst = []
    couple_lst.extend(tokens_to_add)
    sub_np_lst_couples = []
    set_couple_deps(couple_lst, boundary_np_to_the_left, sub_np_lst_couples, [])
    if sub_np_lst_couples:
        for sub_sub_np_lst in sub_np:
            sub_sub_np_lst.append(sub_np_lst_couples)
    for child in complete_children:
        all_sub_of_sub = get_all_valid_sub_np(child, boundary_np_to_the_left)
        for sub_sub_np_lst in sub_np:
            sub_sub_np_lst.append(all_sub_of_sub)
    if not sub_np_lst:
        return []
    # sub_np_lst = [token] + sub_np_lst
    return sub_np


def get_children_expansion(sub_np_lst, lst_children, boundary_np_to_the_left, head):
    others = []
    lst_to_skip, tokens_to_add = remove_conj_if_cc_exist(lst_children)
    for child in lst_children:
        if child in lst_to_skip:
            continue
        # if child.i >= next_catch_word_index:
        #     continue
        sub_np = []
        if child.dep_ in dep_type_optional:
            # prep_child = get_if_has_preposition_child_between(child, head)
            all_sub_of_sub = get_all_valid_sub_np(child, boundary_np_to_the_left)
            sub_np.append(all_sub_of_sub)
            if sub_np:
                sub_np_lst.extend(sub_np)
        elif child.dep_ in combined_with:
            # if child.dep_ == 'acl' or child.dep_ == 'relcl':
            all_sub_of_sub = get_all_valid_sub_special(child, boundary_np_to_the_left)
            # elif child.dep_ == 'poss' or child.dep_ == 'nmod:poss':
            #     all_sub_of_sub = get_all_valid_sub_special(child, poss_to_seq, next_catch_word_index)
            #     if all_sub_of_sub == []:
            #         all_sub_of_sub = get_all_valid_sub_np(child, next_catch_word_index, None)
            if all_sub_of_sub:
                sub_np.append(all_sub_of_sub)
            if sub_np:
                sub_np_lst.extend(sub_np)
        elif child.dep_ in others_to_seq:
            others.append(child)
        else:
            if child.dep_ not in ['nsubj']:
                print(child.dep_)
    couple_lst = []
    if others:
        initialize_couple_lst(others, couple_lst, lst_children)
    couple_lst.extend(tokens_to_add)
    set_couple_deps(couple_lst, boundary_np_to_the_left, sub_np_lst, head)
    # return others


def get_all_valid_sub_np(head, boundary_np_to_the_left):
    sub_np_lst, lst_children = combine_tied_deps_recursively_and_combine_their_children(head, boundary_np_to_the_left)
    # if prep_child:
    #     sub_np_lst.append(prep_child)
    #     lst_children.remove(prep_child)

    get_children_expansion(sub_np_lst, lst_children, boundary_np_to_the_left, head)
    return sub_np_lst
