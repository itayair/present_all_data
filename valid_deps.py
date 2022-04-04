import utils as ut

pro_noun_tags_lst = ['WP', 'PRP', 'DET', 'NN', 'NNS']

# Classify deps for rule based
######################################################################################################################
# always a part of the noun
tied_deps = ['det', 'neg', 'nmod:poss', 'compound', 'mwe', 'case', 'mark', 'auxpass', 'name', 'aux', 'nummod']
tied_couples = [['auxpass', 'nsubjpass']]

# Optional to describe the noun
dep_type_optional = ['advmod', 'npadvmod', 'conj', 'poss', 'nmod:poss',
                     'xcomp', 'nmod:npmod']

dep_type_complement = ['dobj', 'nmod']

# strict sequence for valid expansion
others_to_seq = ['quantmod', 'cop']
couple_to_seq = {'quantmod': ['amod'], 'cop': ['nsubjpass', 'nsubj']}

# Dependency that create relation to another phrase
combined_with = ['acl', 'relcl', 'acl:relcl', 'ccomp', 'advcl']


######################################################################################################################


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


def get_all_offspring(token):
    children_lst = []
    for child in token.children:
        children_lst.append(child)
        children_lst.extend(get_all_offspring(child))
    return children_lst


def get_preposition_if_exist(head):
    prep_lst = []
    for child in head.children:
        if child.dep_ in ['case', 'mark'] and child.i < head.i:
            prep_lst.append(child)
    children_lst = []
    for child in prep_lst:
        children_lst.extend(get_all_offspring(child))
    prep_lst.extend(children_lst)
    return prep_lst


def combine_tied_deps_recursively_and_combine_their_children(head, is_head=False, head_word_index=-1,
                                                             optional_deps_type_lst=[]):
    combined_children_lst = []
    prep_lst = []
    if head_word_index == -1:
        if is_head:
            prep_lst = get_preposition_if_exist(head)
    combined_tied_tokens = [head]
    tied_couples_to_add = get_tied_couples(head.children)
    for child in head.children:
        if child in prep_lst:
            continue
        if head_word_index != -1:
            if head.dep_ == 'nmod':
                if child.dep_ in ['case', 'mark']:
                    continue
        if child.dep_ in tied_deps or child.dep_ in optional_deps_type_lst or child in tied_couples_to_add:
            # if (child.dep_ in ['case', 'mark']) or child.dep_ in optional_deps_type_lst:
            #     temp_tokens, temp_children, _ = combine_tied_deps_recursively_and_combine_their_children(child, -1, 2)
            #     temp_tokens = [(token_couple[0], 2) for token_couple in temp_tokens]
            # else:
            temp_tokens, temp_children, _ = combine_tied_deps_recursively_and_combine_their_children(child)
            combined_tied_tokens.extend(temp_tokens)
            combined_children_lst.extend(temp_children)
        else:
            combined_children_lst.append(child)
    return combined_tied_tokens, combined_children_lst, prep_lst


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
        sub_np_lst_couple, lst_children_first, _ = combine_tied_deps_recursively_and_combine_their_children(couple[0],
                                                                                                         False, -1)
        sub_np_lst_couple_second, lst_children_second, _ = combine_tied_deps_recursively_and_combine_their_children(
            couple[1], False, -1)
        sub_np_lst_couple.extend(sub_np_lst_couple_second)
        sub_np_lst_couple = [(sub_np_lst_couple, 3)]
        all_sub_of_sub = []
        get_children_expansion(all_sub_of_sub, lst_children_first, head)
        get_children_expansion(all_sub_of_sub, lst_children_second, head)
        if all_sub_of_sub:
            sub_np_lst_couple.append(all_sub_of_sub)
        sub_np_lst.append(sub_np_lst_couple)


dep_type_in_sequential = set()


def get_all_valid_sub_special(token):
    # try:
    #     if token.text == '%':
    #         print(token.text)
    # except:
    #     print("error")
    sub_np_lst, lst_children, prep_lst = combine_tied_deps_recursively_and_combine_their_children(token, True, -1, ['nsubj'])
    sub_np_lst = [(sub_np_lst, 2)]
    sub_np = []
    lst_to_skip, couple_lst = remove_conj_if_cc_exist(lst_children)
    lst_to_skip.extend(prep_lst)
    for child in lst_children:
        all_sub_of_sub = []
        if child in lst_to_skip or child.text in ['-', '(', ')', '"']:
            continue
        if child.dep_ in dep_type_optional:
            all_sub_of_sub = get_all_valid_sub_np(child, 3)
        elif child.dep_ in dep_type_complement:
            all_sub_of_sub = get_all_valid_sub_np(child, 1)
        elif child.dep_ in combined_with:
            all_sub_of_sub = get_all_valid_sub_special(child)
        elif child.dep_ == 'amod':
            all_sub_of_sub = get_all_children(child, 4)
            all_sub_of_sub = [(all_sub_of_sub, 3)]
        if all_sub_of_sub:
            sub_np.append(all_sub_of_sub)
    sub_np_lst_couples = []
    set_couple_deps(couple_lst, sub_np_lst_couples, token)
    if sub_np_lst_couples:
        sub_np.append(sub_np_lst_couples)
    if sub_np:
        sub_np_lst.append(sub_np)
    if prep_lst:
        sub_np_lst = [(prep_lst, 2)] + [sub_np_lst]
    return sub_np_lst


def get_all_children(head, head_token_type=4):
    combined_tied_tokens = [head]
    for child in head.children:
        temp_tokens = get_all_children(child, 2)
        combined_tied_tokens.extend(temp_tokens)
    return combined_tied_tokens


def get_children_expansion(sub_np_lst, lst_children, head):
    others = []
    # try:
    #     if head.text == '%':
    #         print(head.text)
    # except:
    #     print("error")
    lst_to_skip, tokens_to_add = remove_conj_if_cc_exist(lst_children)
    for child in lst_children:
        if child in lst_to_skip or child.text in ['-', '(', ')', '"']:
            continue
        sub_np = []
        all_sub_of_sub = []
        if child.dep_ in others_to_seq:
            others.append(child)
        else:
            if child.dep_ in dep_type_optional:
                all_sub_of_sub = get_all_valid_sub_np(child, 3)
            elif child.dep_ in dep_type_complement:
                all_sub_of_sub = get_all_valid_sub_np(child, 1)
            elif child.dep_ in combined_with:
                all_sub_of_sub = get_all_valid_sub_special(child)
            elif child.dep_ == 'amod':
                all_sub_of_sub = get_all_children(child, 4)
                all_sub_of_sub = [(all_sub_of_sub, 3)]
            if all_sub_of_sub:
                sub_np.append(all_sub_of_sub)
            sub_np_lst.extend(sub_np)
    couple_lst = []
    if others:
        initialize_couple_lst(others, couple_lst, lst_children)
    couple_lst.extend(tokens_to_add)
    set_couple_deps(couple_lst, sub_np_lst, head)


def get_all_valid_sub_np(head, type=1, head_word_index=-1):
    sub_np_lst, lst_children, prep_lst = combine_tied_deps_recursively_and_combine_their_children(head, True,
                                                                                                  head_word_index)
    sub_np_lst = [(sub_np_lst, type)]
    lst_children = [item for item in lst_children if item not in prep_lst]
    get_children_expansion(sub_np_lst, lst_children, head)
    if prep_lst:
        sub_np_lst = [(prep_lst, 2)] + [sub_np_lst]
    return sub_np_lst
