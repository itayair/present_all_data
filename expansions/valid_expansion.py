import nltk

from expansions import valid_expansion_utils
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
tied_deps = ['compound', 'mwe', 'case', 'mark', 'auxpass', 'name', 'aux', 'neg', 'quantmod', 'det']
tied_couples = [['auxpass', 'nsubjpass']]

dep_type_optional = ['advmod', 'dobj', 'npadvmod', 'nmod', 'nummod', 'conj', 'aux', 'poss', 'nmod:poss',
                     'xcomp']  # , 'conj', 'nsubj', 'appos'

acl_to_seq = ['acomp', 'dobj', 'nmod']  # acl and relcl + [[['xcomp'], ['aux']], 'dobj']
others_to_seq = ['quantmod', 'cop']  # 'cc',
combined_with = ['acl', 'relcl', 'acl:relcl', 'ccomp', 'advcl', 'amod']  # +, 'cc'
couple_to_seq = {'quantmod': ['amod'], 'cop': ['nsubjpass']}  # 'nsubjpass': ['amod'] ,'cc': ['conj', 'nmod']

low_val_dep = ['neg', 'nmod:poss', 'case', 'mark', 'auxpass', 'aux', 'nummod', 'quantmod', 'cop']
med_val_dep = ['nsubjpass', 'advmod', 'npadvmod', 'conj', 'poss', 'nmod:poss', 'xcomp', 'nmod:npmod', 'dobj', 'nmod',
               'amod', 'nsubj', 'acl', 'relcl', 'acl:relcl', 'ccomp', 'advcl']
max_val_dep = ['compound', 'mwe', 'name']
neglect_deps = ['neg', 'case', 'mark', 'auxpass', 'aux', 'nummod', 'quantmod', 'cop']


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
        # if child.text in stop_words:
        #     continue
        if boundary_np_to_the_left > child.i:
            continue
        if child.dep_ in tied_deps or child in tied_couples_to_add:
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


def get_all_valid_sub_special(token, boundary_np_to_the_left):
    sub_np_lst, lst_children = combine_tied_deps_recursively_and_combine_their_children(token, boundary_np_to_the_left)
    complete_children = []
    lst_to_skip, tokens_to_add = remove_conj_if_cc_exist(lst_children)
    complete_occurrences = 0
    for child in lst_children:
        if child.text in stop_words:
            continue
        if child in lst_to_skip:
            continue
        if child.dep_ in ['dobj', 'advcl', 'nmod']:  # 'cc', 'conj', 'aux', 'auxpass', 'cop', 'nsubjpass'
            all_sub_of_sub = get_all_valid_sub_np(child, boundary_np_to_the_left)
            if complete_occurrences > 0 and child.dep_ in ['nmod', 'advcl'] and len(all_sub_of_sub) > 1:
                new_sub_np_lst = []
                optional_lst = []
                for item in all_sub_of_sub:
                    if isinstance(item, list):
                        optional_lst.append(item)
                    else:
                        new_sub_np_lst.append(item)
                new_sub_np_lst.sort(key=lambda x: x.i)
                new_sub_np_lst = new_sub_np_lst + optional_lst
                if new_sub_np_lst[0].dep_ in ['case', 'mark']:
                    sub_np_lst = sub_np_lst + [new_sub_np_lst]
                    complete_occurrences += 1
                    continue
            sub_np_lst = sub_np_lst + all_sub_of_sub
            complete_occurrences += 1
        else:
            complete_children.append(child)
    if complete_occurrences > 1:
        new_sub_np_lst = []
        optional_lst = []
        for item in sub_np_lst:
            if isinstance(item, list):
                optional_lst.append(item)
            else:
                new_sub_np_lst.append(item)
        sub_np_lst = new_sub_np_lst + optional_lst
    couple_lst = []
    couple_lst.extend(tokens_to_add)
    sub_np_lst_couples = []
    set_couple_deps(couple_lst, boundary_np_to_the_left, sub_np_lst_couples, [])
    if sub_np_lst_couples:
        sub_np_lst.append(sub_np_lst_couples)
    for child in complete_children:
        all_sub_of_sub = get_all_valid_sub_np(child, boundary_np_to_the_left)
        sub_np_lst.append(all_sub_of_sub)
    if not sub_np_lst:
        return []
    return sub_np_lst


def get_children_expansion(sub_np_lst, lst_children, boundary_np_to_the_left, head):
    others = []
    lst_to_skip, tokens_to_add = remove_conj_if_cc_exist(lst_children)
    for child in lst_children:
        if child.text in stop_words:
            continue
        if child in lst_to_skip:
            continue
        sub_np = []
        if child.dep_ in dep_type_optional:
            all_sub_of_sub = get_all_valid_sub_np(child, boundary_np_to_the_left)
            sub_np.append(all_sub_of_sub)
            if sub_np:
                sub_np_lst.extend(sub_np)
        elif child.dep_ in combined_with:
            all_sub_of_sub = get_all_valid_sub_special(child, boundary_np_to_the_left)
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


def get_all_valid_sub_np(head, boundary_np_to_the_left):
    sub_np_lst, lst_children = combine_tied_deps_recursively_and_combine_their_children(head, boundary_np_to_the_left)
    get_children_expansion(sub_np_lst, lst_children, boundary_np_to_the_left, head)
    return sub_np_lst


def get_score(np, head_word):
    val = 0
    for token in np:
        # for word in item[0]:
        #     if word in already_counted:
        #         continue
        #     new_sub_np.append(word)
        if token == head_word:
            val += 1
            continue
        if token.dep_ in neglect_deps or token.lemma_ in stop_words or token.text == '-':
            continue
        val += 1
    return val


def check_validity_span(np):
    hyphen_lst = []
    idx_lst = []
    for token in np:
        if token.text == '-':
            hyphen_lst.append(token)
            continue
        idx_lst.append(token.i)
    if hyphen_lst:
        for hyphen in hyphen_lst:
            if hyphen.i + 1 in idx_lst and hyphen.i - 1 in idx_lst:
                continue
            else:
                return False
        return True
    return True


def get_all_expansions_of_span_from_lst(span_lst):
    # examples_to_visualize = []
    counter = 0
    sub_np_final_lst_collection = []
    counter_duplication = 0
    all_span_with_more_than_hundred = []
    from_span_to_all_expansions = {}
    # all_valid_spans_of_all_expansions = set()
    for head_word, sentence_dep_graph, sentence in span_lst:
        counter += 1
        noun_phrase, head_word_in_np_index, boundary_np_to_the_left = valid_expansion_utils.get_np_boundary(
            head_word.i,
            sentence_dep_graph)
        if noun_phrase is None:
            continue
        if len(noun_phrase) > 15:
            continue
        # examples_to_visualize.append(noun_phrase)
        # all_valid_sub_np = valid_deps.get_all_valid_sub_np(noun_phrase[head_word_in_np_index])
        all_valid_sub_np = get_all_valid_sub_np(noun_phrase[head_word_in_np_index],
                                                boundary_np_to_the_left)
        sub_np_final_lst = valid_expansion_utils.from_lst_to_sequence(all_valid_sub_np)
        sub_np_final_spans = []
        valid_span_lst = []
        for np in sub_np_final_lst:
            # length_from_algorithm = len(sub_np)
            np.sort(key=lambda x: x.i)
            is_valid_np = check_validity_span(np)
            if not is_valid_np:
                continue
            # length_after_remove_duplication = len(new_sub_np)
            # if length_after_remove_duplication != length_from_algorithm:
            #     counter_duplication += 1
            # span = valid_expansion_utils.get_tokens_as_span(new_sub_np)
            # if span not in valid_span_lst:
            #     all_valid_spans_of_all_expansions.add(span)
            #     valid_span_lst.append(span)
            val = get_score(np, head_word)
            # span = valid_expansion_utils.get_tokens_as_span(new_sub_np)
            sub_np_final_spans.append((np, val))
        if len(valid_span_lst) > 100:
            all_span_with_more_than_hundred.append(noun_phrase)
        sub_np_final_spans.sort(key=lambda x: len(x[0]), reverse=True)
        if not sub_np_final_spans:
            continue
        longest_span = valid_expansion_utils.get_tokens_as_span(sub_np_final_spans[0][0])
        from_span_to_all_expansions[longest_span] = set()
        for span_as_lst in sub_np_final_spans:
            span = valid_expansion_utils.get_tokens_as_span(span_as_lst[0])
            from_span_to_all_expansions[longest_span].add(span)
        sub_np_final_lst_collection.append((sub_np_final_spans[0][0], head_word, sub_np_final_spans, sentence))
    # print(max_valid_expansions)
    file_name = ".\output_all_valid_expansions_result.txt"
    with open(file_name, 'w', encoding='utf-8') as f:
        for longest_span, collection in from_span_to_all_expansions.items():
            f.write(longest_span + ': ')
            idx = 0
            f.write('[')
            for span in collection:
                if idx != 0:
                    f.write(', ')
                f.write(span)
                idx += 1
            f.write(']')
            f.write('\n')
    print(counter_duplication)
    return sub_np_final_lst_collection
