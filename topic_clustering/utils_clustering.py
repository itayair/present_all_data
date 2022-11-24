from nltk.corpus import wordnet

import umls_loader

tied_deps = ['compound', 'mwe', 'name', 'nummod']


def get_synonyms_by_word(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.add(l.name())
    # aliases = umls_loader.umls_loader.get_term_aliases(word)
    # for syn in aliases:
    #     synonyms.add(syn)
    return synonyms


def create_dicts_for_words_similarity(dict_word_to_lemma):
    dict_lemma_to_synonyms = {}
    lemma_lst = set()
    for _, lemma in dict_word_to_lemma.items():
        lemma_lst.add(lemma)
    for lemma in lemma_lst:
        synonyms = get_synonyms_by_word(lemma)
        synonyms = set(synonyms)
        synonyms = [synonym for synonym in synonyms if synonym in lemma_lst]
        synonyms.append(lemma)
        dict_lemma_to_synonyms[lemma] = set(synonyms)
    dict_lemma_to_synonyms = {k: v for k, v in
                              sorted(dict_lemma_to_synonyms.items(), key=lambda item: len(item[1]),
                                     reverse=True)}
    return dict_lemma_to_synonyms


def print_dendrogram_tree_by_parenthesis(node, num_of_leaves, phrase_list, already_counted):
    if node['name'] >= num_of_leaves:
        sub_tree = []
        for child in node['children']:
            sub_tree_output = print_dendrogram_tree_by_parenthesis(child, num_of_leaves, phrase_list, already_counted)
            if sub_tree_output is None:
                continue
            sub_tree.append(sub_tree_output)
        return sub_tree
    if node['name'] in already_counted:
        return None
    already_counted.add(node['name'])
    return phrase_list[node['name']]


def create_tree(linked):
    def recurTree(tree):
        k = tree['name']
        if k not in inter:
            return
        for n in inter[k]:
            node = {
                "name": n,
                "parent": k,
                "children": []
            }
            tree['children'].append(node)
            recurTree(node)

    num_rows, _ = linked.shape
    inter = {}
    i = 0
    for row in linked:
        i += 1
        inter[i + num_rows] = [row[0], row[1]]

    tree = {
        "name": i + num_rows,
        "parent": None,
        "children": []
    }
    recurTree(tree);
    return tree


def combine_dicts(dict_clustered_spans_last, dict_clustered_spans):
    new_dict_clustered_spans = {}
    for phrase_main, phrases_lst in dict_clustered_spans.items():
        new_dict_clustered_spans[phrase_main] = []
        for phrase in phrases_lst:
            new_dict_clustered_spans[phrase_main].extend(dict_clustered_spans_last[phrase])
    return new_dict_clustered_spans


def combine_tied_deps_recursively_and_combine_their_children(head):
    combined_tied_tokens = [head]
    for child in head.children:
        if child.dep_ in tied_deps:
            temp_tokens = combine_tied_deps_recursively_and_combine_their_children(child)
            combined_tied_tokens.extend(temp_tokens)
    return combined_tied_tokens


def get_words_as_span(span_lst):
    span = ""
    idx = 0
    for word in span_lst:
        if idx != 0 and word != ',':
            span += ' '
        span += word
        idx += 1
    return span


def is_should_be_removed(dict_noun_lemma_to_counter, span_lst, original_word,
                         dict_word_to_his_synonym, black_list, dict_span_to_topic_entry):
    counter = 0
    for span in span_lst:
        if isinstance(span[0], list):
            continue
        for lemma_word in dict_span_to_topic_entry[span[0]]:
            if original_word == lemma_word or dict_word_to_his_synonym.get(lemma_word, None) == original_word:
                continue
            if lemma_word not in black_list and lemma_word in dict_noun_lemma_to_counter:
                if lemma_word in dict_word_to_his_synonym:
                    entry = dict_word_to_his_synonym[lemma_word]
                else:
                    entry = lemma_word
                # if dict_noun_lemma_to_counter[entry] > 1:
            counter += 1
            break
    if len(span_lst) == counter:
        black_list.add(original_word)
        print(original_word + " should be removed: " + str(len(span_lst)))


def isAbbr(name):
    if name.endswith('s'):
        name = name[0:-2]
    if name.isupper():
        if len(name) >= 2:
            return name
    return ""


def from_tokens_to_lemmas(tokens):
    lemma_lst = []
    for token in tokens:
        lemma_lst.append(token.lemma_)
    return lemma_lst


def get_dict_sorted_and_filtered(dict_noun_lemma_to_value, abbreviations_lst, dict_noun_lemma_to_counter, head_lst):
    dict_noun_lemma_to_value = {key: dict_noun_lemma_to_value[key] for key in dict_noun_lemma_to_value if
                                (key in head_lst or dict_noun_lemma_to_counter[key] > 1) and
                                ord('A') <= ord(key[:1]) <= ord('z') and len(key) > 1 and key not in abbreviations_lst}
    return dict_noun_lemma_to_value


def synonyms_consolidation(dict_noun_lemma_to_span, dict_noun_lemma_to_counter, dict_noun_lemma_to_synonyms,
                           synonyms_type='wordnet'):
    dict_noun_lemma_to_span_new = {}
    dict_noun_lemma_to_counter_new = {}
    dict_noun_lemma_to_span = {k: v for k, v in
                               sorted(dict_noun_lemma_to_span.items(), key=lambda item: len(item[1]),
                                      reverse=True)}
    word_lst = dict_noun_lemma_to_span.keys()
    already_calculated = []
    for word in word_lst:
        if word in already_calculated:
            continue
        synonyms = []
        # synonyms = dict_noun_lemma_to_synonyms[word]
        if synonyms_type == 'wordnet':
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonyms.append(lemma.name())
        else:
            aliases = umls_loader.umls_loader.get_term_aliases(word)
            for syn in aliases:
                synonyms.append(syn)
        dict_noun_lemma_to_span_new[word] = []
        dict_noun_lemma_to_span_new[word].extend(dict_noun_lemma_to_span[word])
        dict_noun_lemma_to_counter_new[word] = dict_noun_lemma_to_counter[word]
        dict_noun_lemma_to_synonyms[word] = dict_noun_lemma_to_synonyms.get(word, set())
        dict_noun_lemma_to_synonyms[word].add(word)
        synonyms = set(synonyms)
        if synonyms:
            for synonym in synonyms:
                if synonym in already_calculated:
                    continue
                if synonym != word and synonym in word_lst:
                    for spans in dict_noun_lemma_to_span[synonym]:
                        # new_spans_lst = []
                        # for span in spans[1]:
                        #     new_spans_lst.append((span[0].replace(synonym, word), span[1]))
                        # dict_noun_lemma_to_span_new[word].append((spans[0], new_spans_lst))
                        dict_noun_lemma_to_span_new[word].append((spans[0], spans[1]))
                    # dict_noun_lemma_to_span_new[word].extend(dict_noun_lemma_to_span[synonym])
                    dict_noun_lemma_to_counter_new[word] += dict_noun_lemma_to_counter[synonym]
                    # dict_word_to_his_synonym[synonym] = word
                    dict_noun_lemma_to_synonyms[word].add(synonym)
                    already_calculated.append(synonym)
        already_calculated.append(word)
    dict_noun_lemma_to_counter_new = {k: v for k, v in
                                      sorted(dict_noun_lemma_to_counter_new.items(), key=lambda item: item[1])}
    dict_noun_lemma_to_span_new = {k: v for k, v in
                                   sorted(dict_noun_lemma_to_span_new.items(), key=lambda item: len(item[1]),
                                          reverse=True)}
    return dict_noun_lemma_to_counter_new, dict_noun_lemma_to_span_new
