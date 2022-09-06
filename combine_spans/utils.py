from nltk.corpus import wordnet
import pickle
from combine_spans import span_comparison


def get_synonyms_by_word(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.add(l.name())
    # aliases = umls_loader.umls_loader.get_term_aliases(word)
    # for syn in aliases:
    #     synonyms.append(syn)
    return synonyms


def from_words_to_lemma_lst(span, dict_word_to_lemma):
    lemmas_lst = []
    for word in span:
        lemma = dict_word_to_lemma.get(word, None)
        if lemma is None:
            lemma = word
        lemmas_lst.append(lemma)
    return lemmas_lst


def load_data_dicts():
    a_file = open("load_data\\data.pkl", "rb")
    dict_of_topics = pickle.load(a_file)
    dict_of_topics = {k: v for k, v in
                      sorted(dict_of_topics.items(), key=lambda item: len(item[1]),
                             reverse=True)}
    b_file = open("load_data\\span_counter.pkl", "rb")
    dict_of_span_to_counter = pickle.load(b_file)
    c_file = open("load_data\\word_to_lemma.pkl", "rb")
    dict_word_to_lemma = pickle.load(c_file)
    d_file = open("load_data\\word_to_synonyms.pkl", "rb")
    dict_lemma_to_synonyms = pickle.load(d_file)
    return dict_of_topics, dict_of_span_to_counter, dict_word_to_lemma, dict_lemma_to_synonyms


dict_of_topics, dict_of_span_to_counter, dict_word_to_lemma, dict_lemma_to_synonyms = load_data_dicts()
dict_of_span_to_counter = {k: v for k, v in
                           sorted(dict_of_span_to_counter.items(), key=lambda item: item[1],
                                  reverse=True)}


def get_frequency_from_labels_lst(global_index_to_similar_longest_np, label_lst):
    num_of_labels = 0
    for label in label_lst:
        for span in global_index_to_similar_longest_np[label]:
            num_of_labels += dict_of_span_to_counter[span]
        # num_of_labels += len(global_index_to_similar_longest_np[label])
    return num_of_labels


def create_dict_lemma_word2vec_and_edit_distance(dict_lemma_to_synonyms, dict_word_to_lemma):
    words_lst = list(dict_word_to_lemma.keys())
    dict_lemma_to_close_words = {}
    counter = 0
    for word, lemma in dict_word_to_lemma.items():
        dict_lemma_to_close_words[word] = []
        # if word not in vocab_word2vec:
        #     counter += 1
        #     continue
        for word_ref_idx in range(counter + 1, len(words_lst)):
            word_ref = words_lst[word_ref_idx]
            lemma_ref = dict_word_to_lemma[word_ref]
            synonyms = [word, lemma] + dict_lemma_to_synonyms[lemma]
            synonyms = list(set(synonyms))
            if lemma == lemma_ref or lemma_ref in synonyms:
                continue
            # if word_ref not in vocab_word2vec:
            #     continue
            # sim_val = Word2Vec_model.similarity(word, word_ref)
            # if 0.8 < sim_val < 0.9:
            dict_lemma_to_close_words[word].extend(
                span_comparison.compare_edit_distance_of_synonyms(synonyms, word_ref, lemma_ref))
        counter += 1
    return dict_lemma_to_close_words


def get_most_frequent_span(lst_of_spans, dict_of_span_to_counter):
    most_frequent_span_value = -1
    most_frequent_span = None
    for span in lst_of_spans:
        val = dict_of_span_to_counter.get(span, 0)
        if val > most_frequent_span_value:
            most_frequent_span_value = val
            most_frequent_span = span
    return most_frequent_span


def create_dicts_for_words_similarity(dict_word_to_lemma):
    dict_lemma_to_synonyms = {}
    lemma_lst = set()
    for _, lemma in dict_word_to_lemma.items():
        lemma_lst.add(lemma)
    for lemma in lemma_lst:
        synonyms = get_synonyms_by_word(lemma)
        synonyms = [synonym for synonym in synonyms if synonym in lemma_lst]
        dict_lemma_to_synonyms[lemma] = synonyms
    dict_lemma_to_synonyms = {k: v for k, v in
                              sorted(dict_lemma_to_synonyms.items(), key=lambda item: len(item[1]),
                                     reverse=True)}
    return dict_lemma_to_synonyms


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


def get_leaves(nodes_lst, leaves_lst, visited):  # function for dfs
    for node in nodes_lst:
        if node not in visited:
            if len(node.children) == 0:
                leaves_lst.add(node)
            visited.add(node)
        get_leaves(node.children, leaves_lst, visited)


def remove_redundant_nodes(leaves_lst, nodes_lst):
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
