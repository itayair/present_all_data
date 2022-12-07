from nltk.corpus import wordnet
import pickle
import torch
import nltk


def load_data_dicts():
    directory_relative_path = "load_data\\sciatica\\"
    a_file = open(directory_relative_path + "noun_lemma_to_example.pkl", "rb")
    topics_dict = pickle.load(a_file)
    topics_dict = {k: v for k, v in
                   sorted(topics_dict.items(), key=lambda item: len(item[1]),
                          reverse=True)}
    b_file = open(directory_relative_path + "span_counter.pkl", "rb")
    dict_span_to_counter = pickle.load(b_file)
    c_file = open(directory_relative_path + "word_to_lemma.pkl", "rb")
    dict_word_to_lemma = pickle.load(c_file)
    d_file = open(directory_relative_path + "lemma_to_synonyms.pkl", "rb")
    dict_lemma_to_synonyms = pickle.load(d_file)
    e_file = open(directory_relative_path + "longest_span_to_counter.pkl", "rb")
    dict_longest_span_to_counter = pickle.load(e_file)
    f_file = open(directory_relative_path + "noun_lemma_to_synonyms.pkl", "rb")
    dict_noun_lemma_to_synonyms = pickle.load(f_file)
    g_file = open(directory_relative_path + "noun_lemma_to_noun_words.pkl", "rb")
    dict_noun_lemma_to_noun_words = pickle.load(g_file)
    h_file = open(directory_relative_path + "noun_lemma_to_counter.pkl", "rb")
    dict_noun_lemma_to_counter = pickle.load(h_file)
    i_file = open(directory_relative_path + "noun_word_to_counter.pkl", "rb")
    dict_noun_word_to_counter = pickle.load(i_file)

    return topics_dict, dict_span_to_counter, dict_word_to_lemma, dict_lemma_to_synonyms, \
           dict_longest_span_to_counter, dict_noun_lemma_to_synonyms, dict_noun_lemma_to_noun_words, \
           dict_noun_lemma_to_counter, dict_noun_word_to_counter


dict_of_topics, dict_span_to_counter, dict_word_to_lemma, dict_lemma_to_synonyms, \
dict_longest_span_to_counter, dict_noun_lemma_to_synonyms, dict_noun_lemma_to_noun_words, dict_noun_lemma_to_counter, \
dict_noun_word_to_counter = load_data_dicts()
dict_span_to_counter.update(dict_noun_word_to_counter)
dict_span_to_counter.update(dict_noun_lemma_to_counter)


# dict_of_span_to_counter = {k: v for k, v in
#                            sorted(dict_of_span_to_counter.items(), key=lambda item: item[1],
#                                   reverse=True)}


def get_synonyms_by_word(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.add(l.name())
    # aliases = umls_loader.umls_loader.get_term_aliases(word)
    # for syn in aliases:
    #     synonyms.append(syn)
    return synonyms


def from_words_to_lemma_lst(span):
    lemmas_lst = []
    for word in span:
        lemma = dict_word_to_lemma.get(word, None)
        if lemma is None:
            lemma = word
        lemmas_lst.append(lemma)
    return lemmas_lst


def word_contained_in_list_by_edit_distance(word, lst_words_ref):
    for word_ref in lst_words_ref:
        val = nltk.edit_distance(word, word_ref)
        if val / max(len(word), len(word_ref)) <= 0.34:
            return True, word_ref
    return False, None


def compare_edit_distance_of_synonyms(synonyms, token, lemma_ref):
    close_words = set()
    for synonym in synonyms:
        edit_distance = nltk.edit_distance(synonym, token)
        edit_distance_lemma = nltk.edit_distance(synonym, lemma_ref)
        if edit_distance / max(len(token), len(synonym)) <= 0.34:
            close_words.add((synonym, token))
            continue
        if edit_distance_lemma / max(len(lemma_ref), len(synonym)) <= 0.34:
            close_words.add((synonym, lemma_ref))
            continue
    return list(close_words)


def remove_token_if_in_span(token, span):
    if token in span:
        span.remove(token)
        return True
    else:
        for synonym_lemma in dict_lemma_to_synonyms.get(token, []):
            if synonym_lemma in span:
                span.remove(synonym_lemma)
                return True
    return False


def is_similar_meaning_between_span(span_1, span_2):
    span_1_lemma_lst = from_words_to_lemma_lst(span_1)
    span_2_lemma_lst = from_words_to_lemma_lst(span_2)
    not_satisfied = []
    for lemma in span_1_lemma_lst:
        is_exist = remove_token_if_in_span(lemma, span_2_lemma_lst)
        if not is_exist:
            not_satisfied.append(lemma)
    if len(not_satisfied) > 2:
        return False
    if len(span_1_lemma_lst) == 1 and not_satisfied:
        return False
    for lemma in not_satisfied:
        is_exist, lemma_to_remove = word_contained_in_list_by_edit_distance(lemma, span_2_lemma_lst)
        if is_exist:
            span_2_lemma_lst.remove(lemma_to_remove)
            continue
        return False
    return True


def create_dicts_length_to_span_and_span_to_list(span_to_group_members, dict_span_to_lst):
    dict_length_to_span = {}
    for span, sub_set in span_to_group_members.items():
        span_as_lst = dict_span_to_lst[span]
        if len(span_as_lst) == 1:
            continue
        dict_length_to_span[len(span_as_lst)] = dict_length_to_span.get(len(span_as_lst), [])
        dict_length_to_span[len(span_as_lst)].append((span, span_as_lst))
        dict_span_to_lst[span] = span_as_lst
    return dict_length_to_span


def get_average_value(spans_lst, dict_span_to_rank):
    average_val = 0
    for span in spans_lst:
        average_val += dict_span_to_rank[span]
    average_val = average_val / len(spans_lst)
    return int(average_val * 10)


def create_dict_lemma_word2vec_and_edit_distance():
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
            dict_lemma_to_close_words[word].extend(compare_edit_distance_of_synonyms(synonyms, word_ref, lemma_ref))
        counter += 1
    return dict_lemma_to_close_words


def get_weighted_average_vector_of_some_vectors_embeddings(spans_embeddings, common_np_lst):
    weighted_average_vector = torch.zeros(spans_embeddings[0].shape)
    total_occurrences = 0
    for idx, embedding_vector in enumerate(spans_embeddings):
        # total_occurrences += dict_of_span_to_counter[common_np_lst[idx]]
        # weighted_average_vector += embedding_vector * common_np_lst[idx]
        weighted_average_vector += embedding_vector
    # weighted_average_vector /= total_occurrences
    weighted_average_vector /= len(common_np_lst)
    return weighted_average_vector


def get_non_clustered_group_numbers(label_to_cluster, span_to_group_members, dict_label_to_spans_group):
    all_group_numbers = set(label_to_cluster.keys())
    already_grouped = set()
    common_span_lst = []
    for common_span, group_numbers in span_to_group_members.items():
        already_grouped.update(group_numbers)
        common_span_lst.append(common_span)
    res_group_numbers = [item for item in all_group_numbers if item not in already_grouped]
    dict_label_to_longest_np_without_common_sub_np = {}
    for num in res_group_numbers:
        dict_label_to_longest_np_without_common_sub_np[num] = dict_label_to_spans_group[num]
    return dict_label_to_longest_np_without_common_sub_np, common_span_lst


def get_most_frequent_span(lst_of_spans):
    most_frequent_span_value = -1
    most_frequent_span = None
    for span in lst_of_spans:
        val = dict_span_to_counter.get(span, 0)
        if val > most_frequent_span_value:
            most_frequent_span_value = val
            most_frequent_span = span
    return most_frequent_span


def convert_dict_label_to_spans_to_most_frequent_span_to_label(dict_label_to_spans_group):
    dict_span_to_label = {}
    for label, tuple_of_spans_lst in dict_label_to_spans_group.items():
        spans_lst = [tuple_of_span[0] for tuple_of_span in tuple_of_spans_lst]
        most_frequent_span = get_most_frequent_span(spans_lst)
        dict_span_to_label[most_frequent_span] = label
    return dict_span_to_label


def update_span_to_group_members_with_longest_answers_dict(span_to_group_members, dict_label_to_spans_group,
                                                           dict_span_to_similar_spans):
    dict_longest_answer_to_label_temp = {}
    for label, tuple_of_spans_lst in dict_label_to_spans_group.items():
        spans_lst = [tuple_of_span[0] for tuple_of_span in tuple_of_spans_lst]
        most_frequent_span = get_most_frequent_span(spans_lst)
        is_common_span = False
        for span, label_lst in span_to_group_members.items():
            if label in label_lst:
                similar_spans_lst = dict_span_to_similar_spans[span]
                intersection_spans_lst = set(spans_lst).intersection(similar_spans_lst)
                if intersection_spans_lst:
                    dict_span_to_similar_spans[span].update(spans_lst)
                    is_common_span = True
                    break
        if not is_common_span:
            dict_longest_answer_to_label_temp[most_frequent_span] = [label]
            dict_span_to_similar_spans[most_frequent_span] = set(spans_lst)

    span_to_group_members.update(dict_longest_answer_to_label_temp)






def get_dict_spans_group_to_score(span_to_group_members, dict_span_to_rank, dict_span_to_similar_spans,
                                  dict_label_to_spans_group):
    # dict_span_to_label = convert_dict_label_to_spans_to_most_frequent_span_to_label(dict_label_to_spans_group)
    # update_span_to_group_members_with_longest_answers_dict(span_to_group_members, dict_span_to_label, dict_span_to_similar_spans)
    update_span_to_group_members_with_longest_answers_dict(span_to_group_members, dict_label_to_spans_group,
                                                           dict_span_to_similar_spans)
    dict_score_to_collection_of_sub_groups = {}
    for key, group in span_to_group_members.items():
        average_val = get_average_value(dict_span_to_similar_spans[key], dict_span_to_rank)
        dict_score_to_collection_of_sub_groups[average_val] = dict_score_to_collection_of_sub_groups.get(
            average_val, [])
        dict_score_to_collection_of_sub_groups[average_val].append((key, set(group)))
    dict_score_to_collection_of_sub_groups = {k: v for k, v in
                                              sorted(dict_score_to_collection_of_sub_groups.items(),
                                                     key=lambda item: item[0], reverse=True)}
    for score, sub_group_lst in dict_score_to_collection_of_sub_groups.items():
        sub_group_lst.sort(key=lambda tup: len(tup[1]), reverse=True)
    return dict_score_to_collection_of_sub_groups
