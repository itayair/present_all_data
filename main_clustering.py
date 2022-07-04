import parse_medical_data
import utils
import pickle
import utils_clustering
import valid_expansion_utils

file_name = "output_noun_result.txt"
file_name_lemma = "output_noun_lemma_result.txt"


def filter_dict_by_lst(dict_noun_lemma_to_span, dict_noun_lemma_to_counter, black_list):
    dict_noun_lemma_to_span = {key: dict_noun_lemma_to_span[key] for key in dict_noun_lemma_to_span if
                               key not in black_list}
    dict_noun_lemma_to_counter = {key: dict_noun_lemma_to_counter[key] for key in dict_noun_lemma_to_counter if
                                  key not in black_list}
    return dict_noun_lemma_to_span, dict_noun_lemma_to_counter


def filter_and_sort_dicts(dict_noun_lemma_to_span, abbreviations_lst, dict_noun_lemma_to_counter, head_lst,
                          dict_word_to_his_synonym, dict_span_to_words):
    dict_noun_lemma_to_span = utils_clustering.get_dict_sorted_and_filtered(dict_noun_lemma_to_span, abbreviations_lst,
                                                                            dict_noun_lemma_to_counter, head_lst)
    dict_noun_lemma_to_counter = utils_clustering.get_dict_sorted_and_filtered(dict_noun_lemma_to_counter,
                                                                               abbreviations_lst,
                                                                               dict_noun_lemma_to_counter, head_lst)
    black_list = set()
    for word, counter in dict_noun_lemma_to_counter.items():
        utils_clustering.is_should_be_removed(dict_noun_lemma_to_counter,
                                              dict_noun_lemma_to_span[word], word,
                                              dict_word_to_his_synonym, black_list, dict_span_to_words)
    dict_noun_lemma_to_span, dict_noun_lemma_to_counter = filter_dict_by_lst(dict_noun_lemma_to_span,
                                                                             dict_noun_lemma_to_counter, black_list)
    return dict_noun_lemma_to_span, dict_noun_lemma_to_counter


def main():
    examples = parse_medical_data.get_examples_from_special_format()
    noun_lst = set()
    head_lst = set()
    noun_lemma_lst = set()
    dict_noun_lemma_to_span = {}
    dict_noun_lemma_to_counter = {}
    abbreviations_lst = set()
    span_lst = set()
    dict_span_to_counter = {}
    dict_span_to_words = {}
    dict_span_to_rank = {}
    dict_word_to_lemma = {}
    counter = 0
    for noun_phrase, head_span, all_valid_expansion in examples:
        span = valid_expansion_utils.get_tokens_as_span(all_valid_expansion[0][0])
        if span in span_lst:
            dict_span_to_counter[span] += 1
            continue
        counter += 1
        dict_span_to_counter[span] = 1
        span_lst.add(span)
        head = head_span.lemma_.lower()
        head_lst.add(head)
        lemmas_already_counted = set()
        for word in all_valid_expansion[0][0]:
            dict_word_to_lemma[word.text.lower()] = word.lemma_.lower()
            if word.pos_ in "NOUN":
                compound_noun = utils_clustering.combine_tied_deps_recursively_and_combine_their_children(word)
                compound_noun.sort(key=lambda x: x.i)
                compound_noun_span = utils.get_tokens_as_span(compound_noun)
                compound_noun_span_lemma_lst = []
                for word_to_lemma in compound_noun:
                    if len(word_to_lemma.text) < 2 or word_to_lemma.dep_ in ['quantmod'] or word_to_lemma.text == '-':
                        continue
                    lemma_word = word_to_lemma.lemma_.lower()
                    if lemma_word in ['sciatica', 'cause', 'causing', 'diagnosing', 'diagnosis', 'pain', 'chest']:
                        continue
                    compound_noun_span_lemma_lst.append(lemma_word)
                    if word_to_lemma.lemma_ not in lemmas_already_counted:
                        sub_string_contain_word = []
                        for sub_span in all_valid_expansion:
                            if word_to_lemma in sub_span[0]:
                                new_span = valid_expansion_utils.get_tokens_as_span(sub_span[0])
                                dict_span_to_rank[new_span] = sub_span[1]
                                sub_string_contain_word.append((new_span, sub_span[1]))
                        if sub_string_contain_word == []:
                            continue
                        dict_noun_lemma_to_span[lemma_word] = dict_noun_lemma_to_span.get(lemma_word, [])
                        dict_noun_lemma_to_span[lemma_word].append((span, sub_string_contain_word))
                        dict_noun_lemma_to_counter[lemma_word] = dict_noun_lemma_to_counter.get(lemma_word, 0)
                        dict_noun_lemma_to_counter[lemma_word] += 1
                        dict_span_to_words[span] = dict_span_to_words.get(span, set())
                        dict_span_to_words[span].add(lemma_word)
                        lemmas_already_counted.add(lemma_word)
                if compound_noun_span_lemma_lst == []:
                    continue
                compound_noun_span_lemma = utils_clustering.get_words_as_span(compound_noun_span_lemma_lst)
                noun_lemma_lst.add(compound_noun_span_lemma)
                noun_lst.add(compound_noun_span)
    dict_word_to_his_synonym = {}
    dict_noun_lemma_to_counter, dict_noun_lemma_to_span = utils_clustering.synonyms_consolidation(
        dict_noun_lemma_to_span,
        dict_noun_lemma_to_counter, dict_word_to_his_synonym, 'wordnet')
    dict_noun_lemma_to_counter, dict_noun_lemma_to_span = utils_clustering.synonyms_consolidation(
        dict_noun_lemma_to_span,
        dict_noun_lemma_to_counter, dict_word_to_his_synonym, 'umls')
    not_in_head_lst = []
    for key in dict_noun_lemma_to_span:
        if key not in head_lst and len(dict_noun_lemma_to_span[key]) < 2:
            not_in_head_lst.append(key)
    dict_noun_lemma_to_span, dict_noun_lemma_to_counter = filter_and_sort_dicts(dict_noun_lemma_to_span,
                                                                                abbreviations_lst,
                                                                                dict_noun_lemma_to_counter, head_lst,
                                                                                dict_word_to_his_synonym,
                                                                                dict_span_to_words)
    a_file = open("data.pkl", "wb")
    b_file = open("span_counter.pkl", "wb")
    c_file = open("word_to_lemma.pkl", "wb")
    pickle.dump(dict_noun_lemma_to_span, a_file)
    pickle.dump(dict_span_to_counter, b_file)
    pickle.dump(dict_word_to_lemma, c_file)
    c_file.close()
    b_file.close()
    a_file.close()
    # a_file = open("data.pkl", "rb")
    # output = pickle.load(a_file)
    # print(output)
    # a_file.close()
    with open(file_name, 'w', encoding='utf-8') as f:
        for span in noun_lst:
            f.write(span + '\n')
    with open(file_name_lemma, 'w', encoding='utf-8') as f:
        for span in noun_lemma_lst:
            f.write(span + '\n')
    print("Done!")


main()
