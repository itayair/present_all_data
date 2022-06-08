import parse_medical_data
import utils
import pickle
import utils_clustering

file_name = "output_noun_result.txt"
file_name_lemma = "output_noun_lemma_result.txt"


def filter_dict_by_lst(dict_noun_lemma_to_span, dict_noun_lemma_to_counter, black_list):
    dict_noun_lemma_to_span = {key: dict_noun_lemma_to_span[key] for key in dict_noun_lemma_to_span if
                               key not in black_list}
    dict_noun_lemma_to_counter = {key: dict_noun_lemma_to_counter[key] for key in dict_noun_lemma_to_counter if
                                  key not in black_list}
    return dict_noun_lemma_to_span, dict_noun_lemma_to_counter


def filter_and_sort_dicts(dict_noun_lemma_to_span, abbreviations_lst, dict_noun_lemma_to_counter, head_lst,
                          dict_word_to_his_synonym):
    dict_noun_lemma_to_span = utils_clustering.get_dict_sorted_and_filtered(dict_noun_lemma_to_span, abbreviations_lst,
                                                                            dict_noun_lemma_to_counter, head_lst)
    dict_noun_lemma_to_counter = utils_clustering.get_dict_sorted_and_filtered(dict_noun_lemma_to_counter,
                                                                               abbreviations_lst,
                                                                               dict_noun_lemma_to_counter, head_lst)
    black_list = set()
    for word, counter in dict_noun_lemma_to_counter.items():
        if counter > 3:
            break
        utils_clustering.is_should_be_removed(dict_noun_lemma_to_span, dict_noun_lemma_to_counter,
                                              dict_noun_lemma_to_span[word], word,
                                              dict_word_to_his_synonym, black_list)
    dict_noun_lemma_to_span, dict_noun_lemma_to_counter = filter_dict_by_lst(dict_noun_lemma_to_span, dict_noun_lemma_to_counter, black_list)
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
    dict_span_to_example = {}
    for example in examples:
        span = example[1]
        if span in span_lst:
            continue
        span_lst.add(span)
        span_doc = example[3]
        head = example[4].lemma_.lower()
        dict_span_to_example[span] = head
        head_lst.add(head)
        lemmas_already_counted = set()
        for word in span_doc:
            name = utils_clustering.isAbbr(word.text)
            if name != "":
                abbreviations_lst.add(name.lower())
            if word.pos_ in "NOUN":
                compound_noun = utils_clustering.combine_tied_deps_recursively_and_combine_their_children(word)
                compound_noun.sort(key=lambda x: x.i)
                compound_noun_span = utils.get_tokens_as_span(compound_noun)
                compound_noun_span_lemma_lst = []
                for word_to_lemma in compound_noun:
                    lemma_word = word_to_lemma.lemma_.lower()
                    compound_noun_span_lemma_lst.append(lemma_word)
                    if word_to_lemma.lemma_ not in lemmas_already_counted:
                        dict_noun_lemma_to_span[lemma_word] = dict_noun_lemma_to_span.get(lemma_word, [])
                        dict_noun_lemma_to_span[lemma_word].append(span)
                        dict_noun_lemma_to_counter[lemma_word] = dict_noun_lemma_to_counter.get(lemma_word, 0)
                        dict_noun_lemma_to_counter[lemma_word] += 1
                        lemmas_already_counted.add(lemma_word)
                compound_noun_span_lemma = utils_clustering.get_words_as_span(compound_noun_span_lemma_lst)
                noun_lemma_lst.add(compound_noun_span_lemma)
                noun_lst.add(compound_noun_span)
    valid_word_to_remove_lst = []
    for abbrev in abbreviations_lst:
        if abbrev in dict_noun_lemma_to_span:
            valid_word_to_remove_lst.append(abbrev)
            dict_noun_lemma_to_span[abbrev].append(abbrev)
    for valid_word in valid_word_to_remove_lst:
        abbreviations_lst.remove(valid_word)
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
    dict_noun_lemma_to_span, dict_noun_lemma_to_counter = filter_and_sort_dicts(dict_noun_lemma_to_span, abbreviations_lst, dict_noun_lemma_to_counter, head_lst, dict_word_to_his_synonym)
    dict_span_to_nouns = {}
    for key, spans in dict_noun_lemma_to_span.items():
        for span in spans:
            dict_span_to_nouns[span] = dict_span_to_nouns.get(span, [])
            dict_span_to_nouns[span].append(key)
    key_to_remove = set()
    for key, value in dict_noun_lemma_to_span.items():
        if len(value) > 1:
            continue
        nouns = dict_span_to_nouns[value[0]]
        nouns_share_span = []
        for noun in nouns:
            if noun == key:
                continue
            if dict_noun_lemma_to_counter[noun] > 1:
                key_to_remove.add(key)
                break
            if dict_noun_lemma_to_counter[noun] == 1:
                nouns_share_span.append(noun)
        # if len(nouns_share_span) > 0:
        #     nouns_share_span.append(key)
        #     head = dict_span_to_example[value[0]]
        #     # for noun in nouns_share_span:
        #     #     if noun == head:
    dict_noun_lemma_to_span, dict_noun_lemma_to_counter = filter_dict_by_lst(dict_noun_lemma_to_span,
                                                                             dict_noun_lemma_to_counter, key_to_remove)
    a_file = open("data.pkl", "wb")
    pickle.dump(dict_noun_lemma_to_span, a_file)
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
