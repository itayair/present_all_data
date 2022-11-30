import parse_medical_data
from topic_clustering import utils_clustering
from expansions import valid_expansion_utils
import pickle

def filter_dict_by_lst(dict_noun_lemma_to_examples, dict_noun_lemma_to_counter, topic_lst):
    dict_noun_lemma_to_examples = {key: dict_noun_lemma_to_examples[key] for key in dict_noun_lemma_to_examples if
                                  key in topic_lst}
    dict_noun_lemma_to_counter = {key: dict_noun_lemma_to_counter[key] for key in dict_noun_lemma_to_counter if
                                  key in topic_lst}
    return dict_noun_lemma_to_examples, dict_noun_lemma_to_counter


def set_cover(dict_noun_lemma_to_examples):
    covered = set()
    topic_lst = set()
    noun_to_spans_lst = []
    for noun, tuples_span_lst in dict_noun_lemma_to_examples.items():
        spans_lst = [tuple_span[0] for tuple_span in tuples_span_lst]
        noun_to_spans_lst.append((noun, set(spans_lst)))
    print("start")
    while True:
        item = max(noun_to_spans_lst, key=lambda s: len(s[1] - covered))
        if len(item[1] - covered) > 0:
            covered.update(item[1])
            topic_lst.add(item[0])
        else:
            break
    return topic_lst


def filter_and_sort_dicts(dict_noun_lemma_to_examples, abbreviations_lst, dict_noun_lemma_to_counter, head_lst):
    dict_noun_lemma_to_examples = utils_clustering.get_dict_sorted_and_filtered(dict_noun_lemma_to_examples,
                                                                               abbreviations_lst,
                                                                               dict_noun_lemma_to_counter, head_lst)
    dict_noun_lemma_to_counter = utils_clustering.get_dict_sorted_and_filtered(dict_noun_lemma_to_counter,
                                                                               abbreviations_lst,
                                                                               dict_noun_lemma_to_counter, head_lst)
    # black_list = set()
    # for word, counter in dict_noun_lemma_to_counter.items():
    #     utils_clustering.is_should_be_removed(dict_noun_lemma_to_counter,
    #                                           dict_noun_lemma_to_examples[word], word,
    #                                           dict_word_to_his_synonym, black_list, dict_span_to_topic_entry)
    # topic_lst = set_cover(dict_noun_lemma_to_examples)
    # dict_noun_lemma_to_examples, dict_noun_lemma_to_counter = filter_dict_by_lst(dict_noun_lemma_to_examples,
    #                                                                             dict_noun_lemma_to_counter, topic_lst)
    return dict_noun_lemma_to_examples, dict_noun_lemma_to_counter


def initialize_token_expansions_information(all_valid_nps_lst, token, dict_span_to_rank, expansions_contain_word,
                                            dict_noun_lemma_to_examples, dict_noun_lemma_to_counter,
                                            dict_span_to_topic_entry, lemma_word, span):
    for sub_span in all_valid_nps_lst:
        if token in sub_span[0]:
            np_span = valid_expansion_utils.get_tokens_as_span(sub_span[0])
            dict_span_to_rank[np_span] = sub_span[1]
            lemma_lst = utils_clustering.from_tokens_to_lemmas(sub_span[0])
            expansions_contain_word.append((np_span, sub_span[1], lemma_lst))
    dict_noun_lemma_to_examples[lemma_word] = dict_noun_lemma_to_examples.get(lemma_word, [])
    dict_noun_lemma_to_examples[lemma_word].append((span, expansions_contain_word))
    dict_noun_lemma_to_counter[lemma_word] = dict_noun_lemma_to_counter.get(lemma_word, 0)
    dict_noun_lemma_to_counter[lemma_word] += 1
    dict_span_to_topic_entry[span] = dict_span_to_topic_entry.get(span, set())
    dict_span_to_topic_entry[span].add(lemma_word)


def combine_word_in_upper_case_to_word_in_lower_if_exist(dict_noun_lemma_to_counter, dict_noun_lemma_to_examples, upper_case_noun_lst):
    entries_to_remove = set()
    for upper_case_noun_entry in upper_case_noun_lst:
        lower_case_noun_entry = upper_case_noun_entry.lower()
        if lower_case_noun_entry in dict_noun_lemma_to_examples and upper_case_noun_entry in dict_noun_lemma_to_examples:
            entries_to_remove.add(upper_case_noun_entry)
            upper_case_noun_entry_examples = dict_noun_lemma_to_examples[upper_case_noun_entry]
            dict_noun_lemma_to_examples[lower_case_noun_entry].extend(upper_case_noun_entry_examples)
            if lower_case_noun_entry in dict_noun_lemma_to_counter and \
                    upper_case_noun_entry in dict_noun_lemma_to_counter:
                dict_noun_lemma_to_counter[lower_case_noun_entry] += dict_noun_lemma_to_counter[upper_case_noun_entry]
    for entry in entries_to_remove:
        dict_noun_lemma_to_examples.pop(entry, None)
        dict_noun_lemma_to_counter.pop(entry, None)



def convert_examples_to_clustered_data():
    examples = parse_medical_data.get_examples_from_special_format(False)
    dict_noun_lemma_to_noun_words = {}
    dict_noun_word_to_counter = {}
    dict_noun_lemma_to_synonyms = {}
    # noun_lemma_to_synonyms_file = open("load_data\\diabetes\\noun_lemma_to_synonyms.pkl", "rb")
    # dict_noun_lemma_to_synonyms = pickle.load(noun_lemma_to_synonyms_file)
    dict_noun_lemma_to_counter = {}
    dict_noun_lemma_to_examples = {}
    head_lst = set()
    abbreviations_lst = set()
    span_lst = set()
    dict_span_to_counter = {}
    dict_longest_span_to_counter = {}
    dict_span_to_topic_entry = {}
    dict_span_to_rank = {}
    dict_word_to_lemma = {}
    counter = 0
    counter_examples = 0
    dict_sentence_to_span_lst = {}
    upper_case_noun_lst = set()
    valid_span_lst = set()
    for biggest_noun_phrase, head_span, all_valid_nps_lst, sentence in examples:
        span = valid_expansion_utils.get_tokens_as_span(biggest_noun_phrase)
        dict_sentence_to_span_lst[sentence] = dict_sentence_to_span_lst.get(sentence, [])
        counter_examples += 1
        if span in span_lst:
            if span not in dict_sentence_to_span_lst[sentence]:
                dict_sentence_to_span_lst[sentence].append(span)
                if span in dict_longest_span_to_counter:
                    dict_longest_span_to_counter[span] += 1
                    counter += 1
                for sub_span in all_valid_nps_lst:
                    dict_span_to_counter[valid_expansion_utils.get_tokens_as_span(sub_span[0])] = dict_span_to_counter.get(valid_expansion_utils.get_tokens_as_span(sub_span[0]), 0) + 1
            continue
        dict_sentence_to_span_lst[sentence].append(span)
        span_lst.add(span)
        if head_span.lemma_.isupper():
            head = head_span.lemma_
        else:
            head = head_span.lemma_.lower()
        head_lst.add(head)
        tokens_already_counted = set()
        is_valid_example = False
        for word in biggest_noun_phrase:
            if word in tokens_already_counted:
                continue
            if not word.lemma_.isupper():
                lemma_word = word.lemma_.lower()
            else:
                lemma_word = word.lemma_
            dict_word_to_lemma[lemma_word] = lemma_word
            if word.pos_ in "NOUN":
                compound_noun = utils_clustering.combine_tied_deps_recursively_and_combine_their_children(word)
                compound_noun.sort(key=lambda x: x.i)
                for token in compound_noun:
                    if len(token.text) < 2 or token.dep_ in ['quantmod'] or token.text == '-':
                        continue
                    if token.lemma_.isupper():
                        lemma_token = token.lemma_
                        normalized_token = token.text
                        upper_case_noun_lst.add(normalized_token)
                    else:
                        lemma_token = token.lemma_.lower()
                        normalized_token = token.text.lower()
                    if lemma_token in ['sciatica', 'cause', 'causing', 'diagnosing', 'diagnosis',
                                       'pain', 'chest', 'abortion', 'diabetes', 'diabete', 'jaundice', 'meningitis',
                                       'pneumonia']:
                        continue
                    if token not in tokens_already_counted:
                        dict_noun_word_to_counter[normalized_token] = dict_noun_word_to_counter.get(normalized_token, 0) + 1
                        dict_noun_lemma_to_noun_words[lemma_token] = dict_noun_lemma_to_noun_words.get(lemma_token, set())
                        dict_noun_lemma_to_noun_words[lemma_token].add(normalized_token)
                        tokens_already_counted.add(token)
                        expansions_contain_word = []
                        initialize_token_expansions_information(all_valid_nps_lst, token, dict_span_to_rank,
                                                                expansions_contain_word,
                                                                dict_noun_lemma_to_examples, dict_noun_lemma_to_counter,
                                                                dict_span_to_topic_entry, lemma_token, span)
                        is_valid_example = True
        if is_valid_example:
            dict_longest_span_to_counter[span] = 1
            if not valid_span_lst:
                dict_span_to_counter[span] = dict_span_to_counter.get(span, 0) + 1
                print("There is longest expansion that isn't in the all_valid_nps_lst")
            for sub_span in all_valid_nps_lst:
                dict_span_to_counter[valid_expansion_utils.get_tokens_as_span(sub_span[0])] = dict_span_to_counter.get(valid_expansion_utils.get_tokens_as_span(sub_span[0]), 0) + 1
            valid_span_lst.add(span)
            counter += 1
    print(counter)
    combine_word_in_upper_case_to_word_in_lower_if_exist(dict_noun_lemma_to_counter, dict_noun_lemma_to_examples,
                                                         upper_case_noun_lst)
    # dict_noun_lemma_to_counter, dict_noun_lemma_to_examples = utils_clustering.synonyms_consolidation(
    #     dict_noun_lemma_to_examples,
    #     dict_noun_lemma_to_counter, dict_noun_lemma_to_synonyms, 'wordnet')
    dict_noun_lemma_to_counter, dict_noun_lemma_to_examples = utils_clustering.synonyms_consolidation(
        dict_noun_lemma_to_examples,
        dict_noun_lemma_to_counter, dict_noun_lemma_to_synonyms, 'umls')
    not_in_head_lst = []
    for key in dict_noun_lemma_to_examples:
        if key not in head_lst and len(dict_noun_lemma_to_examples[key]) < 2:
            not_in_head_lst.append(key)
    dict_noun_lemma_to_examples, dict_noun_lemma_to_counter = filter_and_sort_dicts(dict_noun_lemma_to_examples,
                                                                                   abbreviations_lst,
                                                                                   dict_noun_lemma_to_counter, head_lst)
    dict_lemma_to_synonyms = utils_clustering.create_dicts_for_words_similarity(dict_word_to_lemma)
    dict_lemma_to_synonyms.update(dict_noun_lemma_to_synonyms)
    return dict_noun_lemma_to_examples, dict_span_to_counter, dict_word_to_lemma, dict_lemma_to_synonyms, \
           dict_longest_span_to_counter, dict_noun_lemma_to_synonyms, dict_noun_lemma_to_noun_words, \
           dict_noun_lemma_to_counter, dict_noun_word_to_counter
