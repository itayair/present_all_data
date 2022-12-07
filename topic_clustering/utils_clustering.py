from nltk.corpus import wordnet
import requests
from expansions import valid_expansion_utils
import datetime
import json
from nltk.corpus import stopwords

# import umls_loader
stop_words = set(stopwords.words('english'))
tied_deps = ['compound', 'mwe', 'name', 'nummod']
neglect_deps = ['neg', 'case', 'mark', 'auxpass', 'aux', 'nummod', 'quantmod', 'cop']
query_words = ['sciatica', 'cause', 'causing', 'diagnosing', 'diagnosis',
               'pain', 'chest', 'abortion', 'diabetes', 'diabete', 'jaundice', 'meningitis',
               'pneumonia']

noun_collections_lst = []
dict_span_to_topic_entry = {}
dict_span_to_rank = {}
dict_noun_lemma_to_counter = {}
dict_noun_lemma_to_examples = {}
dict_noun_lemma_to_noun_words = {}
dict_noun_word_to_counter = {}
dict_noun_lemma_to_span = {}


class AnswersCollection:
    def __init__(self, all_valid_nps_lst, token):
        self.dict_span_to_lemma_lst = {}
        self.dict_span_to_rank = {}
        self.longest_np = ""
        self.initialize(token, all_valid_nps_lst)

    def initialize(self, token, all_valid_nps_lst):
        is_longest_np = True
        for sub_span in all_valid_nps_lst:
            if token in sub_span[0]:
                np_span = valid_expansion_utils.get_tokens_as_span(sub_span[0])
                lemma_lst = from_tokens_to_lemmas(sub_span[0])
                self.dict_span_to_lemma_lst[np_span] = lemma_lst
                if is_longest_np:
                    self.longest_np = np_span
                self.dict_span_to_rank[np_span] = sub_span[1]


class NounCollections:
    def __init__(self, noun):
        self.noun = noun
        self.answers_collection_lst = []

    def add_answers_collection(self, answers_collection):
        self.answers_collection_lst.append(answers_collection)


def filter_dict_by_lst(topic_lst):
    global dict_noun_lemma_to_examples, dict_noun_lemma_to_counter
    dict_noun_lemma_to_examples = {key: dict_noun_lemma_to_examples[key] for key in dict_noun_lemma_to_examples if
                                   key in topic_lst}
    dict_noun_lemma_to_counter = {key: dict_noun_lemma_to_counter[key] for key in dict_noun_lemma_to_counter if
                                  key in topic_lst}
    return dict_noun_lemma_to_examples, dict_noun_lemma_to_counter


def set_cover():
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


def update_recurrent_span(dict_sentence_to_span_lst, sentence, span, dict_longest_span_to_counter, all_valid_nps_lst,
                          dict_span_to_counter, counter):
    if span not in dict_sentence_to_span_lst[sentence]:
        dict_sentence_to_span_lst[sentence].append(span)
        if span in dict_longest_span_to_counter:
            dict_longest_span_to_counter[span] += 1
            counter += 1
        for sub_span in all_valid_nps_lst:
            dict_span_to_counter[
                valid_expansion_utils.get_tokens_as_span(sub_span[0])] = dict_span_to_counter.get(
                valid_expansion_utils.get_tokens_as_span(sub_span[0]), 0) + 1
    return counter


def update_new_valid_example(span, dict_longest_span_to_counter, all_valid_nps_lst,
                             dict_span_to_counter,
                             valid_expansion_utils, counter, valid_span_lst):
    dict_longest_span_to_counter[span] = 1
    if not valid_span_lst:
        dict_span_to_counter[span] = dict_span_to_counter.get(span, 0) + 1
        print("There is longest expansion that isn't in the all_valid_nps_lst")
    for sub_span in all_valid_nps_lst:
        dict_span_to_counter[valid_expansion_utils.get_tokens_as_span(sub_span[0])] = dict_span_to_counter.get(
            valid_expansion_utils.get_tokens_as_span(sub_span[0]), 0) + 1
    valid_span_lst.add(span)
    counter += 1
    return counter


def initialize_token_expansions_information(all_valid_nps_lst, token, lemma_word, span):
    expansions_contain_word = []
    answers_collection = AnswersCollection(all_valid_nps_lst, token)
    noun_collections_lst.append(answers_collection)
    for sub_span in all_valid_nps_lst:
        if token in sub_span[0]:
            np_span = valid_expansion_utils.get_tokens_as_span(sub_span[0])
            dict_span_to_rank[np_span] = sub_span[1]
            lemma_lst = from_tokens_to_lemmas(sub_span[0])
            expansions_contain_word.append((np_span, sub_span[1], lemma_lst))
    dict_noun_lemma_to_examples[lemma_word] = dict_noun_lemma_to_examples.get(lemma_word, [])
    dict_noun_lemma_to_examples[lemma_word].append((span, expansions_contain_word))
    dict_noun_lemma_to_counter[lemma_word] = dict_noun_lemma_to_counter.get(lemma_word, 0)
    dict_noun_lemma_to_counter[lemma_word] += 1
    dict_span_to_topic_entry[span] = dict_span_to_topic_entry.get(span, set())
    dict_span_to_topic_entry[span].add(lemma_word)


def add_word_collection_to_data_structures(word, tokens_already_counted, lemma_already_counted,
                                           all_valid_nps_lst, span):
    is_valid_example = False
    if word.pos_ in ['NOUN','ADJ']:
        compound_noun = combine_tied_deps_recursively_and_combine_their_children(word)
        compound_noun.sort(key=lambda x: x.i)
        is_valid_example = False
        for token in compound_noun:
            if token.dep_ in ['quantmod'] or token.text == '-':
                continue
            lemma_token = token.lemma_.lower()
            if lemma_token in lemma_already_counted:
                continue
            lemma_already_counted.add(lemma_token)
            if lemma_token in query_words:
                continue
            if token not in tokens_already_counted:
                dict_noun_word_to_counter[lemma_token] = dict_noun_word_to_counter.get(lemma_token, 0) + 1
                dict_noun_lemma_to_noun_words[lemma_token] = dict_noun_lemma_to_noun_words.get(lemma_token,
                                                                                               set())
                if not dict_noun_lemma_to_noun_words[lemma_token]:
                    noun_collections = NounCollections(lemma_token)
                    noun_collections_lst.append(noun_collections)
                tokens_already_counted.add(token)
                initialize_token_expansions_information(all_valid_nps_lst, token, lemma_token, span)
                is_valid_example = True
    return is_valid_example


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
        if token.dep_ in neglect_deps or token.lemma_ in stop_words or token.text == '-':
            continue
        lemma_lst.append(token.lemma_.lower())
    return lemma_lst


def get_dict_sorted_and_filtered(dict_noun_lemma_to_value, abbreviations_lst, dict_noun_lemma_to_counter, head_lst):
    dict_noun_lemma_to_value = {key: dict_noun_lemma_to_value[key] for key in dict_noun_lemma_to_value if
                                (key in head_lst or dict_noun_lemma_to_counter[key] > 1) and
                                ord('A') <= ord(key[:1]) <= ord('z') and len(key) > 1 and key not in abbreviations_lst}
    return dict_noun_lemma_to_value


def get_synonyms_by_word(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.add(l.name())
    return synonyms


def create_dicts_for_words_similarity(dict_word_to_lemma):
    dict_lemma_to_synonyms = {}
    lemma_lst = set()
    for _, lemma in dict_word_to_lemma.items():
        lemma_lst.add(lemma)
    for lemma in lemma_lst:
        synonyms = get_synonyms_by_word(lemma)
        synonyms = [synonym for synonym in synonyms if synonym in lemma_lst]
        synonyms.append(lemma)
        dict_lemma_to_synonyms[lemma] = set(synonyms)
    # post_data = json.dumps(list(lemma_lst))
    # dict_response = requests.post('http://127.0.0.1:5000/create_synonyms_dictionary/', params={"words": post_data})
    # dict_lemma_to_synonyms_UMLS = dict_response.json()["synonyms"]
    # dict_lemma_to_synonyms.update(dict_lemma_to_synonyms_UMLS)
    # dict_lemma_to_synonyms = {k: v for k, v in
    #                           sorted(dict_lemma_to_synonyms.items(), key=lambda item: len(item[1]),
    #                                  reverse=True)}
    return dict_lemma_to_synonyms


def synonyms_consolidation(dict_noun_lemma_to_synonyms,
                           synonyms_type='wordnet'):
    global dict_noun_lemma_to_examples, dict_noun_lemma_to_counter
    dict_noun_lemma_to_examples_new = {}
    dict_noun_lemma_to_counter_new = {}
    dict_noun_lemma_to_examples = {k: v for k, v in
                                   sorted(dict_noun_lemma_to_examples.items(), key=lambda item: len(item[1]),
                                          reverse=True)}
    word_lst = dict_noun_lemma_to_examples.keys()
    post_data = json.dumps(list(word_lst))
    dict_response = requests.post('http://127.0.0.1:5000/create_noun_synonyms_dictionary/', params={"words": post_data})
    output = dict_response.json()["synonyms"]
    dict_noun_lemma_to_synonyms.update(output)
    for word, synonyms in dict_noun_lemma_to_synonyms.items():
        dict_noun_lemma_to_examples_new[word] = []
        dict_noun_lemma_to_examples_new[word].extend(dict_noun_lemma_to_examples[word])
        dict_noun_lemma_to_counter_new[word] = dict_noun_lemma_to_counter[word]
        for synonym in synonyms:
            if word == synonym:
                continue
            for spans in dict_noun_lemma_to_examples[synonym]:
                dict_noun_lemma_to_examples_new[word].append((spans[0], spans[1]))
            dict_noun_lemma_to_counter_new[word] += dict_noun_lemma_to_counter[synonym]
    dict_noun_lemma_to_counter_new = {k: v for k, v in
                                      sorted(dict_noun_lemma_to_counter_new.items(), key=lambda item: item[1])}
    dict_noun_lemma_to_span_new = {k: v for k, v in
                                   sorted(dict_noun_lemma_to_examples_new.items(), key=lambda item: len(item[1]),
                                          reverse=True)}
    dict_noun_lemma_to_examples = dict_noun_lemma_to_span_new
    dict_noun_lemma_to_counter = dict_noun_lemma_to_counter_new

# def synonyms_consolidation(dict_noun_lemma_to_span, dict_noun_lemma_to_counter, dict_noun_lemma_to_synonyms,
#                            synonyms_type='wordnet'):
#     dict_noun_lemma_to_span_new = {}
#     dict_noun_lemma_to_counter_new = {}
#     dict_noun_lemma_to_span = {k: v for k, v in
#                                sorted(dict_noun_lemma_to_span.items(), key=lambda item: len(item[1]),
#                                       reverse=True)}
#     word_lst = dict_noun_lemma_to_span.keys()
#     already_calculated = []
#     for word in word_lst:
#         if word in already_calculated:
#             continue
#         synonyms = []
#         # synonyms = dict_noun_lemma_to_synonyms[word]
#         if synonyms_type == 'wordnet':
#             for syn in wordnet.synsets(word):
#                 for lemma in syn.lemmas():
#                     synonyms.append(lemma.name())
#         else:
#             # aliases = umls_loader.umls_loader.get_term_aliases(word)
#             # for syn in aliases:
#             #     synonyms.append(syn)
#             dict_response = requests.get('http://127.0.0.1:5000/', params={"word": word})
#             synonyms = dict_response.json()["synonyms"]
#         dict_noun_lemma_to_span_new[word] = []
#         dict_noun_lemma_to_span_new[word].extend(dict_noun_lemma_to_span[word])
#         dict_noun_lemma_to_counter_new[word] = dict_noun_lemma_to_counter[word]
#         dict_noun_lemma_to_synonyms[word] = dict_noun_lemma_to_synonyms.get(word, set())
#         dict_noun_lemma_to_synonyms[word].add(word)
#         synonyms = set(synonyms)
#         if synonyms:
#             for synonym in synonyms:
#                 if synonym in already_calculated:
#                     continue
#                 if synonym != word and synonym in word_lst:
#                     for spans in dict_noun_lemma_to_span[synonym]:
#                         # new_spans_lst = []
#                         # for span in spans[1]:
#                         #     new_spans_lst.append((span[0].replace(synonym, word), span[1]))
#                         # dict_noun_lemma_to_span_new[word].append((spans[0], new_spans_lst))
#                         dict_noun_lemma_to_span_new[word].append((spans[0], spans[1]))
#                     # dict_noun_lemma_to_span_new[word].extend(dict_noun_lemma_to_span[synonym])
#                     dict_noun_lemma_to_counter_new[word] += dict_noun_lemma_to_counter[synonym]
#                     # dict_word_to_his_synonym[synonym] = word
#                     dict_noun_lemma_to_synonyms[word].add(synonym)
#                     already_calculated.append(synonym)
#         already_calculated.append(word)
#     dict_noun_lemma_to_counter_new = {k: v for k, v in
#                                       sorted(dict_noun_lemma_to_counter_new.items(), key=lambda item: item[1])}
#     dict_noun_lemma_to_span_new = {k: v for k, v in
#                                    sorted(dict_noun_lemma_to_span_new.items(), key=lambda item: len(item[1]),
#                                           reverse=True)}
#     return dict_noun_lemma_to_counter_new, dict_noun_lemma_to_span_new
