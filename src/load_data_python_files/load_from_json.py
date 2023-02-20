import json

import spacy

nlp = spacy.load("en_ud_model_sm")


def find_sub_list(sub_lst, lst):
    sll = len(sub_lst)
    for ind in (i for i, e in enumerate(lst) if e == sub_lst[0]):
        if lst[ind:ind + sll] == sub_lst:
            return ind, ind + sll - 1
    return -1, -1


def from_tokens_to_string_list(spacy_doc):
    tokens_lst = []
    for token in spacy_doc:
        tokens_lst.append(token.text)
    return tokens_lst


def get_tokens_from_list_by_indices(start_idx_span, end_idx_span, spacy_doc):
    tokens_lst = []
    for token in spacy_doc:
        tokens_lst.append(token)
    return tokens_lst[start_idx_span:end_idx_span + 1]


def find_spacy_span_in_spacy_sentence(span, sentence):
    span_lst = from_tokens_to_string_list(span)
    sent_lst = from_tokens_to_string_list(sentence)
    return find_sub_list(span_lst, sent_lst)


def get_sentence_ans_span_from_format(example):
    sentence = example[0]
    span = example[1]
    sent_as_doc = nlp(sentence)
    span_as_doc = nlp(span)
    start_idx_span, end_idx_span = find_spacy_span_in_spacy_sentence(span_as_doc, sent_as_doc)
    if start_idx_span == -1:
        sentence, span, None, None
    span_as_doc = get_tokens_from_list_by_indices(start_idx_span, end_idx_span, sent_as_doc)
    return sentence, span, sent_as_doc, span_as_doc


def get_tokens_as_span(words):
    span = ""
    idx = 0
    for word in words:
        # if idx == 0 and token.tag_ in ['IN', 'TO']:
        #     continue
        if idx != 0 and (word != ',' and word != '.'):
            span += ' '
        span += word
        idx += 1
    return span


file_name = '../../input_files/input_json_files/abortion.json'
f = open(file_name, 'r', encoding='utf-8')
data = json.load(f)
examples = []
for type in data:
    for mention in type['mentions']:
        span_as_lst = mention['words'][mention['offsets']['first']: mention['offsets']['last'] + 1]
        examples.append((get_tokens_as_span(mention['words']), get_tokens_as_span(span_as_lst)))
for example in examples:
    get_sentence_ans_span_from_format(example)
f.close()
