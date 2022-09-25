import spacy
# import utils as ut
# import valid_deps
from expansions import valid_expansion_utils, valid_expansion

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


def get_sentence_ans_span_from_format(line):
    temp = line.replace("<e1>", "")
    sentence = temp.replace("</e1>", "")
    span = line[line.find("<e1>") + 4: line.find("</e1>")]
    sent_as_doc = nlp(sentence)
    span_as_doc = nlp(span)
    start_idx_span, end_idx_span = find_spacy_span_in_spacy_sentence(span_as_doc, sent_as_doc)
    if start_idx_span == -1:
        sentence, span, None, None
    span_as_doc = get_tokens_from_list_by_indices(start_idx_span, end_idx_span, sent_as_doc)
    return sentence, span, sent_as_doc, span_as_doc


def get_examples_from_special_format():
    examples = []
    # file_name = 'covid-treatments.txt'
    # file_name = 'text_files\\sciatica_causes_full.txt'
    file_name = 'text_files\\chest_pain_causes.txt'
    # output_file_name = 'output_sciatica_causes_full.txt'
    with open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            sentence, span, sent_as_doc, span_as_doc = get_sentence_ans_span_from_format(line)
            if sent_as_doc is None:
                print(sentence)
                print(span)
                continue
            head_of_span = valid_expansion_utils.get_head_of_span(span_as_doc)
            if head_of_span is None:
                continue
            examples.append((head_of_span, sent_as_doc, sentence))
            # examples.append((sentence, span, sent_as_doc, span_as_doc, head_of_span))
    examples = valid_expansion.get_all_expansions_of_span_from_lst(examples)
    # with open(output_file_name, 'w', encoding='utf-8') as f:
    #     for example in examples:
    #         f.write(example[1] + '\n')
    return examples
