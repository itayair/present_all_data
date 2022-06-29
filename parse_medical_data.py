import spacy
import utils as ut
import valid_deps
import valid_expansion
import valid_expansion_utils
low_val_dep = ['neg', 'nmod:poss', 'case', 'mark', 'auxpass', 'aux', 'nummod', 'quantmod', 'cop']
med_val_dep = ['nsubjpass', 'advmod', 'npadvmod', 'conj', 'poss', 'nmod:poss', 'xcomp', 'nmod:npmod', 'dobj', 'nmod', 'amod', 'nsubj', 'acl', 'relcl', 'acl:relcl', 'ccomp', 'advcl']
max_val_dep = ['compound', 'mwe', 'name']
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


def get_all_expansions_of_span_from_lst(span_lst):
    # examples_to_visualize = []
    counter = 0
    sub_np_final_lst_collection = []
    for head_word, sentence_dep_graph in span_lst:
        if counter > 200:
            break
        noun_phrase, head_word_in_np_index, boundary_np_to_the_left = valid_expansion_utils.get_np_boundary(
            head_word.i,
            sentence_dep_graph)
        if noun_phrase is None:
            continue
        if boundary_np_to_the_left > 20:
            continue
        # examples_to_visualize.append(noun_phrase)
        # all_valid_sub_np = valid_deps.get_all_valid_sub_np(noun_phrase[head_word_in_np_index])
        all_valid_sub_np = valid_expansion.get_all_valid_sub_np(noun_phrase[head_word_in_np_index], boundary_np_to_the_left)
        sub_np_final_lst = []
        sub_np_final_lst = valid_expansion_utils.from_lst_to_sequence(sub_np_final_lst, all_valid_sub_np, [])
        sub_np_final_spans = []
        for sub_np in sub_np_final_lst:
            new_sub_np = list(set(sub_np))
            new_sub_np.sort(key=lambda x: x.i)
            while new_sub_np[0].dep_ in ['case', 'mark']:
                new_sub_np.pop(0)
            val = 0
            for item in new_sub_np:
                # for word in item[0]:
                #     if word in already_counted:
                #         continue
                #     new_sub_np.append(word)
                if item == head_word:
                    val += 3
                    continue
                val_to_add = 0
                if item.dep_ in low_val_dep:
                    val_to_add = 1
                if item.dep_ in med_val_dep or item.dep_ in max_val_dep:
                    val_to_add = 2
                if item.text == '-':
                    val -= (val_to_add + 1)
                val += val_to_add
            # span = valid_expansion_utils.get_tokens_as_span(new_sub_np)
            sub_np_final_spans.append((new_sub_np, val))
        sub_np_final_spans.sort(key=lambda x: len(x[0]), reverse=True)
        sub_np_final_lst_collection.append((noun_phrase, head_word, sub_np_final_spans))
    return sub_np_final_lst_collection


def get_examples_from_special_format():
    examples = []
    # file_name = 'covid-treatments.txt'
    file_name = 'sciatica_causes_full.txt'
    output_file_name = 'output_sciatica_causes_full.txt'
    with open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            sentence, span, sent_as_doc, span_as_doc = get_sentence_ans_span_from_format(line)
            if sent_as_doc is None:
                print(sentence)
                print(span)
                continue
            head_of_span = ut.get_head_of_span(span_as_doc)
            if head_of_span is None:
                continue
            examples.append((head_of_span, sent_as_doc))
            # examples.append((sentence, span, sent_as_doc, span_as_doc, head_of_span))
    examples = get_all_expansions_of_span_from_lst(examples)
    # with open(output_file_name, 'w', encoding='utf-8') as f:
    #     for example in examples:
    #         f.write(example[1] + '\n')
    return examples
