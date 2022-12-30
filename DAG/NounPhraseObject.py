from combine_spans import utils as combine_spans_utils


def is_span_as_lemma_already_exist(span_as_lemmas, span_as_lemmas_lst):
    for span in span_as_lemmas_lst:
        if len(set(span)) != len(set(span_as_lemmas)):
            continue
        intersection_span = set(span).intersection(set(span_as_lemmas))
        if len(intersection_span) == len(set(span_as_lemmas)):
            return True
    return False


def nps_lst_to_string(nps_lst):
    span_lst = set()
    list_of_span_as_lemmas_lst = []
    for np in nps_lst:
        span_lst.add(np[0])
        if not is_span_as_lemma_already_exist(np[1], list_of_span_as_lemmas_lst):
            list_of_span_as_lemmas_lst.append(np[1])
    return span_lst, list_of_span_as_lemmas_lst


class NP:
    def __init__(self, np, label_lst):
        self.span_lst, self.list_of_span_as_lemmas_lst = nps_lst_to_string(np)
        self.lemma_to_occurrences_dict = {}
        self.dict_key_to_synonyms_keys = {}
        self.calculate_common_denominators_of_spans(self.list_of_span_as_lemmas_lst)
        self.common_lemmas_in_spans = []
        self.length = 0
        self.update_top_lemmas_in_spans()
        self.label_lst = set(label_lst)
        self.children = []
        self.parents = set()
        self.frequency = 0
        self.score = 0.0
        self.marginal_val = 0.0
        self.combined_nodes_lst = set()
        self.weighted_average_vector = None

    def update_top_lemmas_in_spans(self):
        lemma_to_average_occurrences_dict = {key: value / len(self.list_of_span_as_lemmas_lst)
                                             for key, value in self.lemma_to_occurrences_dict.items()}
        self.common_lemmas_in_spans = [key for key, value in lemma_to_average_occurrences_dict.items() if
                                       value > 0.5]
        self.length = len(self.common_lemmas_in_spans)

    def get_lemmas_synonyms_in_keys(self, lemma):
        lemmas_keys_lst = list(self.lemma_to_occurrences_dict.keys())
        for lemma_key in lemmas_keys_lst:
            for key in self.dict_key_to_synonyms_keys[lemma_key]:
                if lemma in combine_spans_utils.dict_lemma_to_synonyms.get(key, []) or \
                        key in combine_spans_utils.dict_lemma_to_synonyms.get(lemma, []):
                    return lemma_key
            is_similar, lemma_ref = \
                combine_spans_utils.word_contained_in_list_by_edit_distance(lemma,
                                                                            self.dict_key_to_synonyms_keys[lemma_key])
            if is_similar:
                return lemma_key
        return None

    def calculate_common_denominators_of_spans(self, span_as_lemmas_lst_to_update):
        for span_as_lst in span_as_lemmas_lst_to_update:
            already_counted = set()
            for lemma in span_as_lst:
                lemma_key = self.get_lemmas_synonyms_in_keys(lemma)
                if lemma_key in already_counted:
                    continue
                if lemma_key:
                    already_counted.add(lemma_key)
                    self.lemma_to_occurrences_dict[lemma_key] += 1
                    self.dict_key_to_synonyms_keys[lemma_key].add(lemma)
                else:
                    already_counted.add(lemma)
                    self.lemma_to_occurrences_dict[lemma] = 1
                    self.dict_key_to_synonyms_keys[lemma] = set()
                    self.dict_key_to_synonyms_keys[lemma].add(lemma)

    def add_children(self, children):
        for child in children:
            if child not in self.children:
                self.children.append(child)
        for child in children:
            self.label_lst.update(child.label_lst)

    def add_unique_lst(self, span_as_tokens_lst):
        new_span_lst = []
        for span_as_tokens in span_as_tokens_lst:
            is_already_exist = is_span_as_lemma_already_exist(span_as_tokens, self.list_of_span_as_lemmas_lst)
            # for span in self.list_of_span_as_lemmas_lst:
            #     if len(set(span)) != len(set(span_as_tokens)):
            #         continue
            #     intersection_span = set(span).intersection(set(span_as_tokens))
            #     if len(intersection_span) == len(span_as_tokens):
            #         is_already_exist = True
            #         break
            if not is_already_exist:
                new_span_lst.append(span_as_tokens)
        if new_span_lst:
            self.list_of_span_as_lemmas_lst.extend(new_span_lst)
            self.calculate_common_denominators_of_spans(new_span_lst)
            self.update_top_lemmas_in_spans()

    def update_parents_label(self, np_object, label_lst, visited):
        for parent_object in np_object.parents:
            if parent_object in visited:
                continue
            parent_object.label_lst.update(label_lst)
            visited.add(parent_object)
            self.update_parents_label(parent_object, label_lst, visited)

    def update_children_with_new_parent(self, children, previous_parent):
        for child in children:
            child.parents.remove(previous_parent)
            child.parents.add(self)

    def update_parents_with_new_node(self, parents, previous_node):
        for parent in parents:
            parent.children.remove(previous_node)
            if self not in parent.children:
                parent.children.append(self)

    def combine_nodes(self, np_object):
        self.span_lst.update(np_object.span_lst)
        self.add_unique_lst(np_object.list_of_span_as_lemmas_lst)
        # self.np.extend(np_object.np)
        self.label_lst.update(np_object.label_lst)
        self.update_parents_label(np_object, self.label_lst, set())
        self.update_parents_with_new_node(np_object.parents, np_object)
        self.update_children_with_new_parent(np_object.children, np_object)
        self.parents.update(np_object.parents)
        for child in np_object.children:
            if child not in self.children:
                self.children.append(child)
        # self.combined_nodes_lst.add(np_object)

    def __gt__(self, ob2):
        return self.marginal_val < ob2.marginal_val
