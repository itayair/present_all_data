def nps_lst_to_string(nps_lst):
    new_nps_str = set()
    np_lst = []
    for np in nps_lst:
        new_nps_str.add(np[0])
        np_lst.append(np[1])
    return new_nps_str, np_lst


class NP:
    def __init__(self, np, label_lst):
        self.np_val, self.np = nps_lst_to_string(np)
        self.label_lst = set(label_lst)
        self.children = []
        self.parents = set()
        self.frequency = 0
        self.score = 0.0
        self.marginal_val = 0.0
        self.combined_nodes_lst = set()
        self.weighted_average_vector = None

    def add_children(self, children):
        for child in children:
            if child not in self.children:
                self.children.append(child)
        for child in children:
            self.label_lst.update(child.label_lst)

    def add_unique_lst(self, span_as_tokens_lst):
        new_np_lst = []
        for span_as_tokens in span_as_tokens_lst:
            is_already_exist = False
            for np in self.np:
                intersection_np = set(np).intersection(set(span_as_tokens))
                if len(intersection_np) == len(span_as_tokens):
                    is_already_exist = True
                    break
            if not is_already_exist:
                new_np_lst.append(span_as_tokens)
        self.np.extend(new_np_lst)

    def update_parents_label(self, np_object, label_lst, visited):
        for parent_object in np_object.parents:
            if parent_object in visited:
                continue
            parent_object.label_lst.update(label_lst)
            visited.add(parent_object)
            self.update_parents_label(parent_object, label_lst, visited)

    def update_children_with_new_parent(self, children, previous_parent):
        for child in children:
            if previous_parent not in child.parents:
                continue
            child.parents.remove(previous_parent)
            child.parents.add(self)

    def update_parents_with_new_node(self, parents, previous_node):
        for parent in parents:
            if self not in parent.children:
                parent.children.append(self)
            parent.children.remove(previous_node)

    def combine_nodes(self, np_object):
        self.np_val.update(np_object.np_val)
        self.add_unique_lst(np_object.np)
        # self.np.extend(np_object.np)
        self.label_lst.update(np_object.label_lst)
        self.update_parents_label(np_object, self.label_lst, set())
        self.update_parents_with_new_node(np_object.parents, np_object)
        self.update_children_with_new_parent(np_object.children, np_object)
        self.parents.update(np_object.parents)
        for child in np_object.children:
            if child not in self.children:
                self.children.append(child)
        self.combined_nodes_lst.add(np_object)

    def __gt__(self, ob2):
        return self.marginal_val < ob2.marginal_val