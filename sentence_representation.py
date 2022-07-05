import json
import utils as ut

attr_of_node = ['det', 'neg', 'auxpass', 'aux']

counter_error_example = 0


def from_token_lst_to_span(token_lst):
    idx = 0
    span = ""
    for token in token_lst:
        if idx != 0:
            span += " "
        span += token.text.lower()
        idx += 1
    return span


class Node:
    def __init__(self, span):
        self.type = span[1]
        # span = span[0]
        # span.sort(key=lambda x: x.i)
        self.span = span[0]
        # self.attr_head = []
        # self.bridge_to_head = ""
        # self.basic_span = ""
        # self.is_amod_type = False
        # self.basic_span_as_tokens = []
        # self.initialize_attr_and_basic_span(span)
        ########
        self.children_to_the_left = []
        self.children_to_the_right = []
        # self.kind = kind

    # def initialize_attr_and_basic_span(self, span):
    #     global counter_error_example
    #     basic_lst = []
    #     self.attr_head = []
    #     for token in span:
    #         if token.dep_ in attr_of_node:
    #             self.attr_head.append(token)
    #         else:
    #             basic_lst.append(token)
    #         # else:
    #         #     if token[1] == 4:
    #         #         self.is_amod_type = True
    #     self.basic_span = from_token_lst_to_span(basic_lst)
    #     self.basic_span_as_tokens = basic_lst
    #     # self.bridge_to_head = from_token_lst_to_span(bridge_to_head_lst)
    #     # if self.basic_span == "":
    #     #     counter_error_example += 1

    def add_children(self, child):
        if child.type in [1, 2]:
            self.children_to_the_right.append(child)
        else:
            self.children_to_the_left.append(child)
        # if child.span[-1].i < self.span[-1].i or child.is_amod_type:
        #     self.children_to_the_left.append(child)
        # else:
        #     self.children_to_the_right.append(child)


def _try(o):
    try:
        return o.__dict__
    except:
        return str(o)


class head_phrase:
    def __init__(self, node):
        self.head_phrase_name = node.basic_span
        self.type = node.type
        self.head_node_lst = []
        self.nodes_lst = []
        self.relation_to_nodes = {}
        self.nodes_lst = []
        self.from_node_to_all_his_expansion_to_the_left = {}
        self.from_node_to_sentence = {}
        self.complements_to_nodes = {}

    def add_new_node(self, node, sentence, is_head_node = False):
        if is_head_node:
            self.head_node_lst.append(node)
        self.nodes_lst.append(node)
        new_format_all_valid_sub_np = ut.get_all_options_without_shortcut(node, True)
        sub_np_final_lst_special = ut.from_lst_to_sequence_special(new_format_all_valid_sub_np, [])
        self.from_node_to_sentence[node] = sentence
        valid_expansion_results = set()
        for sub_np in sub_np_final_lst_special:
            valid_span = ut.list_of_nodes_to_span_without_shortcut(sub_np)
            valid_expansion_results.add(valid_span)
        self.from_node_to_all_his_expansion_to_the_left[node] = valid_expansion_results
        for child in node.children_to_the_right:
            if child.type == 2:
                self.relation_to_nodes[child.basic_span] = self.relation_to_nodes.get(
                    child.basic_span, [])
                self.relation_to_nodes[child.basic_span].append(child)
            else:
                self.complements_to_nodes[child.basic_span] = self.complements_to_nodes.get(
                    child.basic_span, [])
                self.complements_to_nodes[child.basic_span].append(child)

    # def toJSON(self):
    #     return json.dumps(self, default=vars,
    #                       sort_keys=True, indent=4)
    def toJSON(self):
        return json.dumps(self, default=lambda o: _try(o), sort_keys=True, indent=0, separators=(',', ':')).replace(
            '\n', '')
