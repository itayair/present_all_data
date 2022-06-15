import pickle
from sentence_transformers import SentenceTransformer
import torch
from torch import nn

# agglomerative clustering
from numpy import unique
import numpy as np
import json
from sklearn.cluster import AgglomerativeClustering
import utils_clustering


#     counter += 1
# print(counter)
# for dep_type in valid_deps.dep_type_in_sequencial:
#     print(dep_type)
# valid_expansion_utils.write_to_file_dict_counter(sub_np_final_lst_collection, output_file)
# # doc = nlp("This is a sentence.")
# # doc2 = nlp("My name is Itay Yair.")
# displacy.serve(examples_to_visualize, style="dep", port=5000)


def get_most_frequent_span(lst_of_spans, dict_of_span_to_counter):
    most_frequent_span_value = -1
    most_frequent_span = None
    for span in lst_of_spans:
        val = dict_of_span_to_counter.get(span, 0)
        if val > most_frequent_span_value:
            most_frequent_span_value = val
            most_frequent_span = span
    return most_frequent_span


def set_cover(universe, subsets):
    """Find a family of subsets that covers the universal set"""
    elements = set(e for s in subsets for e in s)
    # Check the subsets cover the universe
    if elements != universe:
        return None
    covered = set()
    cover = []
    # Greedily add the subsets with the most uncovered points
    while covered != elements:
        subset = max(subsets, key=lambda s: len(s - covered))
        cover.append(subset)
        covered |= subset

    return cover


def cluster_phrases_by_similarity_rate(dict_phrase_to_embeds, phrase_list, cos_sim, phrase_embs, cos_sim_rate):
    counter = 0
    already_matched = set()
    dict_clustered_spans = {}
    for span in phrase_list:
        if span in already_matched:
            counter += 1
            continue
        dict_clustered_spans[span] = [span]
        already_matched.add(span)
        for phrase, embedding in dict_phrase_to_embeds.items():
            if phrase in already_matched:
                continue
            # print("The cosine similarity between " + phrase_list[0] + " and " + phrase + "  is " + str(
            #     {cos_sim(torch.tensor(phrase_embs[0]), torch.tensor(embedding))}))
            cos_sim_examples = cos_sim(torch.tensor(dict_phrase_to_embeds[span]), torch.tensor(embedding))
            if cos_sim_examples > cos_sim_rate:
                dict_clustered_spans[span].append(phrase)
                already_matched.add(phrase)
        counter += 1
    return dict_clustered_spans


def main():
    cos_sim = nn.CosineSimilarity(dim=0)
    a_file = open("data.pkl", "rb")
    dict_of_topics = pickle.load(a_file)
    dict_of_topics = {k: v for k, v in
                      sorted(dict_of_topics.items(), key=lambda item: len(item[1]),
                             reverse=True)}
    b_file = open("span_counter.pkl", "rb")
    dict_of_span_to_counter = pickle.load(b_file)
    # c_file = open("span_to_rank.pkl", "rb")
    # dict_span_to_rank = pickle.load(c_file)

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    dict_topic_to_clustered_spans = {}
    word_to_cluster = {}
    dict_span_to_rank = {}
    for key, example_list in dict_of_topics.items():
        # model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        # output = {}
        # for phrase in phrase_list:
        #     output[phrase] = output.get(phrase, 0) + 1
        # output = {k: v for k, v in
        #           sorted(output.items(), key=lambda item: item[1],
        #                  reverse=True)}
        phrase_list = []
        dict_idx_to_all_valid_expansions = {}
        idx = 0
        for phrase in example_list:
            phrase_list.append(phrase[1][0][0])
            dict_idx_to_all_valid_expansions[idx] = []
            for span in phrase[1]:
                dict_span_to_rank[span[0]] = span[1]
                dict_idx_to_all_valid_expansions[idx].append(span[0])
            idx += 1
        label_to_cluster = {}
        if len(phrase_list) > 1:
            phrase_embs = model.encode(phrase_list)
            clustering = AgglomerativeClustering(distance_threshold=0.08, n_clusters=None, linkage="average",
                                                 affinity="cosine", compute_full_tree=True).fit(phrase_embs)
            for idx, label in enumerate(clustering.labels_):
                label_to_cluster[int(label)] = label_to_cluster.get(int(label), [])
                label_to_cluster[int(label)].extend(dict_idx_to_all_valid_expansions[idx])
            span_to_same_meaning_cluster_of_spans = {}
            for label, cluster in label_to_cluster.items():
                most_frequent_span = get_most_frequent_span(cluster, dict_of_span_to_counter)
                span_to_same_meaning_cluster_of_spans[most_frequent_span] = cluster
        else:
            span_to_same_meaning_cluster_of_spans = {phrase_list[0]: []}
            label_to_cluster[0] = dict_idx_to_all_valid_expansions[idx]
        clusters = []
        for idx, valid_expansions in label_to_cluster.items():
            valid_expansions_temp = list(set(valid_expansions))
            if len(valid_expansions) != len(valid_expansions_temp):
                print(len(valid_expansions) - len(valid_expansions_temp))
            clusters.append(valid_expansions_temp)
        # word_to_cluster[key] = span_to_same_meaning_cluster_of_spans
        word_to_cluster[key] = clusters
        # linked = utils_clustering.create_tree(clustering.children_)
        # already_counted = set()
        # arr = utils_clustering.print_dendrogram_tree_by_parenthesis(linked, len(phrase_list), phrase_list,
        #                                                             already_counted)
        # print(arr)
        # dict_phrase_to_embeds = dict(zip(phrase_list, phrase_embs))
        # cos_sim_rate = 0.95
        # decrease_rate = 0.05
        # dict_clustered_spans_last = {}
        # for phrase in phrase_list:
        #     dict_clustered_spans_last[phrase] = [phrase]
        # for i in range(0, 3):
        #     dict_clustered_spans = cluster_phrases_by_similarity_rate(dict_phrase_to_embeds,
        #                                                               list(output.keys()), cos_sim,
        #                                                               phrase_embs, cos_sim_rate)
        #     new_dict_phrase_to_embeds = {}
        #     new_output = {}
        #     for phrase, similar_phrase in dict_clustered_spans.items():
        #         num_of_items = 0
        #         sum_vec = np.zeros((len(phrase_embs[0]),), dtype=float)
        #         num_of_items += output[phrase]
        #         for clustered_phrase in similar_phrase:
        #             if phrase == clustered_phrase:
        #                 continue
        #             sum_vec += dict_phrase_to_embeds[clustered_phrase]
        #             num_of_items += output[clustered_phrase]
        #         num_of_examples_in_cluster = len(similar_phrase) + len(dict_clustered_spans_last[phrase])
        #         weighted_sum_vec = sum_vec + (dict_phrase_to_embeds[phrase] * len(dict_clustered_spans_last[phrase]))
        #         new_dict_phrase_to_embeds[phrase] = weighted_sum_vec / num_of_examples_in_cluster
        #         new_output[phrase] = num_of_items
        #     output = new_output
        #     output = {k: v for k, v in
        #               sorted(output.items(), key=lambda item: item[1],
        #                      reverse=True)}
        #     dict_clustered_spans_last = utils_clustering.combine_dicts(dict_clustered_spans_last, dict_clustered_spans)
        #     dict_phrase_to_embeds = new_dict_phrase_to_embeds
        #     # print(dict_clustered_spans_last)
        #
        #     cos_sim_rate -= decrease_rate
        # dict_clustered_spans_last = {k: v for k, v in
        #                              sorted(dict_clustered_spans_last.items(), key=lambda item: len(item[1]),
        #                                     reverse=True)}
        # dict_topic_to_clustered_spans[key] = dict_clustered_spans_last
        for key_span in span_to_same_meaning_cluster_of_spans.keys():
            print(key_span)
        break
    # print(word_to_cluster)
    # print(dict_topic_to_clustered_spans)

    # a_file.close()


main()
