import pickle
from sentence_transformers import SentenceTransformer
import torch
from torch import nn
import numpy as np
import json
from sklearn.cluster import AgglomerativeClustering
import utils_clustering


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
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    dict_topic_to_clustered_spans = {}
    for key, phrase_list in dict_of_topics.items():
        # model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        output = {}
        for phrase in phrase_list:
            output[phrase] = output.get(phrase, 0) + 1
        output = {k: v for k, v in
                  sorted(output.items(), key=lambda item: item[1],
                         reverse=True)}
        phrase_embs = model.encode(phrase_list)
        clustering = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage='average', affinity='cosine').fit(phrase_embs)
        linked = utils_clustering.create_tree(clustering.children_)
        already_counted = set()
        arr = utils_clustering.print_dendrogram_tree_by_parenthesis(linked, len(phrase_list), phrase_list,
                                                                    already_counted)
        print(json.dumps(linked, indent=2))
        dict_phrase_to_embeds = dict(zip(phrase_list, phrase_embs))
        cos_sim_rate = 0.95
        decrease_rate = 0.05
        dict_clustered_spans_last = {}
        for phrase in phrase_list:
            dict_clustered_spans_last[phrase] = [phrase]
        for i in range(0, 3):
            dict_clustered_spans = cluster_phrases_by_similarity_rate(dict_phrase_to_embeds,
                                                                      list(output.keys()), cos_sim,
                                                                      phrase_embs, cos_sim_rate)
            new_dict_phrase_to_embeds = {}
            new_output = {}
            for phrase, similar_phrase in dict_clustered_spans.items():
                num_of_items = 0
                sum_vec = np.zeros((len(phrase_embs[0]),), dtype=float)
                num_of_items += output[phrase]
                for clustered_phrase in similar_phrase:
                    if phrase == clustered_phrase:
                        continue
                    sum_vec += dict_phrase_to_embeds[clustered_phrase]
                    num_of_items += output[clustered_phrase]
                num_of_examples_in_cluster = len(similar_phrase) + len(dict_clustered_spans_last[phrase])
                weighted_sum_vec = sum_vec + (dict_phrase_to_embeds[phrase] * len(dict_clustered_spans_last[phrase]))
                new_dict_phrase_to_embeds[phrase] = weighted_sum_vec / num_of_examples_in_cluster
                new_output[phrase] = num_of_items
            output = new_output
            output = {k: v for k, v in
                      sorted(output.items(), key=lambda item: item[1],
                             reverse=True)}
            dict_clustered_spans_last = utils_clustering.combine_dicts(dict_clustered_spans_last, dict_clustered_spans)
            dict_phrase_to_embeds = new_dict_phrase_to_embeds
            # print(dict_clustered_spans_last)

            cos_sim_rate -= decrease_rate
        dict_clustered_spans_last = {k: v for k, v in
                                     sorted(dict_clustered_spans_last.items(), key=lambda item: len(item[1]),
                                            reverse=True)}
        dict_topic_to_clustered_spans[key] = dict_clustered_spans_last
    print(dict_topic_to_clustered_spans)

    # a_file.close()


main()

# print(phrase_list[0])
# for span in similar_spans:
#     print(span)
# print("Phrase:", phrase)
# print("Embedding:", embedding)
# print("")
# tokenizer = AutoTokenizer.from_pretrained("Elron/bleurt-large-512")
# model = AutoModelForSequenceClassification.from_pretrained("Elron/bleurt-large-512")
# model.eval()
#
# references = ["hello world", "hello world"]
# candidates = ["hi universe", "bye world"]

# with torch.no_grad():
#   scores = model(**tokenizer(references, candidates, return_tensors='pt'))[0].squeeze()
# print(scores) # tensor([0.9877, 0.0475])
