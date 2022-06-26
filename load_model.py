import pickle
from sentence_transformers import SentenceTransformer
import torch
from torch import nn
from nltk.corpus import wordnet
from sklearn.cluster import AgglomerativeClustering
import utils_clustering
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()


# from transformers import AutoTokenizer, AutoModel


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


def get_synonyms_by_word(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.add(l.name())
    return synonyms


def from_words_to_lemma_lst(span, dict_word_to_lemma):
    lemmas_lst = []
    for word in span:
        lemma = dict_word_to_lemma.get(word, None)
        if lemma is None:
            lemma = word
        lemmas_lst.append(lemma)
    return lemmas_lst


def is_similar_meaning_between_span(span_1, span_2, dict_word_to_lemma, dict_lemma_to_synonyms):
    span_1_lemma_lst = from_words_to_lemma_lst(span_1, dict_word_to_lemma)
    span_2_lemma_lst = from_words_to_lemma_lst(span_2, dict_word_to_lemma)
    for lemma in span_1_lemma_lst:
        is_exist = False
        if lemma in span_2_lemma_lst:
            is_exist = True
        else:
            for synonym_lemma in dict_lemma_to_synonyms.get(lemma, []):
                if synonym_lemma in span_2_lemma_lst:
                    is_exist = True
                    break
        if not is_exist:
            return False
    return True


def find_similarity_in_same_length_group(lst_spans_tuple, dict_word_to_lemma, dict_lemma_to_synonyms):
    black_list = []
    dict_span_to_similar_spans = {}
    for span_tuple in lst_spans_tuple:
        if span_tuple[0] in black_list:
            continue
        dict_span_to_similar_spans[span_tuple[0]] = [span_tuple[0]]
        black_list.append(span_tuple[0])
        for span_tuple_to_compare in lst_spans_tuple:
            if span_tuple_to_compare[0] in black_list:
                continue
            is_similar = is_similar_meaning_between_span(span_tuple[1], span_tuple_to_compare[1], dict_word_to_lemma,
                                                         dict_lemma_to_synonyms)
            if is_similar:
                black_list.append(span_tuple_to_compare[0])
                dict_span_to_similar_spans[span_tuple[0]].append(span_tuple_to_compare[0])
    return dict_span_to_similar_spans


def combine_similar_spans(span_to_group_members, dict_word_to_lemma, dict_lemma_to_synonyms):
    dict_length_to_span = {}
    for span, sub_set in span_to_group_members.items():
        span_as_lst = span.replace(",", "")
        span_as_lst = span_as_lst.replace(".", "")
        span_as_lst = span_as_lst.split()
        dict_length_to_span[len(span_as_lst)] = dict_length_to_span.get(len(span_as_lst), [])
        dict_length_to_span[len(span_as_lst)].append((span, span_as_lst))
    dict_spans = {}
    for idx, spans in dict_length_to_span.items():
        if idx == 1:
            continue
        dict_spans.update(find_similarity_in_same_length_group(spans, dict_word_to_lemma, dict_lemma_to_synonyms))
    span_to_group_members_new = {}
    for span, sub_set in dict_spans.items():
        span_to_group_members_new[span] = span_to_group_members[span]
        for union_span in sub_set:
            span_to_group_members_new[span].extend(span_to_group_members[union_span])
        span_to_group_members_new[span] = list(set(span_to_group_members_new[span]))
    return span_to_group_members_new


def group_agglomerative_clustering_results(clustering, dict_idx_to_all_valid_expansions, dict_of_span_to_counter):
    label_to_cluster = {}
    dict_label_to_spans_group = {}
    for idx, label in enumerate(clustering.labels_):
        label_to_cluster[int(label)] = label_to_cluster.get(int(label), [])
        label_to_cluster[int(label)].extend(dict_idx_to_all_valid_expansions[idx])
        dict_label_to_spans_group[int(label)] = dict_label_to_spans_group.get(int(label), [])
        dict_label_to_spans_group[int(label)].append(dict_idx_to_all_valid_expansions[idx][0])
    span_to_same_meaning_cluster_of_spans = {}
    for label, cluster in label_to_cluster.items():
        most_frequent_span = get_most_frequent_span(cluster, dict_of_span_to_counter)
        span_to_same_meaning_cluster_of_spans[most_frequent_span] = cluster
    return label_to_cluster, dict_label_to_spans_group, span_to_same_meaning_cluster_of_spans


def initialize_spans_data(example_list, dict_span_to_rank):
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
    return phrase_list, dict_idx_to_all_valid_expansions, dict_span_to_rank


def from_dict_to_lst(label_to_cluster):
    clusters = []
    for idx, valid_expansions in label_to_cluster.items():
        # valid_expansions_temp = list(set(valid_expansions))
        # if len(valid_expansions) != len(valid_expansions_temp):
        #     print(len(valid_expansions) - len(valid_expansions_temp))
        # clusters.append(valid_expansions_temp)
        label_to_cluster[idx] = list(set(valid_expansions))
    return label_to_cluster


def union_groups(clusters, dict_word_to_lemma, dict_lemma_to_synonyms, dict_span_to_rank):
    span_to_group_members = {}
    # idx = 0
    for idx, valid_expansions_lst in clusters.items():
        for span in valid_expansions_lst:
            span_to_group_members[span] = span_to_group_members.get(span, [])
            span_to_group_members[span].append(idx)
        # idx += 1
    span_to_group_members = {k: v for k, v in
                             sorted(span_to_group_members.items(), key=lambda item: len(item[1]),
                                    reverse=True)}
    span_to_group_members = combine_similar_spans(span_to_group_members, dict_word_to_lemma, dict_lemma_to_synonyms)
    span_to_group_members = {k: v for k, v in
                             sorted(span_to_group_members.items(), key=lambda item: len(item[1]),
                                    reverse=True)}
    span_to_group_members_more_than_1_element = {k: v for k, v in span_to_group_members.items() if len(v) > 1}
    span_to_group_members_more_than_1_element = {k: v for k, v in
                                                 sorted(span_to_group_members_more_than_1_element.items(),
                                                        key=lambda item: dict_span_to_rank[item[0]],
                                                        reverse=True)}
    new_span_to_group_members_more_than_1_element = {}
    black_lst = []
    for span, group in span_to_group_members_more_than_1_element.items():
        num_of_new_members = 0
        valid_members = []
        for member in group:
            if member in black_lst:
                continue
            num_of_new_members += 1
            valid_members.append(member)
        if len(valid_members) > 1:
            black_lst.extend(valid_members)
            new_span_to_group_members_more_than_1_element[span] = valid_members
    return new_span_to_group_members_more_than_1_element


def build_hierarchically_groups(label_to_cluster, span_to_group_members, dict_label_to_spans_group):
    dict_sub_string_to_spans = {}
    all_group_numbers = list(range(0, len(label_to_cluster.keys())))
    already_grouped = []
    for sub_string, group_numbers in span_to_group_members.items():
        already_grouped.extend(group_numbers)
        dict_sub_string_to_spans[sub_string] = []
        for num in group_numbers:
            dict_sub_string_to_spans[sub_string].extend(dict_label_to_spans_group[num])
    res_group_numbers = [item for item in all_group_numbers if item not in already_grouped]
    for num in res_group_numbers:
        if len(dict_label_to_spans_group[num]) > 1:
            dict_sub_string_to_spans[dict_label_to_spans_group[num][0]] = dict_label_to_spans_group[num]
        else:
            dict_sub_string_to_spans[dict_label_to_spans_group[num][0]] = []
    return dict_sub_string_to_spans


def create_dicts_for_words_similarity(dict_word_to_lemma):
    dict_lemma_to_synonyms = {}
    lemma_lst = set()
    for _, lemma in dict_word_to_lemma.items():
        lemma_lst.add(lemma)
    for lemma in lemma_lst:
        synonyms = get_synonyms_by_word(lemma)
        synonyms = [synonym for synonym in synonyms if synonym in lemma_lst]
        dict_lemma_to_synonyms[lemma] = synonyms
    dict_lemma_to_synonyms = {k: v for k, v in
                              sorted(dict_lemma_to_synonyms.items(), key=lambda item: len(item[1]),
                                     reverse=True)}
    return dict_lemma_to_synonyms


def load_data_dicts():
    a_file = open("data.pkl", "rb")
    dict_of_topics = pickle.load(a_file)
    dict_of_topics = {k: v for k, v in
                      sorted(dict_of_topics.items(), key=lambda item: len(item[1]),
                             reverse=True)}
    b_file = open("span_counter.pkl", "rb")
    dict_of_span_to_counter = pickle.load(b_file)
    c_file = open("word_to_lemma.pkl", "rb")
    dict_word_to_lemma = pickle.load(c_file)
    return dict_of_topics, dict_of_span_to_counter, dict_word_to_lemma


def main():
    cos_sim = nn.CosineSimilarity(dim=0)
    dict_of_topics, dict_of_span_to_counter, dict_word_to_lemma = load_data_dicts()
    dict_lemma_to_synonyms = create_dicts_for_words_similarity(dict_word_to_lemma)
    # tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
    # model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    word_to_cluster = {}
    dict_span_to_rank = {}
    for key, example_list in dict_of_topics.items():
        phrase_list, dict_idx_to_all_valid_expansions, dict_span_to_rank = initialize_spans_data(example_list,
                                                                                                 dict_span_to_rank)
        if len(phrase_list) > 1:
            # inputs = tokenizer(phrase_list, padding=True, return_tensors="pt")
            # outputs = model(**inputs, output_hidden_states=True)
            #
            # last_hidden_states = outputs.hidden_states[-1]
            # last_hidden_states_temp = []
            # for i in range(len(phrase_list)):
            #     temp = last_hidden_states[i, 0, :]
            #     last_hidden_states_temp.append(temp)
            # last_hidden_states = torch.stack(last_hidden_states_temp)
            phrase_embs = model.encode(phrase_list)
            clustering = AgglomerativeClustering(distance_threshold=0.08, n_clusters=None, linkage="average",
                                                 affinity="cosine", compute_full_tree=True).fit(phrase_embs)
            label_to_cluster, dict_label_to_spans_group, span_to_same_meaning_cluster_of_spans = group_agglomerative_clustering_results(
                clustering, dict_idx_to_all_valid_expansions, dict_of_span_to_counter)
            dict_label_to_spans_group = {k: v for k, v in
                                         sorted(dict_label_to_spans_group.items(), key=lambda item: len(item[1]),
                                                reverse=True)}
            print(dict_label_to_spans_group)
        else:
            word_to_cluster[phrase_list[0]] = []
            continue
        label_to_cluster = from_dict_to_lst(label_to_cluster)
        span_to_group_members = union_groups(label_to_cluster, dict_word_to_lemma, dict_lemma_to_synonyms,
                                             dict_span_to_rank)
        dict_sub_string_to_spans = build_hierarchically_groups(label_to_cluster, span_to_group_members,
                                                               dict_label_to_spans_group)
        word_to_cluster[key] = dict_sub_string_to_spans
        # for key_span in span_to_same_meaning_cluster_of_spans.keys():
        #     print(key_span)
        # break
    print(word_to_cluster)
        # print(dict_topic_to_clustered_spans)

        # a_file.close()


main()
