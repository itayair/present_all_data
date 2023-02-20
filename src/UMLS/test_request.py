import requests
import datetime
import json

# relation_types = ['SY', 'RN', 'RQ', 'PAR', 'AQ', 'QB', 'RB', 'RL', 'CHD', 'RO']
relation_types = ['RB']
synonyms = ["cancer", "carcinoma", "tumor"]
post_data = json.dumps(synonyms)
word = 'equine'
for relation_type in relation_types:
    dict_response = requests.post('http://127.0.0.1:5000/get_broader_terms/',
                                  params={"word": word, "relation_type": relation_type})
    # dict_response = requests.post('http://127.0.0.1:5000/create_synonyms_dictionary/', params={"words": post_data})

    output = dict_response.json()
    if output:
        print(relation_type + ": ", output)
# dict_noun_lemma_to_span, dict_noun_lemma_to_counter, dict_noun_lemma_to_synonyms_new = \
#     dict_response["dict_noun_lemma_to_span"], dict_response["dict_noun_lemma_to_counter"], dict_response[
#         "dict_noun_lemma_to_synonyms"]
# print(dict_noun_lemma_to_span)
# print(dict_noun_lemma_to_counter)
# print(dict_noun_lemma_to_synonyms_new)
