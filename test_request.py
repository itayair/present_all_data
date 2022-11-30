import requests
import datetime
synonyms = ["cancer", "carcinoma", "tumor"]
post_data = ",".join(synonyms)
now = datetime.datetime.now()
print("Current date and time before function: ")
print(now.strftime('%Y-%m-%d %H:%M:%S'))
dict_response = requests.post('http://127.0.0.1:5000/', params={"words": post_data})
now = datetime.datetime.now()
print("Current date and time after function: ")
print(now.strftime('%Y-%m-%d %H:%M:%S'))
print(dict_response.json()["synonyms"])
# dict_noun_lemma_to_span, dict_noun_lemma_to_counter, dict_noun_lemma_to_synonyms_new = \
#     dict_response["dict_noun_lemma_to_span"], dict_response["dict_noun_lemma_to_counter"], dict_response[
#         "dict_noun_lemma_to_synonyms"]
# print(dict_noun_lemma_to_span)
# print(dict_noun_lemma_to_counter)
# print(dict_noun_lemma_to_synonyms_new)
