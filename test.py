import umls_loader
# import parse_medical_data
import json
from fastapi import FastAPI, APIRouter
from pydantic import BaseModel

router = APIRouter()
app = FastAPI()
# class Item(BaseModel):
#     word_lst: list


# @app.post("/")
# def root(words: str):
#     word_lst = words.split(',')
#     return {"synonyms": word_lst}



@app.post("/")
def root(words: str):
    word_lst = words.split(',')
    already_calculated = []
    dict_noun_lemma_to_synonyms = {}
    for word in word_lst:
        if word in already_calculated:
            continue
        aliases = umls_loader.umls_loader.get_term_aliases(word)
        synonyms = set()
        for syn in aliases:
            synonyms.add(syn)
            # dict_response = requests.get('http://127.0.0.1:5000/', params={"word": word})
            # synonyms = dict_response.json()["synonyms"]
        dict_noun_lemma_to_synonyms[word] = set()
        dict_noun_lemma_to_synonyms[word].add(word)
        synonyms = set(synonyms)
        if synonyms:
            for synonym in synonyms:
                if synonym in already_calculated:
                    continue
                if synonym != word and synonym in word_lst:
                    dict_noun_lemma_to_synonyms[word].add(synonym)
                    already_calculated.append(synonym)
        already_calculated.append(word)
    return {"synonyms": dict_noun_lemma_to_synonyms}


# @app.get("/")
# def root(word: str = "Disc"):
#     aliases = umls_loader.umls_loader.get_term_aliases(word)
#     synonyms = set()
#     for syn in aliases:
#         synonyms.add(syn)
#     return {"synonyms": synonyms}

# @app.get("/synonyms_dict/")
# def synonyms_consolidation(item: Item):
#     # dict_noun_lemma_to_span_new = {}
#     # dict_noun_lemma_to_counter_new = {}
#     # dict_noun_lemma_to_synonyms = {}
#     # already_calculated = []
#     # for word in word_lst:
#     #     if word in already_calculated:
#     #         continue
#     #     synonyms = set()
#     #     aliases = umls_loader.umls_loader.get_term_aliases(word)
#     #     for syn in aliases:
#     #         synonyms.add(syn)
#     #     dict_noun_lemma_to_span_new[word] = []
#     #     dict_noun_lemma_to_span_new[word].extend(dict_noun_lemma_to_span[word])
#     #     dict_noun_lemma_to_counter_new[word] = dict_noun_lemma_to_counter[word]
#     #     dict_noun_lemma_to_synonyms[word] = dict_noun_lemma_to_synonyms.get(word, set())
#     #     dict_noun_lemma_to_synonyms[word].add(word)
#     #     if synonyms:
#     #         for synonym in synonyms:
#     #             if synonym in already_calculated:
#     #                 continue
#     #             if synonym != word and synonym in word_lst:
#     #                 for spans in dict_noun_lemma_to_span[synonym]:
#     #                     dict_noun_lemma_to_span_new[word].append((spans[0], spans[1]))
#     #                 dict_noun_lemma_to_counter_new[word] += dict_noun_lemma_to_counter[synonym]
#     #                 dict_noun_lemma_to_synonyms[word].add(synonym)
#     #                 already_calculated.append(synonym)
#     #     already_calculated.append(word)
#     # return dict_noun_lemma_to_counter_new, dict_noun_lemma_to_span_new
#     return item.dict_noun_lemma_to_span, item.dict_noun_lemma_to_counter, item.word_lst
