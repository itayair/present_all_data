import umls_relation_loader
# import parse_medical_data
import json
from fastapi import FastAPI, APIRouter

router = APIRouter()
app = FastAPI()


@app.post("/get_broader_terms/")
def create_synonyms_dictionary(word: str):
    # dict_lemma_to_synonyms = {}
    # for word in word_lst:
    broader_term = umls_relation_loader.umls_relation_loader.get_broader_term(word)
    # dict_lemma_to_synonyms[word] = set()
    # dict_lemma_to_synonyms[word].add(word)
    # synonyms = set(synonyms)
    # if synonyms:
    #     for synonym in synonyms:
    #         if synonym != word and synonym in word_lst:
    #             dict_lemma_to_synonyms[word].add(synonym)
    return {"broader_term": broader_term}
