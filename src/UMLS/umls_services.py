import umls_loader
# import parse_medical_data
import json
from fastapi import FastAPI, APIRouter

router = APIRouter()
app = FastAPI()


# class Item(BaseModel):
#     word_lst: list


# @app.post("/")
# def root(words: str):
#     word_lst = words.split(',')
#     return {"synonyms": word_lst}


@app.post("/create_noun_synonyms_dictionary/")
def create_noun_synonyms_dictionary(words: str):
    word_lst = json.loads(words)
    already_calculated = []
    dict_noun_lemma_to_synonyms = {}
    for word in word_lst:
        if word in already_calculated:
            continue
        synonyms = umls_loader.umls_loader.get_term_aliases(word)
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


@app.post("/create_synonyms_dictionary/")
def create_synonyms_dictionary(words: str):
    word_lst = json.loads(words)
    dict_lemma_to_synonyms = {}
    for word in word_lst:
        synonyms = umls_loader.umls_loader.get_term_aliases(word)
        dict_lemma_to_synonyms[word] = set()
        dict_lemma_to_synonyms[word].add(word)
        synonyms = set(synonyms)
        if synonyms:
            for synonym in synonyms:
                if synonym != word:
                    dict_lemma_to_synonyms[word].add(synonym)
    return {"synonyms": dict_lemma_to_synonyms}



@app.post("/create_abbreviation_dict/")
def create_synonyms_dictionary(abbreviations: str, compound_lst: str):
    abbreviations = json.loads(abbreviations)
    compound_lst = json.loads(compound_lst)
    dict_abbreviation_to_compound = {}
    for abbreviation in abbreviations:
        synonyms = umls_loader.umls_loader.get_term_aliases(abbreviation)
        if synonyms:
            for synonym in synonyms:
                if synonym != abbreviation and synonym in compound_lst:
                    dict_abbreviation_to_compound[abbreviation].add(synonym)
    return {"dict_abbreviation_to_compound": dict_abbreviation_to_compound}


@app.post("/get_broader_terms/")
def create_synonyms_dictionary(word: str, relation_type: str):
    # dict_lemma_to_synonyms = {}
    # for word in word_lst:
    broader_term = umls_loader.umls_loader.get_broader_term(word, relation_type)
    # dict_lemma_to_synonyms[word] = set()
    # dict_lemma_to_synonyms[word].add(word)
    # synonyms = set(synonyms)
    # if synonyms:
    #     for synonym in synonyms:
    #         if synonym != word and synonym in word_lst:
    #             dict_lemma_to_synonyms[word].add(synonym)
    return {"broader_term": broader_term}


@app.post("/get_broader_atoms/")
def create_synonyms_dictionary(word: str, relation_type: str):
    # dict_lemma_to_synonyms = {}
    # for word in word_lst:
    broader_atom = umls_loader.umls_loader.get_broader_atom(word, relation_type)
    # dict_lemma_to_synonyms[word] = set()
    # dict_lemma_to_synonyms[word].add(word)
    # synonyms = set(synonyms)
    # if synonyms:
    #     for synonym in synonyms:
    #         if synonym != word and synonym in word_lst:
    #             dict_lemma_to_synonyms[word].add(synonym)
    return {"broader_atom": broader_atom}


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
