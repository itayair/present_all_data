class noun_cluster_object:
    def __init__(self, noun, synonyms_lst, answers_collections):
        self.noun = noun
        self.synonyms_lst = synonyms_lst
        self.answers_collections_lst = []
        for collection_data in answers_collections:
            self.answers_collections_lst.append(answers_collection_object(collection_data))


class answers_collection_object:
    def __init__(self, collection_data):
        self.longest_answer = collection_data[0]
        self.answers_lst = []
        for answer_data in collection_data[1]:
            self.answers_lst.append(answer_object(answer_data))


class answer_object:
    def __init__(self, answer_data):
        self.answer_span = answer_data[0]
        self.answer_score = answer_data[1]
        self.answer_as_lst = answer_data[2]