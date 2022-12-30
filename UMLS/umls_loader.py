import os
import argparse
import pandas as pd
from tqdm import tqdm

# sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.pardir)))
# os.chdir(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.pardir)))

dict_id_to_similar_concepts = {}
dict_concept_to_id = {}
dict_id_to_concept = {}
# dict_from_relation_type_between_terms = {'SY': {}, 'RN': {}, 'RQ': {}, 'PAR': {}, 'AQ': {}, 'QB': {}, 'RB': {},
#                                          'RL': {}, 'CHD': {}, 'RO': {}}
dict_from_relation_type_between_terms = {}
dict_explicit_relation_type_between_terms = {}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--umls_dict_dir', type=str,
                        default='C:/Users/iy245/UMLS_ALL/umls-2022AB-metathesaurus-full/2022AB/META/MRCONSO.RRF',
                        help="the directory of the umls dictionary corpus")
    parser.add_argument('--umls_relation_dir', type=str,
                        default='C:/Users/iy245/UMLS_ALL/umls-2022AB-metathesaurus-full/2022AB/META/MRREL.RRF',
                        help="the directory of the umls relation corpus")
    return parser.parse_args()


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class UMLSLoader(metaclass=Singleton):
    def __init__(self, corpus_dir_dict, corpus_dir_relation):
        tqdm.pandas()
        self.umls_df_chunks: pd.DataFrame = self.read_df_for_dict(corpus_dir_dict)
        self.create_dicts_from_pd_table()
        self.read_df_for_relation(corpus_dir_relation)

    @staticmethod
    def read_df_for_dict(corpus_dir):
        print("start read dict table")
        col_names = ['CUI', 'LAT', 'TS', 'LUI', 'STT', 'SUI', 'ISPREF', 'AUI', 'SAUI', 'SCUI', 'SDUI', 'SAB', 'TTY',
                     'CODE', 'STR', 'SRL', 'SUPPRESS', 'CVF']
        umls_df_chunks = pd.read_table(corpus_dir, sep='|', chunksize=1000000, encoding='utf-8', index_col=False,
                                       header=None, names=col_names, low_memory=False)
        print("End read dict table")
        # umls_df['STR'] = umls_df['STR'].str.lower()
        # print("End convert str in table to lower case")
        return umls_df_chunks

    @staticmethod
    def read_df_for_relation(corpus_dir):
        print("Start read table for relation")
        col_names = ['CUI1', 'AUI1', 'STYPE1', 'REL', 'CUI2', 'AUI2', 'STYPE2', 'RELA', 'RUI', 'SRUI', 'SAB', 'SL',
                     'RG', 'DIR', 'SUPPRESS', 'CVF']
        chunks = pd.read_table(corpus_dir, sep='|', chunksize=1000000, encoding='utf-8', index_col=False, header=None,
                               names=col_names,
                               low_memory=False)
        counter = 0
        num_of_relation = 0
        all_relations = set()
        num_of_error = 0
        for chunk in chunks:
            indexed_df = chunk.reset_index()
            for index, row in indexed_df.iterrows():
                relation_type = row['REL']
                all_relations.add(relation_type)
                dict_from_relation_type_between_terms[relation_type] = dict_from_relation_type_between_terms.get(
                    relation_type, {})
                first_concept_cui = row['CUI1']
                num_of_relation += 1
                dict_from_relation_type_between_terms[relation_type][first_concept_cui] = \
                    dict_from_relation_type_between_terms[relation_type].get(first_concept_cui, set())
                dict_from_relation_type_between_terms[relation_type][first_concept_cui].add(row['CUI2'])
            counter += 1
            print(counter)
        print(num_of_relation)
        print(all_relations)

    def create_dicts_from_pd_table(self):
        counter = 0
        ENG_is_valid = False
        for chunk in self.umls_df_chunks:
            chunk_temp = chunk
            chunk_temp['STR'] = chunk_temp['STR'].str.lower()
            indexed_df = chunk_temp.reset_index()
            for index, row in indexed_df.iterrows():
                if row['LAT'] != 'ENG':
                    continue
                if not ENG_is_valid:
                    print("ENG is valid ENTRY")
                    ENG_is_valid = True
                dict_id_to_similar_concepts[row['CUI']] = dict_id_to_similar_concepts.get(row['CUI'], set())
                dict_id_to_similar_concepts[row['CUI']].add(row['STR'])
                dict_concept_to_id[row['STR']] = row['CUI']
                # dict_id_to_concept[row['CUI']] = row['STR']
            counter += 1
            print(counter)

    def get_term_aliases(self, term):
        # cuis = self.umls_df[self.umls_df['STR'] == term]['CUI'].unique()
        # return self.umls_df['STR'][self.umls_df['CUI'].isin(cuis)].unique().astype(str)
        cuis = dict_concept_to_id.get(term, None)
        if not cuis:
            return set()
        else:
            return dict_id_to_similar_concepts[cuis]

    def get_broader_term(self, term, relation_type):
        # cuis = self.umls_df[self.umls_df['STR'] == term]['CUI'].unique()
        # return self.umls_df['STR'][self.umls_df['CUI'].isin(cuis)].unique().astype(str)
        if dict_from_relation_type_between_terms.get(relation_type, None):
            relation_type_dict = dict_from_relation_type_between_terms[relation_type]
            term_cui = dict_concept_to_id.get(term, None)
            if not term_cui:
                return set()
            related_term_cui_lst = relation_type_dict.get(term_cui, None)
            if related_term_cui_lst:
                broader_terms_lst = []
                for related_term_cui in related_term_cui_lst:
                    similar_concepts = dict_id_to_similar_concepts.get(related_term_cui, None)
                    if similar_concepts:
                        broader_terms_lst.append(similar_concepts)
                if broader_terms_lst:
                    return broader_terms_lst
        return set()

    # def get_nps(self):
    #     return self.umls_dict_df['STR'].astype(str), self.umls_dict_df['CUI'].astype(str)


args = parse_args()
# cwd = os.getcwd()  # Get the current working directory (cwd)
# files = os.listdir(cwd)  # Get all the files in that directory
# print("Files in %r: %s" % (cwd, files))
# umls_loader = None
umls_loader = UMLSLoader(args.umls_dict_dir, args.umls_relation_dir)
# print(umls_loader.get_term_aliases('BMI'))
# if __name__ == '__main__':
#     args = parse_args()
#     umls_loader = UMLSLoader(args.umls_dir)
#     print(umls_loader.get_term_aliases('BMI'))
