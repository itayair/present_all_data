import os
import argparse
import pandas as pd
from tqdm import tqdm

# sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.pardir)))
# os.chdir(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.pardir)))

dict_id_to_term = {}
dict_term_to_id = {}
dict_term_to_isa_term = {}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--umls_dir', type=str,
                        default='C:/Users/iy245/UMLS_ALL/umls-2022AB-metathesaurus-full/2022AB/META/MRREL.RRF',
                        help="the directory of the umls corpus")
    return parser.parse_args()


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class UMLSRelationLoader(metaclass=Singleton):
    def __init__(self, corpus_dir):
        tqdm.pandas()
        self.read_df(corpus_dir)
        # self.create_dicts_from_pd_table()

    @staticmethod
    def read_df(corpus_dir):
        print("Start read table")
        col_names = ['CUI1', 'AUI1', 'STYPE1', 'REL', 'CUI2', 'AUI2', 'STYPE2', 'RELA', 'RUI', 'SRUI', 'SAB', 'SL',
                     'RG', 'DIR', 'SUPPRESS', 'CVF']
        chunks = pd.read_table(corpus_dir, sep='|', chunksize=1000000, encoding='utf-8', index_col=False, header=None, names=col_names,
                                low_memory=False)
        # print("Finish read table")
        # counter = 0
        # for chunk in chunks:
        #     chunk['STYPE1'] = chunk['STYPE1'].str.lower()
        #     chunk['STYPE2'] = chunk['STYPE2'].str.lower()
        #     counter += 1
        # print("num of chunks is ",counter)
        # print("Finish convert to lower case")
        other_relations = {}
        counter = 0
        for chunk in chunks:
            indexed_df = chunk.reset_index()
            idx = 0
            already_printed = False
            for index, row in indexed_df.iterrows():
                rela = row['RELA']
                other_relations[rela] = other_relations.get(rela, 0)
                other_relations[rela] += 1
                if row['RELA'] == 'isa':
                    type_1 = row['CUI1']
                    dict_term_to_isa_term[type_1] = dict_term_to_isa_term.get(type_1, set())
                    dict_term_to_isa_term[type_1].add(row['STYPE2'])
            counter += 1
            print(counter)
        print(other_relations)
        # df = pd.DataFrame()
        # % time
        # df = pd.concat(
        #     chunk.groupby(['lat', 'long', chunk['date'].map(lambda x: x.year)])['rf'].agg(['sum']) for chunk in chunks)
        # return umls_df

    # def create_dicts_from_pd_table(self):
    #     indexed_df = self.umls_df.reset_index()
    #     other_relations = set()
    #     counter = 0
    #     for index, row in indexed_df.iterrows():
    #         if counter %1000000 == 0:
    #             print(counter)
    #         counter += 1
    #         other_relations.add(row['REL'])
    #         if row['REL'] == 'isa':
    #             type_1 = row['STYPE1']
    #             dict_term_to_isa_term[type_1] = dict_term_to_isa_term.get(type_1, set())
    #             dict_term_to_isa_term[type_1].add(row['STYPE2'])
            # dict_id_to_term[row['CUI1']] = dict_id_to_term.get(row['CUI1'], set())
            # dict_id_to_term[row['CUI1']].add(row['STR'])
            # dict_term_to_id[row['STR']] = row['CUI']

    def get_broader_term(self, term):
        # cuis = self.umls_df[self.umls_df['STR'] == term]['CUI'].unique()
        # return self.umls_df['STR'][self.umls_df['CUI'].isin(cuis)].unique().astype(str)
        broader_term = dict_term_to_isa_term.get(term, None)
        if not broader_term:
            return set()
        else:
            return broader_term

    # def get_nps(self):
    #     return self.umls_df['STR'].astype(str), self.umls_df['CUI'].astype(str)


args = parse_args()
cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(cwd)  # Get all the files in that directory
print("Files in %r: %s" % (cwd, files))
# umls_loader = None
umls_relation_loader = UMLSRelationLoader(args.umls_dir)
# print(umls_loader.get_term_aliases('BMI'))
# if __name__ == '__main__':
#     args = parse_args()
#     umls_loader = UMLSLoader(args.umls_dir)
#     print(umls_loader.get_term_aliases('BMI'))
