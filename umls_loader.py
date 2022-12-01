import os
import argparse
import pandas as pd
from tqdm import tqdm

# sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.pardir)))
# os.chdir(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.pardir)))

dict_id_to_similar_concepts = {}
dict_word_to_id = {}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--umls_dir', type=str, default='corpora/umls/MRCONSO.RRF',
                        help="the directory of the umls corpus")
    return parser.parse_args()


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class UMLSLoader(metaclass=Singleton):
    def __init__(self, corpus_dir):
        tqdm.pandas()
        self.umls_df: pd.DataFrame = self.read_df(corpus_dir)
        self.create_dicts_from_pd_table()

    @staticmethod
    def read_df(corpus_dir):
        col_names = ['CUI', 'LAT', 'TS', 'LUI', 'STT', 'SUI', 'ISPREF', 'AUI', 'SAUI', 'SCUI', 'SDUI', 'SAB', 'TTY',
                     'CODE', 'STR', 'SRL', 'SUPPRESS', 'CVF']
        umls_df = pd.read_table(corpus_dir, sep='|', encoding='utf-8', index_col=False, header=None, names=col_names, low_memory=False)
        umls_df['STR'] = umls_df['STR'].str.lower()
        return umls_df

    def create_dicts_from_pd_table(self):
        indexed_df = self.umls_df.reset_index()
        for index, row in indexed_df.iterrows():
            dict_id_to_similar_concepts[row['CUI']] = dict_id_to_similar_concepts.get(row['CUI'], set())
            dict_id_to_similar_concepts[row['CUI']].add(row['STR'])
            dict_word_to_id[row['STR']] = row['CUI']

    def get_term_aliases(self, term):
        # cuis = self.umls_df[self.umls_df['STR'] == term]['CUI'].unique()
        # return self.umls_df['STR'][self.umls_df['CUI'].isin(cuis)].unique().astype(str)
        cuis = dict_word_to_id.get(term, None)
        if not cuis:
            return set()
        else:
            return dict_id_to_similar_concepts[cuis]

    def get_nps(self):
        return self.umls_df['STR'].astype(str), self.umls_df['CUI'].astype(str)

args = parse_args()
cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(cwd)  # Get all the files in that directory
print("Files in %r: %s" % (cwd, files))
# umls_loader = None
umls_loader = UMLSLoader(args.umls_dir)
# print(umls_loader.get_term_aliases('BMI'))
# if __name__ == '__main__':
#     args = parse_args()
#     umls_loader = UMLSLoader(args.umls_dir)
#     print(umls_loader.get_term_aliases('BMI'))
