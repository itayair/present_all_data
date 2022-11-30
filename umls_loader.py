import os
import argparse
import pandas as pd
from tqdm import tqdm

# sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.pardir)))
# os.chdir(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.pardir)))


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

    @staticmethod
    def read_df(corpus_dir):
        col_names = ['CUI', 'LAT', 'TS', 'LUI', 'STT', 'SUI', 'ISPREF', 'AUI', 'SAUI', 'SCUI', 'SDUI', 'SAB', 'TTY',
                     'CODE', 'STR', 'SRL', 'SUPPRESS', 'CVF']
        umls_df = pd.read_table(corpus_dir, sep='|', encoding='utf-8', index_col=False, header=None, names=col_names, low_memory=False)
        umls_df['STR'] = umls_df['STR'].str.lower()
        return umls_df

    def get_term_aliases(self, term):
        cuis = self.umls_df[self.umls_df['STR'] == term]['CUI'].unique()
        return self.umls_df['STR'][self.umls_df['CUI'].isin(cuis)].unique().astype(str)

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
