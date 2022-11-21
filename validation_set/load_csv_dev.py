import csv
import pandas as pd


def extract_clustered_data_second_stage_from_csv_file():
    clustered_by_noun_file = open('../csv/chest_pain_data_clustered_stage_2.csv', encoding="utf8")
    csv_reader_clustered_by_noun_file = csv.reader(clustered_by_noun_file)
    header = next(csv_reader_clustered_by_noun_file)
    dict_number_to_noun = {}
    for idx, noun in enumerate(header):
        dict_number_to_noun[idx] = noun


def extract_clustered_data_first_stage_from_csv_file():
    # clustered_by_noun_file = open('./csv/chest_pain_data_clustered.csv', encoding="utf8")
    # csv_reader_clustered_by_noun_file = csv.reader(clustered_by_noun_file)
    # header = next(csv_reader_clustered_by_noun_file)
    # dict_number_to_noun = {}
    # for idx, noun in enumerate(header):
    #     dict_number_to_noun[idx] = noun
    # for row in csv_reader_clustered_by_noun_file:
    # df = pd.read_csv('./csv/chest_pain_data_clustered.csv', header=None, index_col=0, squeeze=True)
    # d = df.to_dict()
    # print(d)
    dict_noun_to_examples = {}
    with open('../csv/chest_pain_data_clustered.csv', encoding="utf8") as f:
        records = csv.DictReader(f)
        for row in records:
            for key in row.keys():
                if row[key].strip() == "":
                    continue
                dict_noun_to_examples[key.strip()] = dict_noun_to_examples.get(key.strip(), [])
                dict_noun_to_examples[key.strip()].append(row[key])
    print(dict_noun_to_examples)


extract_clustered_data_first_stage_from_csv_file()
