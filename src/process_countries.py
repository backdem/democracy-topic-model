#!/usr/bin/env python
# coding: utf-8

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
import pandas as pd
import os
import csv
import json
import argparse
from datetime import datetime

parser = argparse.ArgumentParser(description='Process all country reports and generate reports.')
parser.add_argument('-o', '--outputfolder', nargs='?', default=None,
                    help='output folder.')
parser.add_argument('-s', '--dataset', nargs='?', default='../data/all_countries_0.0.2.csv',
                    help='dataset to use.')
parser.add_argument('-d', '--dictionary', nargs='?', default='../data/dict_2.json',
                    help='seed dictionary to use.')
parser.add_argument('-c', '--country', nargs='?', default=None,
                    help='select single country to process.')
parser.add_argument('-y', '--year', nargs='?', default=None,
                    help='select single year to process.')
parser.add_argument('-n', '--ngram', nargs='?', default=2,
                    help='max ngrams to cluster.')
args = parser.parse_args()

# load data setof all countries, years and sources
data_file = args.dataset
dict_file = args.dictionary
root_folder = args.outputfolder

if not os.path.exists(data_file):
    print(data_file)
    raise SystemExit("--dataset parameter file not found.")
if not os.path.exists(dict_file):
    raise SystemExit("--dictionary parameter file not found.")
if not root_folder or not os.path.exists(root_folder):
    raise SystemExit("--outputfolder folder does not exist")

# create output folders
# Get the current timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create the execution folder name with the timestamp
folder_name = f"run_{timestamp}"

run_folder = os.path.join(root_folder, folder_name)
os.mkdir(run_folder)
model_folder = os.path.join(run_folder, "models")
os.mkdir(model_folder)
image_folder = os.path.join(run_folder, "images")
os.mkdir(image_folder)
sentence_folder = os.path.join(run_folder, "sentences")
os.mkdir(sentence_folder)
topic_folder = os.path.join(run_folder, "topics")
os.mkdir(topic_folder)
report_folder = os.path.join(run_folder, "reports")
os.mkdir(report_folder)


all_countries_data = pd.read_csv(data_file, dtype={'year': str}, comment='#')
# cast sentence column to string
all_countries_data['sentence'] = all_countries_data['sentence'].astype(str)
all_countries_data['country'] = all_countries_data['country'].astype(str)
country_alt_names = {
        "türkiye": "turkey",
        "north-macedonia": "north_macedonia",
        "bosnia-and-herzegovina": "bosnia-herzegovina",
        "czech-republic": "czechia"
        }

# fix alternate name for countries
for name in list(country_alt_names.keys()):
    all_countries_data["country"] = all_countries_data["country"]\
            .replace(name, country_alt_names[name])

countries = pd.Series((all_countries_data['country']))
# remove nan from dataset
countries = set(list(countries.dropna()))
years = pd.Series(all_countries_data['year'])
# remove nan from dataset
years = set(list(years.dropna()))

# load topic seeds
dictionary = None
with open(dict_file, 'r') as file:
    dictionary = json.load(file)


def get_seed_lists(dictionary, ngram_size):
    # create list of topics wit max ngram_size
    seeds = []
    for topic in dictionary:
        seed = [w for w in topic["words"] if len(w.split()) <= ngram_size]
        seeds.append(seed)
    return seeds

# check for cli country/year parameter to bypass all countries
if args.country and args.country in countries:
    print(f"Only processing country {args.country}")
    countries = [args.country]
if args.year and args.year in years:
    print(f"Only processing year {args.year}")
    years = [args.year]


# iterate over all countries and all year reports
for country in countries:
    for year in years:
        # choose a country and year
        df = pd.DataFrame(all_countries_data)

        country_data = df[(df['year'] == year) & (df['country'] == country)]
        # reset index; needed for proper parsing by BERT
        country_data = country_data.reset_index(drop=True)
        corpus_size = len(country_data)
        # only consider reports with more than 99 sentences
        if corpus_size < 100:
            continue
        print(f"Processing {country} and year {year}")

        # load BERT model paraphrase-MiniLM-L3-v2 (multilingual)
        # or all-MiniLM-L6-v2 (english)
        # setting min_topic_size to 7 and n_grams from 1 to 3
        # we need to explore these parameters. Other parameters:
        # https://maartengr.github.io/BERTopic/getting_started/parameter%20tuning/parametertuning.html
        # guided topic modeling:
        # https://maartengr.github.io/BERTopic/getting_started/guided/guided.html
        representation_model = KeyBERTInspired()
        seed_topic_list = get_seed_lists(dictionary, args.ngram)
        model = BERTopic(representation_model=representation_model,
                         seed_topic_list=seed_topic_list, verbose=True,
                         embedding_model='all-MiniLM-L6-v2',
                         min_topic_size=5,
                         n_gram_range=(1, args.ngram))
        # fit model to our data
        topics, _ = model.fit_transform(country_data.sentence)

        # save barchart of top 10 topics
        fig = model.visualize_barchart(top_n_topics=10)
        filename = country + "_" + year + "_barchart.png"
        fig.write_image(os.path.join(image_folder, filename))

        # save hierarchy of topics
        fig = model.visualize_hierarchy(top_n_topics=30)
        filename = country + "_" + year + "_hierarchy.png"
        fig.write_image(os.path.join(image_folder, filename))

        # search topics close to our categorie
        report = []
        for cat in dictionary:
            row = []
            
            topics = cat["words"]
            max_similarity = 0
            for topic in topics:
                similar_topics, similarities = model.find_topics(topic, top_n=1)
                # most_similar = similar_topics[0]
                if similarities[0] > max_similarity:
                    max_similarity = similarities[0]
                    most_similar = similar_topics[0]
                    best_topic = topic
                    
            info = model.get_topic_info(most_similar)
            normalized_count = info["Count"][0] / corpus_size
            row.append(cat["name"])                    
            row.append(most_similar)
            row.append(best_topic)
            row.append(max_similarity)
            row.append(normalized_count)
            report.append(row)
            
        # save the info about matching topics to dimensions
        filename = country + "_" + year + "_report.csv"
        with open(os.path.join(report_folder, filename), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(report)
            
        # save sentence with topic mapping
        # get document info
        doc_info = model.get_document_info(country_data.sentence)
        # write csv
        file_name = country + "_" + year + "_sentences.csv"
        with open(os.path.join(sentence_folder, file_name), mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Sentence", "Topic_Name", "Topic_No", "Probability"])
            for document, topic, name, top_n_words, prob in zip(doc_info["Document"], doc_info["Topic"], doc_info["Name"], doc_info["Top_n_words"], doc_info["Probability"]):
                writer.writerow([document, name, topic, prob])
                
        # save topic information
        info = model.get_topic_info()
        file_name = country + "_" + year + "_topics.csv"
        with open(os.path.join(topic_folder, file_name), mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Topic_No", "Topic_Name", "Count", "Topic_Words"])
            for row in zip(info["Topic"], info["Name"], info["Count"]):
                if row[0] == -1:
                    continue
                row = row + (model.get_topic(row[0]),)
                writer.writerow(row)

        # save model
        file_name = country + "_" + year + "_model"
        model.save(os.path.join(model_folder, file_name))
