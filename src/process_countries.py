#!/usr/bin/env python
# coding: utf-8

from bertopic import BERTopic
import pandas as pd
import os
import csv
import json

# load data setof all countries, years and sources
data_file = '../data/all_countries_0.0.2.csv'
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
dict_file = '../data/dict_2.json'
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


# iterate over all countries and all year reports
for country in countries:
    for year in years:
        # choose a country and year
        df = pd.DataFrame(all_countries_data)

        country_data = df[(df['year'] == year) & (df['country'] == country)]
        # reset index; needed for proper parsing by BERT
        country_data = country_data.reset_index(drop=True)
        data_len = len(country_data)
        # only consider reports with more than 99 sentences
        if data_len < 100:
            continue
        print(f"Processing {country} and year {year}")

        # load BERT model paraphrase-MiniLM-L3-v2 (multilingual)
        # or all-MiniLM-L6-v2 (english)
        # setting min_topic_size to 7 and n_grams from 1 to 3
        # we need to explore these parameters. Other parameters:
        # https://maartengr.github.io/BERTopic/getting_started/parameter%20tuning/parametertuning.html
        # guided topic modeling:
        # https://maartengr.github.io/BERTopic/getting_started/guided/guided.html
        seed_topic_list = get_seed_lists(dictionary, 2)
        model = BERTopic(seed_topic_list=seed_topic_list, verbose=True,
                         embedding_model='all-MiniLM-L6-v2',
                         min_topic_size=5,
                         n_gram_range=(1, 2))
        # fit model to our data
        topics, _ = model.fit_transform(country_data.sentence)

        # save barchart of top 10 topics
        fig = model.visualize_barchart(top_n_topics=10)
        filename = country + "_" + year + "_barchart.png"
        fig.write_image(os.path.join("../results/", filename))

        # save hierarchy of topics
        fig = model.visualize_hierarchy(top_n_topics=30)
        filename = country + "_" + year + "_hierarchy.png"
        fig.write_image(os.path.join("../results/", filename))

        # search topics close to our categorie
        report = []
        for cat in dictionary:
            row = []
            row.append(cat["name"])
            topics = cat["words"]
            max_similarity = 0
            for topic in topics:
                similar_topics, similarities = model.find_topics(topic, top_n=1)
                # most_similar = similar_topics[0]
                if similarities[0] > max_similarity:
                    max_similarity = similarities[0]
                    most_similar = similar_topics[0]
                    best_topic = topic
            row.append(model.get_topic(most_similar))
            row.append(most_similar)
            row.append(best_topic)
            row.append(max_similarity)
            report.append(row)

        filename = country + "_" + year + "_report.csv"
        with open(os.path.join("../results/", filename), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(report)

        # save model
        model.save(f"../models/{country}_{year}.model")
