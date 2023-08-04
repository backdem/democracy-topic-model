#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import json
import string
import re
import utils
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download stopwords and initialize stemmer
nltk.download('stopwords')
nltk.download('punkt')
stopwords = set(stopwords.words('english'))
stemmer = PorterStemmer()


def get_stemmed_tokens(data_list):
    all_stems = []
    for text in data_list:
        text = re.sub(r'\d+', '', text)
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)
        # Tokenize the sentence
        tokens = word_tokenize(text)

        # Remove stopwords and perform stemming
        filtered_tokens = [token for token in tokens if token.lower() not in stopwords and token.isalnum()]
        stems = [stemmer.stem(word) for word in filtered_tokens]
        all_stems += stems
    return all_stems


def get_stemmed_mapping(text):
    text = re.sub(r'\d+', '', text)
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    text = text.lower()
    # Tokenize the sentence
    tokens = word_tokenize(text)

    # Remove stopwords and perform stemming
    filtered_tokens = [token for token in tokens if text not in stopwords and token.isalnum()]
    stems = [stemmer.stem(word) for word in filtered_tokens]
    return (stems, filtered_tokens)


def get_stem_word_dict(data_list, stem_word_dict):
    if not stem_word_dict:
        stem_word_dict = {}
    for text in data_list:
        text = re.sub(r'\d+', '', text)
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)
        # Tokenize the text into individual words
        words = word_tokenize(text)
        # Retrieve the stem for each word
        stems = [stemmer.stem(word) for word in words]
        # Create a dictionary to associate each stem with its corresponding words
        for word, stem in zip(words, stems):
            stem_word_dict.setdefault(stem, []).append(word)

    for key in stem_word_dict.keys():
        vals = stem_word_dict[key]
        stem_word_dict[key] = list(set(vals))

    return stem_word_dict


# In[123]:


def get_words(data_list):
    all_words = []
    for text in data_list:
        text = re.sub(r'\d+', '', text)
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)
        # Tokenize the text into individual words
        words = word_tokenize(text)
        all_words += words
    return all_words


# In[124]:


def get_ngrams(data_list, n):
    words = get_words(data_list)
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram_str = ' '.join(words[i:i+n]).lower()
        ngrams.append(ngram_str)
    return ngrams


def get_dict_counts(tokens, dictionary):
    counts = np.zeros(len(dictionary))
    token_counts = [{} for i in range(len(dictionary))]
    for token in tokens:
        for i in range(len(dictionary)):
            if token in dictionary[i]:
                counts[i] += 1
                token_counts[i][token] = token_counts[i].get(token, 0) + 1
    return ([c for c in counts if len(tokens) > 0], token_counts)


def get_vocab_counts(tokens):
    vocab_counts = {}
    for token in tokens:
        if token not in vocab_counts.keys():
            vocab_counts[token] = 1
        else:
            vocab_counts[token] += 1
    return dict(sorted(vocab_counts.items(), key=lambda x: x[1], reverse=True))


def load_json_dict(dict_file):
    with open(dict_file, 'r') as file:
        dictionary = json.load(file)
        return dictionary


# load data setof all countries, years and sources
data_file = '../data/all_countries_0.0.6.csv'
dict_file = "../data/dict_6.json"
json_dict = load_json_dict(dict_file)


def get_topic_name(n, dict_file=json_dict):
    return json_dict[n-1]["name"]


def get_no_topics():
    return len(json_dict)


def count_stems(stems, words, stemmed_dictionary):
    counts = {}
    for i, stem in enumerate(stems):
        for j, dimension in enumerate(stemmed_dictionary):
            if stem in dimension:
                counts[words[i]] = counts.get(words[i], [0, stem,  get_topic_name(j+1)])
                counts[words[i]][0] += 1
    return counts


def count_ngrams(ngrams, dictionary):
    counts = {}
    for ngram in ngrams:
        for j, dimension in enumerate(dictionary):
            if ngram in dimension:
                counts[ngram] = counts.get(ngram, [0, ngram,  get_topic_name(j+1)])
                counts[ngram][0] += 1
    return counts


def process_corpus(df, dict_file):
    results = {"dictionary": [], "country": [], "year": [], "dimension": [], "source": [], "count": []}
    hash_map = {}
    dictionaries = []
    for n in range(1, 5):
        dictionary, topics = utils.get_seed_lists(dict_file, n, exact=True)
        dictionaries.append(dictionary)

    for index, row in df.iterrows():
        sentence = row['sentence']
        country = row['country']
        year = row['year']
        source = row['source']

        # process 1 ngram
        stemmed_dictionary = [list(get_stemmed_tokens(l)) for l in dictionaries[0]]
        stems, words = get_stemmed_mapping(sentence)
        counts = count_stems(stems, words, stemmed_dictionary)
        for word in counts.keys():
            values = counts[word]
            coumpund_key = (word, country, year, source)
            hash_map[coumpund_key] = hash_map.get(coumpund_key, [0, values[1], values[2]])
            hash_map[coumpund_key][0] += values[0]

        # process n ngrams
        for n in range(2, 5):
            dictionary = dictionaries[n-1]
            ngrams = get_ngrams([sentence], n)
            counts = count_ngrams(ngrams, dictionary)
            for word in counts.keys():
                values = counts[word]
                coumpund_key = (word, country, year, source)
                hash_map[coumpund_key] = hash_map.get(coumpund_key, [0, values[1], values[2]])
                hash_map[coumpund_key][0] += values[0]

    print("building DataFrame")
    for key in hash_map.keys():
        values = hash_map[key]
        word, country, year, source = key
        count = values[0]
        dimension = values[2]
        results["dictionary"].append(word)
        results["country"].append(country)
        results["year"].append(year)
        results["source"].append(source)
        results["dimension"].append(dimension)
        results["count"].append(count)
    return pd.DataFrame(results)


countries, years, all_countries_data, sources = utils.get_countries_data(data_file)
df = process_corpus(all_countries_data, dict_file)
print("writing csv file")
df.to_csv("non_aggregated_counts.csv", index=False)
print("done")
