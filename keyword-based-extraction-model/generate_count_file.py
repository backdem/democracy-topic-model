#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import json
import string
import re
import utils
import nltk
import numpy as np
import os
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download stopwords and initialize stemmer
nltk.download('stopwords')
nltk.download('punkt')
stopwords = set(stopwords.words('english'))
stemmer = PorterStemmer()

global_dict = None

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
    

def get_topic_name(n, dictionary=global_dict):
    return dictionary[n-1]["name"]


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
            dimension = values[2]
            stem = values[1]
            count = values[0]
            # TODO problem lies here with word plural[ism|ity]. One is
            # in the electoral whilst another is on the participatory. 
            # The compound key needs to include the dimension 
            #coumpund_key = (word, country, year, source)
            #hash_map[coumpund_key] = hash_map.get(coumpund_key, [0, values[1], values[2]])
            #hash_map[coumpund_key][0] += values[0]
            coumpund_key = (word, country, year, source, dimension)
            hash_map[coumpund_key] = hash_map.get(coumpund_key, [0, stem])
            hash_map[coumpund_key][0] += count

        # process n ngrams
        for n in range(2, 5):
            dictionary = dictionaries[n-1]
            ngrams = get_ngrams([sentence], n)
            counts = count_ngrams(ngrams, dictionary)
            for word in counts.keys():
                values = counts[word]
                dimension = values[2]
                count = values[0]
                coumpund_key = (word, country, year, source, dimension)
                hash_map[coumpund_key] = hash_map.get(coumpund_key, [0, values[1]])
                hash_map[coumpund_key][0] += count

    print("building DataFrame")
    for key in hash_map.keys():
        values = hash_map[key]
        word, country, year, source, dimension = key
        count = values[0]
        results["dictionary"].append(word)
        results["country"].append(country)
        results["year"].append(year)
        results["source"].append(source)
        results["dimension"].append(dimension)
        results["count"].append(count)
    return pd.DataFrame(results)

# Exported function: Generates a count file from the keyword dictionary.
# The generation takes a while. Writes the results as a CSV which can be used
# to bypass the regeneration.
def get_keyword_extraction_counts(corpus_file, dictionary_file, regenerate=False):
    
    result_file = "non_aggregated_counts.csv"
    if not os.path.exists(corpus_file):
        raise FileNotFoundError(f"The file at '{corpus_path}' was not found.")
    if not os.path.exists(dictionary_file):
        raise FileNotFoundError(f"The file at '{dictionary_file}' was not found.")
        
    if regenerate == False and os.path.exists(result_file):
        df = pd.read_csv(result_file)
        return df
    
    json_dict = load_json_dict(dictionary_file)
    # Set global variable needed by function get_topic_name
    global_dict = json_dict
    countries, years, all_countries_data, sources = utils.get_countries_data(corpus_file)
    df = process_corpus(all_countries_data, dict_file)    
    df.to_csv("non_aggregated_counts.csv", index=False)    
    return df
