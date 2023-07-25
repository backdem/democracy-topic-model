import pandas as pd
import json
import nltk
import spacy

# Load the English language model
nlp = spacy.load('en_core_web_sm')

nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

def combine_dicts_arrays(one, two, unique=False):
    for key in two.keys():
        one.setdefault(key, [])
        one[key] += two[key]
        if unique:
            one[key] = list(set(one[key]))
    return one

def get_part_of_speech(text):
    ner = {
        'PERSON': [],
        'ORGANIZATION': [],
        'LOCATION':[]
    }
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    ner_tags = nltk.ne_chunk(pos_tags)
    
    for entity in ner_tags.subtrees():
        if entity.label() in ['PERSON', 'ORGANIZATION', 'LOCATION']:
            ner[entity.label()].append(' '.join([leaf[0] for leaf in entity.leaves()]))
    return (pos_tags, ner)

def get_country_data(data_file, countries, years):
    all_countries_data = pd.read_csv(data_file, dtype={'year': str}, comment='#')
    all_countries_data['sentence'] = all_countries_data['sentence'].astype(str)
    df = pd.DataFrame(all_countries_data)
    country_data = df[(df['year'].isin(years)) & (df['country'].isin(countries))]
    # reset index; needed for proper parsing by BERT
    country_data = country_data.reset_index(drop=True)
    return country_data

def add_pos_column(country_data):
    country_data["sentence_pos"] = country_data["sentence"].apply(lambda x: len(x.split()))

    return country_data
    
def get_seed_lists(data_file, ngram_size, exact=False):
    dictionary = None
    topics = []
    with open(data_file, 'r') as file:
        dictionary = json.load(file)
        seeds = []
        for topic in dictionary:
            if exact:
                seed = [w for w in topic["words"] if len(w.split()) == ngram_size]
            else:
                seed = [w for w in topic["words"] if len(w.split()) <= ngram_size]
            seeds.append(seed)
            topics.append(topic["name"])
        return (seeds, topics)
    

def get_countries_data(csv_file):
    all_countries_data = pd.read_csv(csv_file, dtype={'year': str}, comment='#')
    # cast sentence column to string
    all_countries_data['sentence'] = all_countries_data['sentence'].astype(str)
    all_countries_data['country'] = all_countries_data['country'].astype(str)
    
    # Done in generate_dataset in scrape_tools
    #country_alt_names = {
    #        "türkiye": "turkey",
    #        "north_macedonia": "north-macedonia",
    #        "bosnia-and-herzegovina": "bosnia-herzegovina",
    #        "czech-republic": "czechia"
    #        }

    # fix alternate name for countries
    #for name in list(country_alt_names.keys()):
    #    all_countries_data["country"] = all_countries_data["country"]\
    #            .replace(name, country_alt_names[name])

    countries = pd.Series((all_countries_data['country']))
    # remove nan from dataset
    countries = set(list(countries.dropna()))
    years = pd.Series(all_countries_data['year'])
    sources = pd.Series(all_countries_data['source'])
    # remove nan from dataset
    years = set(list(years.dropna()))
    sources = set(list(sources.dropna()))
    
    return (countries, years, pd.DataFrame(all_countries_data), sources)


