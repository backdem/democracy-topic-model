import pandas as pd
import json

def get_country_data(data_file, countries, years):
    all_countries_data = pd.read_csv(data_file, dtype={'year': str}, comment='#')
    all_countries_data['sentence'] = all_countries_data['sentence'].astype(str)
    df = pd.DataFrame(all_countries_data)
    country_data = df[(df['year'].isin(years)) & (df['country'].isin(countries))]
    # reset index; needed for proper parsing by BERT
    country_data = country_data.reset_index(drop=True)
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
    
    return (countries, years, pd.DataFrame(all_countries_data))


