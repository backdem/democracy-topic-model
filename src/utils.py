import pandas as pd

def get_country_data(data_file, countries, years):
    all_countries_data = pd.read_csv(data_file, dtype={'year': str}, comment='#')
    all_countries_data['sentence'] = all_countries_data['sentence'].astype(str)
    df = pd.DataFrame(all_countries_data)
    country_data = df[(df['year'].isin(years)) & (df['country'].isin(countries))]
    # reset index; needed for proper parsing by BERT
    country_data = country_data.reset_index(drop=True)
    return country_data
                       