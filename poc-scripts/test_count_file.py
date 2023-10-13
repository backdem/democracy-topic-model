import pandas as pd
from nltk.stem import PorterStemmer #PorterStemmer
stemmer = PorterStemmer()

non_aggregated_counts = 'non_aggregated_counts.csv'
dims_all_years_all_countries = 'dims_all_years_all_countries.csv'

df_non_aggregated_counts = pd.read_csv(non_aggregated_counts)
df_all_years = pd.read_csv(dims_all_years_all_countries)

countries = list(set(list(pd.Series((df_all_years['country'])).dropna())))

country = countries[0]
dims = ["electoral", "participatory", "media", "liberal_institution", "liberal_rights"]
dim_n = 1
#for country in countries:
for country in ["poland"]:
    for index, dim in enumerate(dims):
        df_one_country = df_non_aggregated_counts[(df_non_aggregated_counts['country'] == country) & (df_non_aggregated_counts['dimension'] == dim)]
        df_one_country_all_years = df_all_years[df_all_years['country'] == country]
        dim_sum = df_one_country['count'].sum()
        dim_sum2 = df_one_country_all_years.iloc[0][2+index]
        diff = dim_sum - dim_sum2
        print(f'{country} {dim} {dim_sum} {df_one_country_all_years.iloc[0][1+dim_n]} {diff}')
        #print(f'{country} {dim} {diff}')


df_poland = df_non_aggregated_counts[df_non_aggregated_counts['country'] == 'poland']

df_poland_grouped = df_poland.groupby(['dictionary', 'country', 'dimension'])['count'].sum()

sum = 0
dict = {}
for group_label, value in df_poland_grouped.items():
    word, _, dim = group_label
    if(dim == 'participatory'):
        stem = stemmer.stem(word)
        dict[stem] = dict.get(stem, 0) + value
        #print(f'{word} {stemmer.stem(word)} {value}')
        sum += value

for w in dict.keys():
    print(f'{w} {dict[w]}')
print(sum)

