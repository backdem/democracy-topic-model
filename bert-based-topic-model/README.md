# bert-based-topic-model
In this approach we use BERTopic modelling on a corpus of country-based evaluation reports. The corpus is scraped and generated using scripts found in [scrape-tool](https://github.com/backdem/scrape-tool) repo. 

## setup
To run the Jupyter notebook; first install the dependencies
```
pip install -r requirements.txt
```
the run the Jupyter interface in the source directory
```
jupyter lab
```

### country_topic_model_bert.ipynb
This notebook implements [guided topic modeling](https://maartengr.github.io/BERTopic/getting_started/guided/guided.html) using BERTopic and a dictionary to seed and guid the modeling. 

### ambiguous_sentences_topic_modelling.ipynb
This notebook analyses the set of sentences labelled as ambiguous to find latent topics in the ambiguity. It applies BERTopic and SentenceTransformer. KeyBERTInspired is used as a representation model to better name the found topics. 

## dataset files
The notebook will attempt to download the corpus file and dictionary file from the [democracy-dataset](https://github.com/backdem/democracy-datasets) repo. The corpus is a CSV file structured as sentence, country, year and source where sentence is a sentence from the report, country, year and source identify where the sentece was extracted from.
