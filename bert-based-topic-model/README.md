# bert-based-topic-model
In this approach we use BERTopic modelling on a corpus of country-based evaluation reports. The corpus is scraped and generated using scripts found in [scrape-tool](https://github.com/backdem/scrape-tool) repo. 

## setup
To run the Jupyter notebook; first install the dependancies
```
pip install -r requirments.txt
```
the run the Jupyter interface in the source directory
```
jupyter lab
```
## dataset files
The notebook will attempt to download the corpus file and dictionary file from the [democracy-dataset](https://github.com/backdem/democracy-datasets) repo. The corpus is a CSV file structured as sentence, country, year and source where sentence is a sentence from the report, country, year and source identify where the sentece was extracted from.
