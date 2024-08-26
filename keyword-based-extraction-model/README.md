# keyowrd-based-extraction-model

In this appraoch we apply a dictionary to the corpus to measure the different dimensions of democracy. Each dimension is characterised by a set of keyword/keyphrases listed in our dictionary. The notebook will process
the corpus and generate statistical plots comparing various aspects of the measures.
The corpus is scraped and generated using scripts found in [scrape-tool](https://github.com/backdem/scrape-tool) repo.

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

## genrate intermediate datafile

On the first run the notebook will generate an intermediate datafile, non_aggregated_counts.csv, which counts all the dictionary keywords in the corpus. This file is then used to generate the plots. The generation of this intermediate file will take a while.

## more details

[DETAILS.md](./DETAILS.md)
