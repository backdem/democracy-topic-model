## BERTopic Modelling

### Introduction

BERTopic was employed to perform topic modeling on the processed text corpus. This approach was guided by a dictionary of terms provided by domain experts, ensuring that the identified topics were relevant to the project's objectives. The dictionary-driven approach allowed for a more focused analysis, aligning the topics with predefined dimensions of interest.

### Corpus Preparation

Corpus has been prepared by scraping various online sources. The corpus dataset is found in this [repository](https://github.com/backdem/democracy-datasets).
Each entry in the corpus included the country, year, source, and the sentence itself. Additionally, the script calculated the length of each sentence in words, which was used as a feature during the topic modeling process.

### Guiding Dictionary

A key aspect of this topic modeling approach was the use of a guiding dictionary, stored in a JSON file, which contained terms categorized by domain experts.
The dictionary which is found in this [repository](https://github.com/backdem/democracy-datasets) contains keywords for each of the democratic dimensions e.g. electoral, media, particpatory.
This dictionary was crucial for steering the BERTopic model towards discovering topics that were meaningful within the context of the research.

### Country-Specific Analysis

The BERTopic model was applied on a per-country basis, allowing for the identification of topics specific to each country.
This means that certain parameters such as `min_topic_size` needed to be adjusted for each country else number of topics
retrieved for each country varied widely.

### Modeling Process

The BERTopic model was trained on the sentences from the selected country's corpus. The model identified clusters of sentences that represented coherent topics, with the guidance of the predefined dictionary terms. This process involved:

- **Sentence Embeddings**: Sentences were transformed into embeddings that captured their semantic content.
- **Topic Discovery**: Using the embeddings, the BERTopic model identified clusters of sentences that formed distinct topics.
- **Topic Refinement**: The model refined these topics by considering the guiding dictionary, ensuring alignment with the research objectives.

### Challenges and Considerations

Several challenges were encountered during the BERTopic modeling process:

- **Dictionary Alignment**: Ensuring that the topics discovered by the model aligned well with the guiding dictionary required careful tuning and validation.
- **Country-Specific Variability**: The variability in text content across different countries necessitated a flexible modeling approach that could adapt to different national contexts.

Although the topic modelling was instrumental to find latent topics, aligning to our dictionary proved non-feasible.