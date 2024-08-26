## BERTopic Modelling

### Introduction

BERTopic was employed to perform topic modeling on the processed text corpus. This approach was guided by a dictionary of terms provided by domain experts, ensuring that the identified topics were relevant to the project's objectives. The dictionary-driven approach allowed for a more focused analysis, aligning the topics with predefined dimensions of interest.

### Corpus Preparation

The corpus, which had been structured and labeled in the previous step, was loaded into a Pandas DataFrame. Each entry in the corpus included the country, year, source, and the sentence itself. Additionally, the script calculated the length of each sentence in words, which was used as a feature during the topic modeling process.

### Guiding Dictionary

A key aspect of this topic modeling approach was the use of a guiding dictionary, stored in a JSON file, which contained terms categorized by domain experts. This dictionary was crucial for steering the BERTopic model towards discovering topics that were meaningful within the context of the research.

### Country-Specific Analysis

The BERTopic model was applied on a per-country basis, allowing for the identification of topics specific to each country. The script enabled the selection of a country from the corpus, ensuring that the analysis could be tailored to specific national contexts. This approach facilitated a more granular understanding of the topics relevant to each country.

### Modeling Process

The BERTopic model was trained on the sentences from the selected country's corpus. The model identified clusters of sentences that represented coherent topics, with the guidance of the predefined dictionary terms. This process involved:

- **Sentence Embeddings**: Sentences were transformed into embeddings that captured their semantic content.
- **Topic Discovery**: Using the embeddings, the BERTopic model identified clusters of sentences that formed distinct topics.
- **Topic Refinement**: The model refined these topics by considering the guiding dictionary, ensuring alignment with the research objectives.

### Challenges and Considerations

Several challenges were encountered during the BERTopic modeling process:

- **Dictionary Alignment**: Ensuring that the topics discovered by the model aligned well with the guiding dictionary required careful tuning and validation.
- **Country-Specific Variability**: The variability in text content across different countries necessitated a flexible modeling approach that could adapt to different national contexts.
