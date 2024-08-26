## Keyword-based Topic Extraction

### Introduction

After encountering challenges in aligning the topics found by the BERTopic model with our predefined dictionary, we adopted a keyword-based topic extraction approach. This method relies on a dictionary of terms provided by domain experts, ensuring that the topics extracted from the text corpus directly correspond to the key dimensions of interest.

### Text Preprocessing

Before extracting topics, the text corpus underwent extensive preprocessing to standardize the content. This involved several steps:

- **Tokenization**: The text was split into individual tokens (words) using the `nltk` library.
- **Stopword Removal**: Common English stopwords, which do not contribute to the semantic content of the text, were removed.
- **Stemming**: Words were reduced to their root forms using the Porter Stemmer, allowing for more accurate matching with the dictionary terms.

### Keyword Matching and Counting

The core of the keyword-based extraction method involved matching the processed tokens against a predefined dictionary. This dictionary, created by domain experts, contained terms categorized under various topics or dimensions relevant to the research. The matching process differed for unigrams compared to other n-grams:

- **Unigram Matching**: For unigrams, both the dictionary terms and the corpus were stemmed using the Porter Stemmer. This ensured that variations of a word (e.g., "democracy" and "democratic") were correctly identified and matched. After stemming, the occurrences of each stemmed unigram were counted across the corpus.
- **N-gram Matching (Bigrams and Trigrams)**: For bigrams and trigrams, exact matches were used instead of stemming. The corpus was scanned for precise phrases as listed in the dictionary. This approach was critical for capturing the meaning of multi-word expressions, where the context provided by the phrase was essential for topic identification.

### Normalization Techniques

To ensure that the keyword counts were meaningful and comparable across different documents and datasets, several normalization techniques were applied:

- **Z-Score Normalization**: The keyword counts were converted into Z-scores, allowing for the identification of keywords that were unusually frequent or infrequent relative to the mean.
- **Residual Analysis**: Residuals were calculated after fitting a linear regression model to the keyword counts. This analysis identified keywords that were over- or under-represented in specific documents compared to what was expected.
- **Linear Model Fitting**: A linear model was applied to predict keyword occurrences based on document features. By normalizing the counts with this model, the analysis could focus on keywords that were genuinely significant, beyond what could be expected by chance.

### Compiling Results

The results of the keyword matching and normalization were compiled into a structured dataset, where each document (or sentence) was associated with normalized counts of the various unigrams, bigrams, and trigrams. The dataset allowed for detailed analysis of the prevalence of each topic across different countries, years, and sources.

### Statistical Analysis and Visualization

To interpret the results, statistical analysis was conducted on the normalized keyword counts. This analysis involved:

- **Distribution Analysis**: Examining the distribution of normalized unigrams, bigrams, and trigrams to identify the most and least prevalent topics.
- **Visualization**: Using tools such as `plotly` to create visual representations of the keyword distributions, providing insights into the topics covered by the corpus.

### Challenges and Considerations

Several challenges were encountered during the keyword-based extraction and normalization process:

- **Stemming Consistency**: Ensuring that the stemming process for unigrams did not overly reduce words, leading to incorrect keyword matches.
- **N-gram Matching**: Managing the complexity of exact matching for bigrams and trigrams, especially in cases where phrases could overlap or be interpreted differently.
- **Normalization Accuracy**: Applying and interpreting the results of Z-score normalization and residual analysis required careful consideration to avoid misinterpreting the significance of certain keywords.
- **Dictionary Coverage**: The predefined dictionary needed to be comprehensive enough to cover the varied language used in the corpus. Iterative refinement of the dictionary was necessary to improve accuracy.
