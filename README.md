# democracy-topic-model
Democracies in Europe and beyond are facing threats of sliding back into authoritarianism. Despite initially promising signs of liberalization, ‘democratic backsliding’ has prominently occurred in Russia and Turkey, but also in Poland and Hungary and in established democracies such as France and the UK. Democratic backsliding has attracted the attention of international agencies (e.g., Freedom House, V.Dem), which regularly assesses the quality of democracy in different countries. Nevertheless, such attempts suffer from subjectivity bias as they mostly rely on qualitative judgments produced by country experts. We lack a comparative view of the dimensions and quality of democratic assessments. In this context, BackDem addresses the following questions:
* To what degree assessment reports vary in grading countries by traits of democracy and over time?
* How precise are existing assessments of democracies and do assessment reports systematically differ in their vagueness?

BackDem aims to develop a digital tool for text processing that: 
1) Maps dimensions of democratic quality in texts.
2) Assesses the precision of democratic assessments.

## repository structure
The code in the repo is split into two main approaches, both are python notebooks. For more details switch to the appropriate folder: 
* Using [BERTopic](bert-based-topic-model) to do topic modelling on country reports.
* Using [keyword extraction](keyword-based-extraction-model) and a dcitionary to measure democractic dimensions. 
