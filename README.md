# Problem Statement 

A model need to build that is able to classify customer complaints based on the products/services. By doing so, the complain tickets can segregate into their relevant categories and, therefore, help in the quick resolution of the issue.

The topic modelling on the <b>.json</b> data provided by the company. Since this data is not labelled, it is need to apply NMF to analyse patterns and classify tickets into the following five clusters based on their products/services:

* Credit card / Prepaid card

* Bank account services

* Theft/Dispute reporting

* Mortgages/loans

* Others 


Each ticket can be mapped onto its respective department/category with the help of topic modelling. The data obtained can be used to train any supervised model such as logistic regression, decision tree, or random forest. By using this trained model, any new customer complaint support ticket can be classified into its relevant department.

# Complain Classification
This Python script performs complain classification based on customer complaints regarding products/services. The goal is to classify each complaint into one of the following categories: Credit card/Prepaid card, Bank account services, Theft/Dispute reporting, Mortgages/loans, and Others. The script utilizes topic modeling using NMF (Non-Negative Matrix Factorization) to identify patterns and assign categories to the complaints. The overall pipeline consists of the following steps:

1. Data loading: The script loads complaint data in JSON format and converts it into a DataFrame.
2. Text preprocessing: The complaint text is preprocessed by converting it to lowercase, removing text in square brackets, removing punctuation, and removing words containing numbers.
3. Exploratory data analysis (EDA): The script visualizes the distribution of complaint character lengths and generates a word cloud to display the top 40 words by frequency among the processed complaints. It also identifies the top unigrams, bigrams, and trigrams by frequency.
4. Feature extraction: TF-IDF (Term Frequency-Inverse Document Frequency) is used to convert the complaint texts into a matrix of features.
5. Topic modeling: NMF is applied to the feature matrix to perform topic modeling and identify the best number of clusters. The resulting topics and their associated words are displayed.
6. Model building using supervised learning: In this script, NMF is used as an unsupervised learning technique for topic modeling. Therefore, there is no explicit model building using supervised learning.
7. Model training and evaluation: As the script focuses on unsupervised learning and topic modeling, there is no separate model training or evaluation step.
8. Model inference: The final step of the script maps the complaints to their respective topics based on the NMF results.

Dependencies:
- numpy
- pandas
- re
- nltk
- spacy
- string
- en_core_web_sm
- seaborn
- matplotlib
- plotly
- scikit-learn
- textblob
- wordcloud

Note: Some libraries may need to be installed using pip.

To run the script, ensure that the complaint data in JSON format is accessible. Modify the file path in the script to load the data correctly.
