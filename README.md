## Sentiment Analysis on Amazon Reviews - Polarity

This repository contains my approach for perfroming sentiment analysis on [Amazon Reviews-Polarity](https://course.fast.ai/datasets).
I have performed various steps in for capturing sentiments in this dataset including data preprocessing , data downsampling , exploratory data anlysis , building and training model , evaluating model .

## About the dataset
34,686,770 Amazon reviews from 6,643,669 users on 2,441,053 products, from the Stanford Network Analysis Project (SNAP). This subset contains 1,800,000 training samples and 200,000 testing samples in each polarity sentiment.


### Description
The Amazon reviews polarity dataset is constructed by taking review score 1 and 2 as negative, and 4 and 5 as positive. Samples of score 3 is ignored. In the dataset, class 1 is the negative and class 2 is the positive. Each class has 1,800,000 training samples and 200,000 testing samples.

The files train.csv and test.csv contain all the training samples as comma-sparated values. There are 3 columns in them, corresponding to class index (1 or 2), review title and review text. The review title and text are escaped using double quotes ("), and any internal double quote is escaped by 2 double quotes (""). New lines are escaped by a backslash followed with an "n" character, that is "\n".


## Data Preprocessing & Downsampling
For preprocessing the Dataset we have to apply several transformations to the entire dataset viz:
- Removing Punctuations.  `gpp.strip_punctuation`
    - We need to remove punctuations to treat each word sentiment equally.
- Remove stopwords .  `gpp.remove_stopwords`
    - Words such as *the , for , in* which does not convey message to the machine should be removed.
- Removing extra whitespaces. `gpp.strip_multiple_whitespaces`
    - Removing extra white space for saving memory.
- Removing small words. `gpp.strip_short`
    - removing words with length < 3.
- Stripping HTML Tags. `gpp.strip_tags`
    - Removing any HTML tags that will not aid in sentiment analysis.
- Removing Numericals `gpp.strip_numeric`
    - Numericals not necessarily convey any meaning to our sentiments.

Since the whole dataset contains 34,686,770 Amazon reviews I have to downsample the data to as it is not possible to evaluate on the whole dataset. I have downsampled the data with equal distributions of each `class` in the dataset.

## Exploratory Data Analysis
I have performed EDA with `TextBlob` to calculate sentiment scores and found out that there is a high correlation with sentiment of user and the `class` in which the review belong.

## Sentiment Analysis with BERT model
Since words alone cannot define the sentiment of the user and combination of words can convey different meaning it is quite difficult to perform sentiment analysis with classical ML models. 

There are many models that can generalise for sentiment analysis but I found [`BERT Experts`](https://tfhub.dev/google/experts/bert/wiki_books/sst2/2) That has been properly trained specifically for Sentiment analysis and has performed better than any other models.

## Limitations and Improvements
### Limitations
- I didn't trained the model on the entire dataset.
- `Data Augmentation` was not applied.
### Improvements
- `Efficient Feature Engineering` can be done on the data that can make the model to generalize well 
- `Data augmentation` can be applied to improve performance and accuracy on the data.
- Increasing batch size and training for more iterations can be done.