import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as funct
import torch.optim as optim

import torch

import constant


def load_yelp_orig_data():
    path_to_yelp_reviews = constant.INPUT_FOLDER + '/yelp_academic_dataset_review.json'

    # read the entire file into a python array
    with open(path_to_yelp_reviews, 'r', encoding='utf-8', errors='replace') as f:
        data = f.readlines()

    # remove the trailing "\n" from each line
    data = map(lambda x: x.rstrip(), data)

    data_json_str = "[" + ','.join(data) + "]"

    # now, load it into pandas
    data_df = pd.read_json(data_json_str)

    data_df.head(100000).to_csv(constant.OUTPUT_FOLDER + '/output_reviews_top.csv')


# load_yelp_orig_data()

top_data_df = pd.read_csv(constant.OUTPUT_FOLDER + '/output_reviews_top.csv')
print("Columns in the original dataset:\n")
print(top_data_df.columns)

print("Number of rows per star rating:")
print(top_data_df['stars'].value_counts())


# Function to map stars to sentiment
def map_sentiment(stars_received):
    if stars_received <= 2:
        return -1
    elif stars_received == 3:
        return 0
    else:
        return 1


# Mapping stars to sentiment into three categories
top_data_df['sentiment'] = [map_sentiment(x) for x in top_data_df['stars']]
# Plotting the sentiment distribution
plt.figure()
pd.value_counts(top_data_df['sentiment']).plot.bar(title="Sentiment distribution in df")
plt.xlabel("Sentiment")
plt.ylabel("No. of rows in df")
plt.show()


# Function to retrieve top few number of each category
def get_top_data(top_n=5000):
    top_data_df_positive = top_data_df[top_data_df['sentiment'] == 1].head(top_n)
    top_data_df_negative = top_data_df[top_data_df['sentiment'] == -1].head(top_n)
    top_data_df_neutral = top_data_df[top_data_df['sentiment'] == 0].head(top_n)
    top_data_df_small = pd.concat([top_data_df_positive, top_data_df_negative, top_data_df_neutral])
    return top_data_df_small


# Function call to get the top 10000 from each sentiment
top_data_df_small = get_top_data(top_n=10000)

# After selecting top few samples of each sentiment
print("After segregating and taking equal number of rows for each sentiment:")
print(top_data_df_small['sentiment'].value_counts())
top_data_df_small.head(10)

# Pre processing

# Removing the stop words
from gensim.parsing.preprocessing import remove_stopwords

print(remove_stopwords("Restaurant had a really good service!!"))
print(remove_stopwords("I did not like the food!!"))
print(remove_stopwords("This product is not good!!"))


from gensim.utils import simple_preprocess

# Tokenize the text column to get the new column 'tokenized_text'
top_data_df_small['tokenized_text'] = [simple_preprocess(line, deacc=True) for line in top_data_df_small['text']]
print(top_data_df_small['tokenized_text'].head(10))


from gensim.parsing.porter import PorterStemmer

porter_stemmer = PorterStemmer()

# Get the stemmed_tokens
top_data_df_small['stemmed_tokens'] = [[porter_stemmer.stem(word) for word in tokens] for tokens in
                                       top_data_df_small['tokenized_text']]
top_data_df_small['stemmed_tokens'].head(10)


from sklearn.model_selection import train_test_split


# Train Test Split Function
def split_train_test(top_data_df_small, test_size=0.3, shuffle_state=True):
    X_train, X_test, Y_train, Y_test = train_test_split(top_data_df_small[
                                                            ['business_id', 'cool', 'date', 'funny', 'review_id',
                                                             'stars', 'text', 'useful', 'user_id', 'stemmed_tokens']],
                                                        top_data_df_small['sentiment'],
                                                        shuffle=shuffle_state,
                                                        test_size=test_size,
                                                        random_state=15)
    print("Value counts for Train sentiments")
    print(Y_train.value_counts())
    print("Value counts for Test sentiments")
    print(Y_test.value_counts())
    print(type(X_train))
    print(type(Y_train))
    X_train = X_train.reset_index()
    X_test = X_test.reset_index()
    Y_train = Y_train.to_frame()
    Y_train = Y_train.reset_index()
    Y_test = Y_test.to_frame()
    Y_test = Y_test.reset_index()
    print(X_train.head())
    return X_train, X_test, Y_train, Y_test


# Call the train_test_split
X_train, X_test, Y_train, Y_test = split_train_test(top_data_df_small)


# Use cuda if present
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device available for running: ")
print(device)
