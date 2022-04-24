import pandas as pd
import matplotlib.pyplot as plt
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


#load_yelp_orig_data()

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
