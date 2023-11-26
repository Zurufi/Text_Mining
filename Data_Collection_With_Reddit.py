# Ali Alzurufi
# Professor Lauren
# Date: September 29 2023
# MCS 5223: Text Mining and Data Analytics

""" Description: This program will retrieve records from 2 subreddits and transform the json format into a final dataframe with target columns 0 and 1 for the subreddits.
    The final dataframe will be put into a text file. """


import praw
import pandas as pd

reddit = praw.Reddit(
    client_id="client_id",
    client_secret="client_secret",
    user_agent="my user agent",
)

subreddit_name = "OnePiece"
max_records = 100

post_data = []

# get information from the subreddit
for submission in reddit.subreddit(subreddit_name).hot(limit=max_records):
    post_info = {
        "Title": submission.title,
        "Author": str(
            submission.author
        ),  # Convert to string to handle possible None values
        "Target": 0,
    }
    post_data.append(post_info)

# convert to dataframe
df1 = pd.DataFrame(post_data)


subreddit_name2 = "Luffy"
post_data2 = []

# get information from the subreddit
for submission in reddit.subreddit(subreddit_name2).hot(limit=max_records):
    post_info2 = {
        "Title": submission.title,
        "Author": str(submission.author),
        "Target": 1,
    }
    post_data2.append(post_info2)

# Convert to dataframe
df2 = pd.DataFrame(post_data2)

final_df = pd.concat(
    [df1[["Title", "Author", "Target"]], df2[["Title", "Author", "Target"]]]
)

# save as text file
text_file = "your.txt"

# Save the df to text file
final_df.to_csv(text_file, sep="\t", index=False)



