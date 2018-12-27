# The Brakefield Enterprises RPG Classifier&trade;
### A Data Science Project by Brian Brakefield

## Table of Contents

- [Overview](#Overview)
- [Reading Reddit's API](#Reading_Reddit's_API)
- [Model Building](#Model_Building)
- [Making the "Product"](#Making_the_"Product")


## Overview:

The objective of this project was to build a model that can distinguish whether a Role-Playing Game can be classified as Pen-and-Paper or Video Game, based only on text data from the game's subreddit. My process was as follows:
1. gather and format data from [Reddit](https://www.reddit.com/) subs for [pen-and-paper](https://www.reddit.com/r/rpg/) and [video game](https://www.reddit.com/r/rpg_gamers/) RPGs.
2. build and test models to predict whether which posts came from which subreddit
3. write a python script to import the training data, rebuild the most accurate model, scrape and format the data from a subreddit, predict whether posts are closer to pen-and-paper or video game RPGs, then return which subreddit the mean of posts' predictions belong to (in use, a user calls the function on a subreddit in question, and the function tells them whether they're looking at a pen-and-paper or a video game RPG)

## Reading Reddit's API

For each of the two subreddits in question:
1. I used a ```For``` loop and Reddit's API (the ```'.json'``` extension) to extract all available json data from each of the [pen-and-paper RPG](https://www.reddit.com/r/rpg/) and [video game RPG](https://www.reddit.com/r/rpg_gamers/) subreddits and store that data as a list.
2. From the list of lists, I extracted what features I needed (title text, body text, post url, and subreddit) and built a Pandas Dataframe using those fields reformatted as columns:
| Column  | Description | Data Type|
| ------------- | ------------- |-------------|
| sub  | which subreddit from which this post came | int |
| name  | post's unique name  | str |
| text  | Combined title and post body text  | str |
| url | web address for the individual post  | str |
| comments  | List of strings, one string for each comment's texts  | List |

3. Iterating over the URLs in the DataFrames, I used Reddit's API to scrape all the text, if any, from the comments stores for a given entry. I added the data back into the appropriate working DataFrame.
4. I exported the DataFrames into csv files so that I could work with the data without re-running my code over and over. I decided that 1000 posts with up to 50 comments each would be provide plenty of data from which we could train our models.

## Model Building

After reading in the training data csvs, I concatenated the dataframes and iterated over each row to combine the text from the post and all the comments into one long string.

Next I split the data into training and testing data, stratifying it to ensure the approximate 50/50 split of data remained. This allowed me to create a Count Vectorizer to sort the 4000 most common words in the subreddits' training data. Having thus converted the strings into numeric columns, I could finally start testing models.

The first I ran was a Naive Bayes model, which predicted posts' subreddits 95% of the time. I also tested a VotingClassifier model which included KNeighborsClassifier, LogisticRegression, AdaBoostClassifier, and Naive Bayes as well. After testing with different voting weights and hyperparameters, the best scores I was able to reach were the same as the Naive Bayes model, which was understandably faster to compute! Finally, I tested a Random Forest model by itself, once again testing multiple hyperparameters by use of a GridSearchCV. The best score I reached using a RandomForest was 92.2%. Having run all these tests, I thus determined to utilize the Naive Bayes model for my final product.

## Making the "Product"

My last step was to incoprorate all these steps into a single Python script. The [RPG-Classifier](RPG-Classifier.ipynb) file in this repository is an example of the function imported and run in a simple notebook. The function runs in just over 5 minutes due to the manual delays I wrote into the scraping code so as not to get 429 codes when hitting Reddit's API. I'm happy with the product anyway, however, as every test case I ran was successful.
