Tweet Analysis
=======================

A set scripts for collecting, wrangling, cleaning, and analysis. This is all done on top of a MongoDB database which is iniated and collects data from the Twitter streaming API via Twython. Requires most python science packages (numpy, scipy, scikit-learn).

####Collect_store.py 
Includes the functions to sample the Twitter streaming API and move those tweets into a defined MongoDB collection Tweets. It also includes another Twitter API call function which takes a users screenname as the argument and fills a collection with info on that users friend.

####Data_analyze.py

This module includes 1 class and 2 functions:

1. ```ComputeStats()``` which pulls a few fields fo every entry in the collection and transforms that to pandas dataframe. It includes one method ```get_stats(type)``` which accepts a string of one of the features and computes basic summary statistics for that feature.

2. ```get_platform(k)``` is a function which accepts 1 argument k which determines the size of sample to take from the collection. From that sample the function counts the source of the tweet in the database and then writes that as a csv. 

3. ```friend_list()``` d populates a collection with data about a users friends and the source of their last tweet. 

####Model.py

Function ```classify``` currently builds/evaluates classifiers ```explore``` prints summary stats and histograms of friends, followers, statuses, listed counts.
