"""
Primarily for querying the database and collecting statistics on collected fields.
Queries are transformed into pandas dataframes for analysis.
"""
import pymongo
import numpy as np
import pandas as pd
import random
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from collect_store import get_friends



random.seed()

#Set up connection to db
client = pymongo.MongoClient()
db = client.twitter
tweets = db.tweets
user = db.user


"""
Function makes and empty query and returns the source field
it then turns it into a format that can be then be sent to
d3 for plotting.
"""

def get_platform(k):

    #Get a random sample of entries from db based on resevoir sampling algorithim.
    #Avoid loading entire dataset into memory when sampling
    results = list(tweets.find({}, {"source":1})[0:k])

    #Get the size of the collection and size of the sample
    size = tweets.count()
    stream = tweets.find({}, {"source":1})[k+1:size]

    for tweet in stream:
        index = random.randrange(0, size)
        if index < k :
            results[index] = tweet

    results = list(results)

    #Iterate over list of results and append entries to list
    platform = []
    for result in results:
        try:
            platform.append(result['source'])
        except KeyError:
            platform.append(None)


    #Cleaning data and moving it to a pandas series object so it can normalized and written
    platform = ['Twitter for Android' if x=='Twitter for  Android' else x for x in platform]
    platform = np.array(platform)
    platform = pd.Series(platform)
    platform = platform.value_counts(normalize=True)
    s1 = pd.Series(platform.index)
    s2 = pd.Series(platform.values)
    platform = pd.concat([s1, s2], axis=1)
    platform.columns = ['platform','freq']

    #Send the data to folder to update data for d3 graphics
    a = platform.to_json('data.json', orient="records")

    return a



#A class which holds the dataframe of the query for users of a particular twitter platform and computes
#summary statistics on a number of fields

class ComputeStats:


    def __init__(self):

        results = list(tweets.find({"source":"Twitter for iPhone", "user.followers_count":{ "$lt" : 1000}}, {"source":1,"user.screen_name":1,"retweeted":1, "user.verified":1, "user.followers_count":1,"user.friends_count":1, "user.listed_count":1,"user.statuses_count":1}))

        username = []
        retweeted = []
        verified = []
        followers = []
        friends = []
        stat_count = []
        source = []
        listed = []

        #Loop through results cursor and append lists for each attribute
        for result in results:

            username.append(result['user']['screen_name'])
            retweeted.append(result['retweeted'])
            verified.append(result['user']['verified'])
            followers.append(result['user']['followers_count'])
            friends.append(result['user']['friends_count'])
            stat_count.append(result['user']['statuses_count'])
            listed.append(result['user']['listed_count'])
            try:
                source.append(result['source'])
            except KeyError:
                source.append(None)

        #Create a numpy array from the lists and then a pandas dataframe
        stats_ar = np.array((username, retweeted, verified, followers, friends, stat_count, listed, source))
        self.stats_ar = stats_ar.transpose()
        stats_df = pd.DataFrame(self.stats_ar, columns=['username','retweeted','verified','followers','friends',
                                                        'stat_count', 'listed', 'source'])
        self.stats_df = stats_df.convert_objects(convert_numeric=True)
        self.stats_df.to_json('../twitter-source-analysis/d3/platform_data.json',orient="records")




    #Function to compute the summary statistics for the number of followers and friends of a particular
    #platform user takes argument type as the string of column you want to compute summery stats for.
    def get_stats(self, type):
        stats = self.stats_df[type].astype(int).describe()
        self.stats_df.boxplot(column=(type))
        return stats


#Selects a users friends and compiles a dataframe of basic data for each user who is friends with
#a give user. Built on the collection for each user.
def friend_list():

    results = list(user.find({}))

    username = []
    date_created = []
    loc = []
    source = []
    friends = []
    followers = []

    for result in results:

        username.append(result['screen_name'])
        date_created.append(result['created_at'])
        loc.append(result['location'])
        try:
            source.append(result['source'])
        except KeyError:
            source.append('none')
        friends.append(result['friends'])
        followers.append(result['followers'])
    friends_ar = np.array((username, source, date_created, loc, friends, followers))
    friends_ar = friends_ar.transpose()
    friends_df = pd.DataFrame(friends_ar, columns=['username','source','date_created','location','followers','friends'])
    a = friends_df.to_json('friend_list.json', orient="records")

    return friends_df

















