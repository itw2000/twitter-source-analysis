
"""
This file deals api calls to twitter, defining the schema for storage
in mongoDB and storing data in collections.
"""

import pymongo
from pymongo import errors
from twython import TwythonStreamer, Twython, TwythonError
import re
import string

#Open twitter api key
filename = 'apikey.txt'
fi = open(filename, 'r')
L1 = fi.readline()
L1 = string.split(L1, ",")


ckey = L1[0]
csecret = L1[1]
atoken = L1[2]
asecret = L1[3]

#Set up OAuth
twitter = Twython(ckey, csecret, atoken, asecret)

#Connect to mongoDB
client = pymongo.MongoClient()

#client.drop_database('twitter')

db = client.twitter

tweets = db.tweets

users = db.tweets


class MyStreamer(TwythonStreamer):
    """
This is the Twython streaming class that takes in streaming data from twitter.
I have modified it to send data to the collection Tweets using the data model indicated
    """

    def on_success(self, data):
        #Checks if a tweet coming from the streamer are actually tweets
        if 'text' in data:
            print data['text'].encode('utf-8')
            #Creates a dictionary to store data from the returned tweet
            result = dict()
            result['id'] = data['id']
            result['user'] = data['user']
            result['place'] = data['place']
            result['retweeted'] = data['retweeted']
            result['text'] = data['text']
            result['created_at'] = data['created_at']
            #Get the platform the tweet was sent
            try:
                result['source'] = (string.split(re.sub('</a>', '', data['source']), '>'))[1]
            except IndexError:
                result['source'] = data['source']
            #Insert the tweet dict into the collection tweets
            try:
                tweets.insert(result)
            except errors.DuplicateKeyError:
                tweets.insert(None)

    def on_error(self, status_code, data):
        print status_code
        self.disconnect()


#Creates a MyStream instance, streamer, and calls the statuses.sample method
def get_stream():

    streamer = MyStreamer(ckey, csecret, atoken, asecret)

    streamer.statuses.sample()


def get_friends(tweet_id):

#Create collection of friends attributes and store in mongo

    #Get the twitter cursor for the get_friends_list api call
    results = twitter.cursor(twitter.get_friends_list, user_id=tweet_id)
    for data in results:

        result = {}
        #For each user looking for last tweet and entering the source for that user into the db
        try:
            last_tweet = twitter.get_user_timeline(screen_name=data['screen_name'], count=1)

            try:
                result['source'] = (string.split(re.sub('</a>', '', last_tweet[0]['source']), '>'))[1]
            except IndexError:
                try:
                    result['source'] = last_tweet[0]['source']
                except IndexError:
                    result['source'] = 0

        except TwythonError:
            print "Fault"

        #For each entry in the cursor go through and assign the defined fields int he collection.
        result['id'] = data['id']
        result['screen_name'] = data['screen_name']
        result['location'] = data['location']
        result['time_zone'] = data['time_zone']
        result['created_at'] = data['created_at']
        result['followers'] = data['followers_count']
        result['friends'] = data['friends_count']

        #Insert the document into the collection.
        db['user'].insert(result)

    """
    data = twitter.show_user(screen_name=screen)

    result = {}

    result['id'] = data['id']
    result['screen_name'] = data['screen_name']
    result['location'] = data['location']
    result['time_zone'] = data['time_zone']
    result['created_at'] = data['created_at']
    result['followers'] = data['followers_count']
    result['friends'] = data['friends_count']

    users.insert(result)
    """

































