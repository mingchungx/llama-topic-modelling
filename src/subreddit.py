"""
This script collects data using the REDDIT API for the subreddit r/LocalLLaMA.
"""

import praw
from dotenv import load_dotenv
import os

load_dotenv()
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_APP_NAME = os.getenv("REDDIT_APP_NAME")
REDDIT_USERNAME = os.getenv("REDDIT_USERNAME")
REDDIT_PASSWORD = os.getenv("REDDIT_PASSWORD")

reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_APP_NAME
)

SUBREDDIT = reddit.subreddit('LocalLLaMA')

MIN_SUBMISSIONS = 5 # this is our minimum number of submissions per request
MAX_SUBMISSIONS = 1000 # this is defined maximum number of submissions per request

class SubmissionResult:
    def __init__(self, title, score, upvote_ratio, poster):
        self.title = title
        self.score = score
        self.upvote_ratio = upvote_ratio
        self.poster = poster

def get_subreddit_data_min():
    data = []
    for submission in SUBREDDIT.top(limit=MIN_SUBMISSIONS):
        submission_result = SubmissionResult(
            submission.title,
            submission.score,
            submission.upvote_ratio,
            submission.author if submission.author else "[deleted]"
        )
        data.append(submission_result)

        # comments = reddit.submission(id=submission.id).comments
        # comments.replace_more(limit=None)
        # for comment in comments.list():
        #     submission_result = SubmissionResult(comment.body, comment.score)
        #     data.append(submission_result)

    return data

def search_subreddit_data_min(query="privacy"):
    data = []
    for submission in SUBREDDIT.search(query, sort="relevance", time_filter="all", limit=MIN_SUBMISSIONS):
        submission_result = SubmissionResult(
            submission.title,
            submission.score,
            submission.upvote_ratio,
            submission.author if submission.author else "[deleted]"
        )
        data.append(submission_result)
    return data

def get_subreddit_data_max():
    data = []
    for submission in SUBREDDIT.top(limit=MAX_SUBMISSIONS):
        submission_result = SubmissionResult(
            submission.title,
            submission.score,
            submission.upvote_ratio,
            submission.author if submission.author else "[deleted]"
        )
        data.append(submission_result)
    return data

def search_subreddit_data_max(query="privacy"):
    data = []
    for submission in SUBREDDIT.search(query, sort="relevance", time_filter="all", limit=MAX_SUBMISSIONS):
        submission_result = SubmissionResult(
            submission.title,
            submission.score,
            submission.upvote_ratio,
            submission.author if submission.author else "[deleted]"
        )
        data.append(submission_result)
    return data
