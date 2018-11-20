import json
import boto3
import requests
import hashlib
import os
from urllib.parse import parse_qs

bucket_name = os.environ['BUCKET_NAME']
slack_token = os.environ['SLACK_API_TOKEN']
slack_channel_id = os.environ['SLACK_CHANNEL_ID']
rekognition_collection_id = os.environ['REKOGNITION_COLLECTION_ID']

from doorman import guess
from doorman import train
from doorman import unknown
from doorman import truportalevents
