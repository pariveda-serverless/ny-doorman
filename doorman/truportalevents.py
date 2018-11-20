import json
import boto3
from botocore.vendored import requests
import hashlib
import os
from datetime import datetime

def del_none(d):
    """
    Delete keys with the value ``None`` in a dictionary, recursively.

    This alters the input so you may wish to ``copy`` the dict first.
    """
    # For Python 3, write `list(d.items())`; `d.items()` won’t work
    # For Python 2, write `d.items()`; `d.iteritems()` won’t work
    for key, value in list(d.items()):
        if value is None or value == "":
            del d[key]
        elif isinstance(value, dict):
            del_none(value)
    return d  # For convenience

def truportalevents(event, context):
    truportal_username = os.environ['TRUPORTAL_USERNAME']
    truportal_password = os.environ['TRUPORTAL_PASSWORD']
    truportal_ip = os.environ['TRUPORTAL_IP']
    slack_token = os.environ['SLACK_API_TOKEN']

    data = {
        "username": truportal_username,
        "password": truportal_password
    }

    headers = {
        'Content-Type': 'application/json'
    }

    auth_url = "https://%s/api/auth/login" % truportal_ip

    print(auth_url)

    resp = requests.post(auth_url, verify=False, headers=headers, json=data)

    print(resp.json())

    session_key = resp.json()['sessionKey']

    headers = {
        'Content-Type': 'application/json',
        'Authorization': session_key
    }

    door_ids = [4,5]

    skippable_events = ["Door Unlocked", "Pending First User", "Door Forced Alarm", "Door Forced Restore"]

    for door_id in door_ids:
        door_url = "https://%s/api/events?deviceId=%s&limit=10" % (truportal_ip, door_id)

        resp = requests.get(door_url, verify=False, headers=headers)

        eventlist = resp.json()

        eventlist.sort(key=lambda x: x["id"])

        dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
        table = dynamodb.Table('doorman-events')

        for event in eventlist:
            del_none(event)

            response = table.get_item(
                Key={
                    'id': event["id"]
                }
            )

            print(event["id"])
            if not "Item" in response:
                print("No item found")

                datetimeofevent = datetime.utcfromtimestamp(event["timestamp"])

                if door_id == 4:
                    data = {
                        "channel": "sea-doorman",
                        "text": ":alert: The Seattle office's {} has triggered a new message: {}".format(event["deviceName"].lower(), event["description"]),
                        "link_names": True,
                    }
                else:
                    data = {
                        "channel": "sea-cret",
                        "text": ":alert: The Seattle office's {} has triggered a new message: {}".format(event["deviceName"].lower(), event["description"]),
                        "link_names": True,
                    }

                if not event["description"] in skippable_events:
                    resp = requests.post("https://slack.com/api/chat.postMessage", headers={'Content-Type':'application/json;charset=UTF-8', 'Authorization': 'Bearer %s' % slack_token}, json=data)

                response = table.put_item(Item = event)
            else:
                print("Item found")

    return {
        "statusCode": 200,
        "body": json.dumps('Hello from Lambda!')
    }
