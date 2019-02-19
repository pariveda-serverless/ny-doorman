import json
import boto3
import requests
import hashlib
import os

bucket_name = os.environ['BUCKET_NAME']
slack_token = os.environ['SLACK_API_TOKEN']
slack_channel_id = os.environ['SLACK_CHANNEL_ID']
slack_training_channel_id = os.environ['SLACK_TRAINING_CHANNEL_ID']
rekognition_collection_id = os.environ['REKOGNITION_COLLECTION_ID']

# TODO
#truportal_username = os.environ['TRUPORTAL_USERNAME']
#truportal_password = os.environ['TRUPORTAL_PASSWORD']
#door_id = os.environ['DOOR_ID']
#truportal_ip = os.environ['TRUPORTAL_IP']

def guess(event, context):
    client = boto3.client('rekognition')
    key = event['Records'][0]['s3']['object']['key']
    event_bucket_name = event['Records'][0]['s3']['bucket']['name']
    image = {
        'S3Object': {
            'Bucket': event_bucket_name,
            'Name': key
        }
    }
    # print(image)

    s3 = boto3.resource('s3')

    try:
        resp = client.search_faces_by_image(
            CollectionId=rekognition_collection_id,
            Image=image,
            MaxFaces=1,
            FaceMatchThreshold=80)

    except Exception as ex:
        # no faces detected, delete image
        print("No faces found, deleting")
        s3.Object(bucket_name, key).delete()
        return

    if len(resp['FaceMatches']) == 0:
        # no known faces detected, let the users decide in slack
        print("No matches found, sending to unknown")
        new_key = 'unknown/%s.jpg' % hashlib.md5(key.encode('utf-8')).hexdigest()
        s3.Object(bucket_name, new_key).copy_from(CopySource='%s/%s' % (bucket_name, key))
        s3.ObjectAcl(bucket_name, new_key).put(ACL='public-read')
        s3.Object(bucket_name, key).delete()
        return
    else:
        print ("Face found")
        print (resp)
        user_id = resp['FaceMatches'][0]['Face']['ExternalImageId']
        similarity = resp['FaceMatches'][0]['Similarity']
        # move image
        new_key = 'detected/%s/%s.jpg' % (user_id, hashlib.md5(key.encode('utf-8')).hexdigest())
        s3.Object(bucket_name, new_key).copy_from(CopySource='%s/%s' % (event_bucket_name, key))
        s3.ObjectAcl(bucket_name, new_key).put(ACL='public-read')
        s3.Object(bucket_name, key).delete()

        # fetch the username for this user_id
        data = {
            "token": slack_token,
            "user": user_id
        }
        print(data)
        resp = requests.post("https://slack.com/api/users.info", data=data)
        print(resp.content)
        print(resp.json())
        username = resp.json()['user']['name']
        userid = resp.json()['user']['id']

        if int(similarity) > 80:
            data = {
                "channel": slack_channel_id,
                "text": "Welcome @{} ".format(username),
                "link_names": True,
                "attachments": [
                    {
                        "image_url": "https://s3.amazonaws.com/%s/%s" % (bucket_name, new_key),
                        "fallback": "Nope?",
                        "callback_id": new_key,
                        "attachment_type": "default"
                    }
                ]
            }
            resp = requests.post("https://slack.com/api/chat.postMessage", headers={'Content-Type':'application/json;charset=UTF-8', 'Authorization': 'Bearer %s' % slack_token}, json=data)

        data = {
            "channel": slack_training_channel_id,
            "text": "Matched {} with similarity {:.2f}%)".format(username, similarity),
            "link_names": True,
            "attachments": [
                {
                    "image_url": "https://s3.amazonaws.com/%s/%s" % (bucket_name, new_key),
                    "fallback": "Nope?",
                    "callback_id": new_key,
                    "attachment_type": "default",
                    "actions": [{
                            "name": "username",
                            "text": "Select a username...",
                            "type": "select",
                            "data_source": "users"
                        },
                        {
                            "name": "username",
                            "text": "Guess Was Right",
                            "style": "primary",
                            "type": "button",
                            "value": userid
                        }
                    ]
                }
            ]
        }


        resp = requests.post("https://slack.com/api/chat.postMessage", headers={'Content-Type':'application/json;charset=UTF-8', 'Authorization': 'Bearer %s' % slack_token}, json=data)
        
        # TODO
        '''
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

        door_url = "https://%s/api/devices/doors/%s/state?command=grant-access" % (truportal_ip, door_id)

        print(door_url)

        #resp = requests.post(door_url, verify=False, headers=headers, json=data)

        print(resp.json())
        '''
        
        return {}
