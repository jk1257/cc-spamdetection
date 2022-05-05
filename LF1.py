import os
import io
import re
import json
import logging
import boto3
import email
import urllib.parse
import datetime
import re
import numpy as np
from sms_spam_classifier_utilities import one_hot_encode
from sms_spam_classifier_utilities import vectorize_sequences


def lambda_handler(event, context):
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    vocabulary_length = 9013
    endpoint_name = os.environ['ENDPOINT_NAME']

    session = boto3.Session()
    s3_session = session.client('s3')
    response = s3_session.get_object(Bucket=bucket, Key=key)

    email_obj = email.message_from_bytes(response['Body'].read())
    from_email = email_obj.get('From')
    if type(email_obj.get_payload()[0]) == str:
        body = email_obj.get_payload()
    else:
        body = email_obj.get_payload()[0].get_payload()

    runtime = session.client('runtime.sagemaker')
    vocabulary_length = 9013
    input_mail = [body.strip()]

    temp_1 = one_hot_encode(input_mail, vocabulary_length)
    input_mail = vectorize_sequences(temp_1, vocabulary_length)

    data = json.dumps(input_mail.tolist())
    response = runtime.invoke_endpoint(EndpointName=endpoint_name, ContentType='application/json', Body=data)
    res = json.loads(response["Body"].read())

    if res['predicted_label'][0][0] == 0:
        label = 'HAM'
    else:
        label = 'SPAM'
    score = round(res['predicted_probability'][0][0], 4)
    score = score*100
    

    message = "We received your email sent at " + str(email_obj.get('To')) + " with the subject " + str(email_obj.get('Subject')) + ".\nHere \
    is a 240 character sample of the email body:\n\n" + body[:240] + "\nThe email was \
    categorized as " + str(label) + " with a " + str(score) + "% confidence."

    email_client = session.client('ses')
    response_email = email_client.send_email(
        Destination={'ToAddresses': [from_email]},
        Message={
            'Body': {
                'Text': {
                    'Charset': 'UTF-8',
                    'Data': message,
                },
            },
            'Subject': {
                'Charset': 'UTF-8',
                'Data': 'Spam analysis of your email',
            },
        },
        Source=str(email_obj.get('To')),
    )

    return 'Success!'
    