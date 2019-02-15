import json
import urllib.parse
import boto3

print('Loading function')

s3 = boto3.client('s3')
sns = boto3.client('sns')


def lambda_handler(event, context):
    
    # Get the object from the event and show its content type
    arn = 'arn:aws:sns:us-east-1:636148530971:plabs-reb-sns-batch'
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
    
    try:
        #Reads JSON from whithin file
        s32 = boto3.resource('s3')
        content_object = s32.Object(bucket, key)
        file_content = content_object.get()['Body'].read().decode('ascii')
        json_content = json.loads(file_content)
        
        print('Json Object: {}'.format(json_content))
        
        responseSns = sns.publish(
                        TargetArn=arn,
                        Message=json.dumps(json_content)
                        #Message=file_content
                    )
                    
    except Exception as e:
        print(e)
        print('Error getting object {} from bucket {}. Make sure they exist and your bucket is in the same region as this function.'.format(key, bucket))
        raise e

