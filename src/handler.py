# Handler designed for aws lambda
import boto3
from detect import detect_from_image
from os.path import exists

MODEL_NAME = "yolov2"
MODEL_FILE = "yolov2.h5"
MODEL_DIR = "/tmp"

model_path = "{}/{}".format(MODEL_DIR, MODEL_FILE)

if not exists(model_path):
    s3_client = boto3.client('s3')
    s3_client.download_file(MODEL_NAME, MODEL_FILE, model_path)

def detect_images_from_urls(urls):
    out = []
    for url in urls:
        detection = detect_from_image(url)
        detection["status"] = "SUCCESS"
        detection["url"] = url
        out.append(detection)
    return out

def handler(event, context):
    urls = event.get("urls") or event["multiValueQueryStringParameters"]["urls"]
    if type(urls) is str:
        urls = [urls]
    out = detect_images_from_urls(urls)

    return {
        "isBase64Encoded": False,
        "statusCode": 200,
        "headers": { },
        "body": str(out)
    }