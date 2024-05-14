import boto3
import json

def get_bedrock_client():
    bedrockClient = boto3.client("bedrock-runtime")

    return bedrockClient


def invoke_bedrock_embedding_model(bedrockClient, body, EMBEDDING_MODEL_ID, ACCEPT, CONTENT_TYPE):
    response = bedrockClient.invoke_model(
        body=body,
        modelId=EMBEDDING_MODEL_ID,
        accept=ACCEPT,
        contentType=CONTENT_TYPE,
    )
    response_body = json.loads(response.get("body").read())

    embedding = response_body.get("embedding")

    return embedding
