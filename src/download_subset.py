def get_s3_client():
    return boto3.client("s3", region_name="us-west-2", config=Config(signature_version=UNSIGNED))