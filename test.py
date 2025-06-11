import boto3
from botocore.client import Config
import requests

# ==== 🔑 Replace with your actual R2 credentials and endpoint ====
ACCESS_KEY = 'sssssssssssssssss'
SECRET_KEY = 'sdfffffffffff'
ENDPOINT_URL = 'https://3b0d5fa769d0ad9288cc7ffc64baba9b.r2.cloudflarestorage.com'  # example: https://abc1234567890.r2.cloudflarestorage.com
BUCKET = 'documents'
OBJECT_KEY = 'pdfs/e55b902a-4070-44b2-8c3a-9fc8d861a03b/2506.07903v1.pdf'
# ==================================================================

# Step 1: Create S3-compatible client
s3 = boto3.client(
    's3',
    endpoint_url=ENDPOINT_URL,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    config=Config(signature_version='s3v4'),
    region_name='auto'
)

# Step 2: Generate a presigned URL valid for 1 hour
try:
    signed_url = s3.generate_presigned_url(
        'get_object',
        Params={'Bucket': BUCKET, 'Key': OBJECT_KEY},
        ExpiresIn=3600
    )
    print("✅ Signed URL generated:\n", signed_url)
except Exception as e:
    print("❌ Failed to generate signed URL:", e)
    exit(1)

# Step 3: Test downloading the object
try:
    response = requests.get(signed_url, timeout=10)
    if response.status_code == 200:
        print("✅ File accessed successfully.")
        print(f"Content-Type: {response.headers.get('Content-Type')}")
        print(f"Size: {len(response.content)} bytes")
    else:
        print(f"❌ Access failed. Status code: {response.status_code}")
        print(response.text)
except requests.exceptions.RequestException as req_err:
    print("❌ Failed to download file:", req_err)
