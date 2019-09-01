import os
import json
import requests
import boto3

'''
save you credentials in aws_creds.json
{
  "bucket" : '',
  "access_key": '',
  "secret_key": '',
  "s3_host": '',
  "end_point": ''
}
'''
class _aws():
    def __init__(self):
        self.bucket_name = None
        self.secret_key = None
        self.access_key = None
        self.end_point = None
        self.s3_host = None
        self.load_creds()

    def load_creds(self):
        with open('aws_creds.json') as f:
            data = json.load(f)
            self.bucket_name = data["aws_bucket"]
            self.secret_key = data["access_key"]
            self.access_key = data["secret_key"]
            self.s3_host = data["s3_host"]
            self.end_point = data["end_point"]

'''
save you credentials in server_creds.json
{
    "server_ip":''
}
'''
class _server():
    def __init__(self):
        self.server_ip = None
        self.load_creds()

    def load_creds(self):
        with open('server_creds.json') as f:
            data = json.load(f)
            self.server_ip = data["server_ip"]
            print(self.server_ip)

class events():
    def __init__(self):
        self.server = _server()
        self.aws = _aws()

    def upload_to_s3_bucket(self, file_name):
        AWS_ACCESS_KEY_ID = self.aws.access_key
        AWS_SECRET_ACCESS_KEY = self.aws.secret_key
        END_POINT = self.aws.end_point
        S3_HOST = self.aws.s3_host
        BUCKET_NAME = self.aws.bucket_name
        FILENAME = file_name
        UPLOADED_FILENAME = "image/"+FILENAME

        s3 = boto3.client('s3',
                         region_name=END_POINT,
                         aws_access_key_id=AWS_ACCESS_KEY_ID,
                         aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

        filename = file_name
        s3.upload_file(filename, 
                       BUCKET_NAME, 
                       "images/" + filename,ExtraArgs={'ACL': "public-read"})
        
        path='https://{}.s3.amazonaws.com/{}'.format(bucket_name, "reports/" + filename)
        return path

    # run prediction for a given image URL
    # URL will be returned from firebase database
    def predict(self, url, access_key):
        print(self.server.server_ip)
        post_req = self.server.server_ip+"predict"

        in_data = dict()
        in_data["image_url"] = url
        in_data["access_key"] = access_key

        in_data = json.dumps(in_data)
        r = requests.post(post_req, data = in_data)

        print(r.json())


    '''
    to be implemented:
    all images whether from test frame work, IOS or andriod app
    will first upload image to S3 buckets get the image url
    send this url to REST api`s to get prediction response.
    '''
    def upload_and_predict(self, file_name):
        pass
        # image_url = upload_to_s3_bucket(file_name)
        # predict(image_url)

    # get server access key
    def get_access_key(self):
        get_req = self.server.server_ip+"get_key"

        in_data = dict()
        in_data["request_key"] = "give_me_access"
        in_data = json.dumps(in_data)

        r = requests.post(get_req, data = in_data)
        r = r.json()
        print(r)
        if r["success"]:
            access_key = r["access_key"]
            return access_key
        else:
            print(r)
            exit()
def main():
    ev = events()
    access_key = ev.get_access_key()
    
    url = raw_input("image_url: ")
    ev.predict(url, access_key)

    #file_name = "test_image.jpg"
    #ev.upload_to_s3_bucket(file_name)

if __name__ == "__main__":
    main()
