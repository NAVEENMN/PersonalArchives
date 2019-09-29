import json
import pymongo

def construct_error_resp(code=400, message="Invalid request"):
    return { 'statusCode': code, 'body': json.dumps(message)}

db_url = "mongodb+srv://admin:cortexadmin123@cluster0-onaoj.mongodb.net/test?retryWrites=true&w=majority"

class database():
    def __init__(self):
        self.client = pymongo.MongoClient(db_url)
        self.databases = self.client.database_names()
    
    def get_conn(self, db_name=None):
        conn = None
        if db_name not in self.databases:
            return construct_error_resp(404, "db not found")
        else:
            conn = self.client[db_name]
        return conn            

db = database()
def handle_task(task, payload):
    result = {}
    if task == "get_loc":
        db_conn = db.get_conn(db_name="sample_geospatial")
        collection = db_conn["shipwrecks"]
        if "query" in payload:
            result = collection.find_one(payload["query"])
    return result

def lambda_handler(event, context):
    params = event
    if "body" in event:
        params = json.loads(event["body"]) # json
    if "task" not in params:
        resp = invalid_error
    else:
        response = handle_task(params["task"], params["payload"])
        response.pop("_id") # object is not json searlize error
        resp = { 'statusCode': 200, 'body': json.dumps(response)}
    return resp
