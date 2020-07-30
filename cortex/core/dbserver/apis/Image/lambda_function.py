import json
import pymongo
from bson import json_util, ObjectId

def construct_resp(code=400, message="Invalid request"):
    return { 'statusCode': code, 'body': json.dumps(message, default=json_util.default)}
    
db_url = "mongodb+srv://admin:cortexadmin123@cluster0-onaoj.mongodb.net/test?retryWrites=true&w=majority"

class database():
    def __init__(self):
        self.client = pymongo.MongoClient(db_url)
        self.databases = self.client.database_names()
    
    def get_conn(self, db_name=None):
        conn = None
        if db_name not in self.databases:
            return construct_resp(404, "db not found")
        else:
            conn = self.client[db_name]
        return conn            

db = database()
def handle_task(task, payload):
    result = {}
    
    if task == "get_image_data":
        db_conn = db.get_conn(db_name="images")
        collection = db_conn["metadata"]
        try:
            image_id = payload["query"]["image_id"]
            obj = ObjectId(str(image_id))
            res = collection.find_one({"_id":obj})
            message = {"status":"ok", "data": res}
            result = construct_resp(code=200, message=message)
        except Exception as e:
            result = construct_resp(code=500, message=str(e))
            
    if task == "add_image_data":
        db_conn = db.get_conn(db_name="images")
        collection = db_conn["metadata"]
        try:
            res = collection.insert_one(payload)
            res = str(res.inserted_id)
            message = {"status":"ok", "image_id": res}
            result = construct_resp(code=200, message=message)
        except Exception as e:
            result = construct_resp(code=500, message=str(e))
            
    return result

def lambda_handler(event, context):
    params = event
    if "body" in event:
        params = json.loads(event["body"]) # json
    if ("task" not in params) or ("payload" not in params):
        response = invalid_error
    else:
        response = handle_task(params["task"], params["payload"])
    return response
