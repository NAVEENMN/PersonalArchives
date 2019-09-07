import os
import json
import flask
import pymongo

current_dir = os.path.dirname(os.path.abspath(__file__))

app = flask.Flask(__name__, template_folder='template')
db_url = "mongodb+srv://test_user:test@cluster0-onaoj.mongodb.net/test?retryWrites=true&w=majority"

class database():
    def __init__(self):
        self.client = pymongo.MongoClient(db_url)
        self.collections = self.client.database_names()

    def get_conn(self, collection_name=None):
        conn = None
        if collection_name not in self.collections:
            return {"error_id": 400, "desp": "collections not found"}
        else:
            if collection_name == "geographical_data":
                conn = client.geographical_data
        return conn

    def insert_a_record(self, collection=None, data=None):
        conn = self.get_conn(collection)
        # add data verification methods
        # create a seperate file for data verification
        post_id = db.posts.insert_one(data).inserted_id
        return post_id

db = database()

@app.route("/insert_record", methods=["GET", "POST"])
def insert_record():
    response = {"success": False}
    if flask.request.method == "POST":
        req_data = request.get_json()
        # verify if request from valid source
        collection = req_data['collection_name']
        data = req_data['data']
        try:
            insert_id = db.insert_a_record(collection, data)
            response = {"success": True}
        except:
            return response
        return response
    if flask.request.method == "GET":
        response = {"success": True, "attr": "collection_name, data"}
        return response


def main():
    app.run(host="0.0.0.0", port=80)

if __name__ == "__main__":
    main()
