from errors import *

class database():
  def __init__(self):
    self.db = dict()

  def add_entry(self, key, value):
    error = Error_ID.OK
    response = key
    try:
      self.db[key] = value
    except:
      error = Error_ID.ERR_DB_ENTRY
    return response, error

  def delete_entry(self, key):
    error = Error_ID.OK
    if key in self.db.keys():
      del self.db[key], error
    else:
      error = Error_ID.ERR_KEY_NOT_FOUND_IN_DB
    return error

  def has_token(self, token, doc_idx):
    response = False
    if token in self.db[doc_idx]:
      response = True
    return response

  def get_entry(self, key):
    return list(self.db[key])

  def get_doc_ids(self):
    error = Error_ID.OK
    response = None
    if len(self.db.keys()) == 0:
      error = Error_ID.ERR_EMPTY_DB
    else:
      response = self.db.keys()
    return list(self.db.keys()), error

  def show_entries(self):
    print(" ")
    print("key    tokens")
    print("-------------")
    for key in self.db.keys():
      print(key, self.db[key])
    print(" ")
