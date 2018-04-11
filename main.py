from enums import *
from database import *
from errors import *
from utils import *
from os import sys

# case when single token is queries
def query_token(db, token):
  error = Error_ID.OK
  response, error_id = db.get_doc_ids()
  if (error_id != Error_ID.OK):
    error = error_id
  else:
    doc_ids = response
    f = lambda element: token in element
    response = filter(f, map(db.get_entry, doc_ids))
    print(list(response))
  return response, error

# case when multiple tokens are quried
# example token_a & token_b, token_a | token_b
def query_tokens(token_a, token_b, operation):
  print("ok")

class simple_search_engine():
  def __init__(self):
    print("simple search engine")
    print("--------------------")
    print(" press q anything to quit")
    print("--------------------")
    print(" ")

    self.errors = Errors()
    self.db = database()


  def handle_input(self, response):
    
    if response["action"] == ACTION.DISPLAY:
      self.db.show_entries()
      return

    if response["action"] == ACTION.INDEX:
      doc_index = response["doc_idx"]
      tokens = response["tokens"]
      response, error_id = self.db.add_entry(key=doc_index, value=tokens)
      if (error_id != Error_ID.OK):
        print("out: index ", self.errors.strn(error_id))
      else:
        print("out: index ok ", response)
    else:
      response, error_id = query_token(self.db, response["expression"])
      if (error_id != Error_ID.OK):
        print("out: ", self.errors.strn(error_id))
      else:
        print("out: index ok ", response)

  def run(self):
    error = Error_ID.OK
    command = None

    while (command != "q"): 
     response, error_id = get_input()
     
     if (error_id != Error_ID.OK):
        print("out: ", self.errors.strn(error_id))
     else:
        # --- exit program ----
        if response == "q":
          print(" ")
          print("exiting")
          sys.exit(0)
        # ---- handle validated input ----
        self.handle_input(response)

def main():
  simple_search_engine().run()

if __name__ == "__main__":
  main()
