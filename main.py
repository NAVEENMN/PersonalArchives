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
    f = lambda element: True if token[0] in db.get_entry(element) else False
    response = list(filter(f, doc_ids))
  return response, error

# case when multiple tokens are quried
# example token_a & token_b, token_a | token_b

# And operation
def comman_elements(lst_a, lst_b):
  a_set = set(lst_a)
  b_set = set(lst_b)
  if (a_set & b_set):
    return a_set & b_set
  else:
    return set()

# Or operation
def join_elements(lst_a, lst_b):
  a_set = set(lst_a)
  b_set = set(lst_b)
  if (a_set | b_set):
    return a_set | b_set
  else:
    return set()

def get_tokens(que):
  error = Error_ID.OK
  query = que[0]
  db = que[1]
  response, error_id = db.get_doc_ids()
  doc_ids = response
  f = lambda element: True if query in db.get_entry(element) else False
  response = list(filter(f, doc_ids))
  return response

def query_tokens(db, query_a, query_b, operation):
  error = Error_ID.OK
  response, error_id = db.get_doc_ids()
  if (error_id != Error_ID.OK):
    error = error_id
  else:
    response = list(map(get_tokens, [[query_a, db], [query_b, db]]))
    if operation == "&":
      response = comman_elements(response[0], response[1])
    else:
      response = join_elements(response[0], response[1])
  return response, error

def hand_op(db, token, oper):
  error = Error_ID.OK
  response = None
  parts = token.split(oper)
  if len(parts) > 1:
    response, error = query_tokens(db, parts[0], parts[1], oper)
  else:
    response, error = query_token(db, [token])
  return response, error

def merge(res1, res2, oper):
  if oper == "&":
    result = comman_elements(res1, res2)
  else:
    result = join_elements(res1, res2)
  return result

def foo(db, token):
  error = Error_ID.OK
  response = None
  if "&" in token:
    response, error = hand_op(db, token, "&")
  else:
    response, error = hand_op(db, token, "|")
  if error == Error_ID.OK:
    return response
  else:
    return set()

def reduce_results(db, query):
  error = Error_ID.OK
  if len(query) == 1:
    response = foo(db, query[0])
  else:
    result = foo(db, query[0])
    response = reduce(lambda x,y: merge(foo(db, x), foo(db, y[1:]), y[0]), query)
  return response, error

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
      if response["single"]:
        response, error_id = query_token(self.db, response["expression"])
      else:
        qa = response["expression"]
        response, error_id = reduce_results(self.db, qa)
      if (error_id != Error_ID.OK):
        print("out: ", self.errors.strn(error_id))
      else:
        print("out: results ", response)

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
