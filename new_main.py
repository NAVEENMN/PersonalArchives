import sys
from enum import Enum
from errors import *

class node():
  def __init__(self, op, left_child, right_child):
    self.op = op
    self.left = left_child
    self.right = right_child

class op_tree():
  def __init__(self):
    self.root = None
  def insert(self, op, left, right):
    _node = node(op, left, right)
    dangling_node = (type(left) == str and type(right) == str)
    if not ((self.root != None) and (dangling_node)):
      self.root = _node
    return _node
  def print_tree(self, node):
    if node == None:
      return
    if type(node) == str:
      print("tr", node)
      return
    self.print_tree(node.left)
    print("tr", node.op)
    self.print_tree(node.right)

def apply_op(lst_a, lst_b, op):
  result = set()
  a_set = set(lst_a)
  b_set = set(lst_b)
  if op == "&":
    if (a_set & b_set):
      result = a_set & b_set
  else:
    if (a_set | b_set):
      result = a_set | b_set
  return list(result)

def run_query(db, operation, left, right):
  error_id = Error_ID.OK
  response, error_id = db.get_doc_ids()
  if (error_id != Error_ID.OK):
    error = error_id
  else:
    result_left, result_right= [], []
    ids = response

    if type(left) == str:
      foo = lambda idx: True if left in db.get_entry(idx) else False
      result_left = list(filter(foo, ids))
    else:
      result_left = left

    if type(right) == str:
      foo = lambda idx: True if right in db.get_entry(idx) else False
      result_right = list(filter(foo, ids))
    else:
      result_right = right
   
    result = apply_op(result_left, result_right, operation)
    return result

def reduce_tree(db, node):
  if node == None:
    return
  if type(node) == str:
    print("run", node)
    ids, error_id = db.get_doc_ids()
    foo = lambda idx: True if node in db.get_entry(idx) else False
    result = list(filter(foo, ids))
    return result 
  if type(node.left) == str and type(node.right) == str:
    print("run", node.left, node.op, node.right)
    return run_query(db, node.op, node.left, node.right)
  left_result = list(reduce_tree(db, node.left))
  right_result = list(reduce_tree(db, node.right))
  print("run", left_result, node.op, right_result)
  return run_query(db, node.op, left_result, right_result)

def res_query(term):
  ids, error_id = db.get_doc_ids()
  foo = lambda idx: True if term in db.get_entry(idx) else False
  result = list(filter(foo, ids))
  return result

def build_query_tree(query):
  op_stk, term_stk = [], []
  cnt = 0
  term = ""
  _op_tree = op_tree()
  print(query)
  if query.isalpha():
    node = _op_tree.insert("&", query, query)
    return node
  for x in range(0, len(query)):
    if query[x] == "(" or query[x] == ")":
      if query[x] == "(":
        cnt += 1
      else:
        cnt -= 1
        if term != "":
          term_stk.append(term)
          term = ""
        print(query[x], cnt, term_stk, op_stk)
        right = term_stk.pop()
        if len(term_stk) > 0:
          left = term_stk.pop()
        else:
          continue
        node = _op_tree.insert(op_stk.pop(), left, right)
        term_stk.append(node)
    else:
        print(query[x], cnt, term_stk, op_stk)
        if query[x] == "&" or query[x] == "|":
          if term != "":
            term_stk.append(term)
            term = ""
          op_stk.append(query[x])
        else:
          term += query[x]
  
  if term != "":
    term_stk.append(term)
    term = ""
  
  while len(op_stk) != 0:
    right = term_stk.pop()
    left = term_stk.pop()
    node = _op_tree.insert(op_stk.pop(), left, right)
    term_stk.append(node)
  
  #_op_tree.print_tree(_op_tree.root)
  return _op_tree.root
   
# DataBase
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
    return list(self.db.keys()), error

# Enum Definations
class ACTION(Enum):
  INDEX = 0
  QUERY = 1
  DISPLAY = 2

class Error_ID(Enum):
  OK = 0
  ERR_EMPTY_INPUT = 1
  ERR_INVALID_COMMAND = 2
  ERR_INVALID_DOC_INDEX = 3
  ERR_NO_DOC_ID = 4
  ERR_NO_TOKENS = 5
  ERR_KEY_NOT_FOUND_IN_DB = 6
  ERR_DB_ENTRY = 7
  ERR_TOKENS_NON_ALPHA = 8
  ERR_EMPTY_DB = 9
  ERR_NO_EXP = 10
  ERR_INVALID_EXP = 11
  ERR_INVALID_TOKENS_EXP = 12

# Error Handlings
def err_to_str(err_id):
  if err_id == 0:
    return "ok"
  if err_id == 1:
    return "error input is empty"
  if err_id == 2:
    return ""

class Errors():
  def __init__(self):
    self.err_str = {}
    self.err_str[Error_ID.OK] = "ok"
    self.err_str[Error_ID.ERR_EMPTY_INPUT] = "error input is empty"
    self.err_str[Error_ID.ERR_INVALID_COMMAND] = "error invalid input (valid: index | query)"
    self.err_str[Error_ID.ERR_INVALID_DOC_INDEX] = "error invalid document index"
    self.err_str[Error_ID.ERR_NO_DOC_ID] = "error no document id was provided"
    self.err_str[Error_ID.ERR_NO_TOKENS] = "error no tokens were provided"
    self.err_str[Error_ID.ERR_KEY_NOT_FOUND_IN_DB] = "error key not found in db"
    self.err_str[Error_ID.ERR_DB_ENTRY] = "error unknown occured while db entry"
    self.err_str[Error_ID.ERR_TOKENS_NON_ALPHA] = "error tokens with non alphanumeric provided"
    self.err_str[Error_ID.ERR_EMPTY_DB] = "error database is empty"
    self.err_str[Error_ID.ERR_NO_EXP] = "error no expression provided"
    self.err_str[Error_ID.ERR_INVALID_EXP] = "error invalid expression provided"
    self.err_str[Error_ID.ERR_INVALID_TOKENS_EXP] = "error expression contains invalid tokens"
  def strn(self, err_id):
    return self.err_str[err_id]

def get_and_validate_input():
  error = Error_ID.OK
  response = {}
  command = str(input("in: "))
  
  if command == "q":
    return command, error

  if len(command) == 0:
    error = Error_ID.ERR_EMPTY_INPUT
  else:
    parts = command.split(" ")
    if (parts[0] == "index") or (parts[0] == "query"):

      if (parts[0] == "index"):

        response["action"] = ACTION.INDEX
        
        # check if valid doc id
        if parts[1].isdigit():
          response["doc_idx"] = int(parts[1])
        else:
          error = Error_ID.ERR_INVALID_DOC_INDEX

        # check if valid tokens
        response["tokens"] = parts[2:]
      
      else:

        response["action"] = ACTION.QUERY
        response["expression"] = parts[1]
    
    else:
      error = Error_ID.ERR_INVALID_COMMAND
  
  return response, error

class simple_search_engine():
  def __init__(self):
    self.db = database()
    self.errors = Errors()

  def handle_input(self, response):
      error_id = Error_ID.OK
      if response["action"] == ACTION.INDEX:
        doc_index = response["doc_idx"]
        tokens = response["tokens"]
        response, error_id = self.db.add_entry(key=doc_index, value=tokens)
        if (error_id != Error_ID.OK):
          print("out: index ", self.errors.strn(error_id))
        else:
          print("out: index ok ", response)
      else:
        # build query tree
        root = build_query_tree(response["expression"])
        result = reduce_tree(self.db, root)
        print(result)

  def run(self):
    command = None
    while (command != "q"):
      response, error_id = get_and_validate_input()
      if response == "q":
        print("\nexiting")
        sys.exit(0)
      if (error_id != Error_ID.OK):
          print("out: ", self.errors.strn(error_id))
      else:
          self.handle_input(response)

def main():
  simple_search_engine().run()

if __name__ == "__main__":
  main()
