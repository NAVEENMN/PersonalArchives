import sys
from utils import *
from functools import reduce

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

def _query(db, term):
  error_id = Error_ID.OK
  ids, error_id = db.get_doc_ids()
  result = []
  if type(term) == str:
    foo = lambda idx: True if term in db.get_entry(idx) else False
    result = list(filter(foo, ids))
  else:
    result = term
  return result

def run_query(db, operation, left, right):
  error_id = Error_ID.OK
  result = []
  response, error_id = db.get_doc_ids()
  if (error_id != Error_ID.OK):
    error = error_id
  else:
    result_left, result_right= [], []
    ids = response
    result_left = _query(db, left)
    result_right = _query(db, right)
    _map = map(lambda x: _query(db, x), [left, right])
    result = reduce(lambda x,y: apply_op(x, y, operation), _map)
    result = list(result)
  return result

def reduce_tree(db, node):
  if node == None:
    return
  
  if type(node) == str:
    return _query(db, node)

  if type(node.left) == str and type(node.right) == str:
    return run_query(db, node.op, node.left, node.right)

  left_result = list(reduce_tree(db, node.left))
  right_result = list(reduce_tree(db, node.right))
  return run_query(db, node.op, left_result, right_result)

def build_query_tree(query):
  op_stk, term_stk = [], []
  term = ""
  _op_tree = op_tree()

  if query.isalpha():
    node = _op_tree.insert("&", query, query)
    return node

  for x in range(0, len(query)):
    if query[x] == "(" or query[x] == ")":
      if query[x] == ")":
        if term != "":
          term_stk.append(term)
          term = ""
        right = term_stk.pop()
        if len(term_stk) > 0:
          left = term_stk.pop()
        else:
          continue
        node = _op_tree.insert(op_stk.pop(), left, right)
        term_stk.append(node)
    else:
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
   
def get_and_validate_input():
  error = Error_ID.OK
  response = {}
  command = str(input("in: "))
  
  if command == "q":
    return command, error

  error = validate_input(command)

  if error == Error_ID.OK:
    parts = command.split(" ")
    action = parts[0]
    if action == "index":
      response["action"] = action
      response["doc_idx"] = int(parts[1])
      response["tokens"] = parts[2:]  
    else:
      response["action"] = action
      response["expression"] = parts[1]
  
  return response, error

class simple_search_engine():
  def __init__(self):
    self.db = database()
    self.errors = Errors()

  def handle_input(self, payload):
    error_id = Error_ID.OK
    response = None
    if payload["action"] == "index":
      doc_index = payload["doc_idx"]
      tokens = payload["tokens"]
      _, error_id = self.db.add_entry(key=doc_index, value=tokens)
      if (error_id != Error_ID.OK):
        response = "out: index " + self.errors.strn(error_id)
      else:
        response = "out: index ok " + str(doc_index)
    else:
      # build query tree
      try:
        root = build_query_tree(payload["expression"])
      except:
        error_id = Error_ID.ERR_INVALID_EXP
        response = "out: "+self.errors.strn(error_id)
        return response, error_id
      
      # reduce tree
      try:
        result = reduce_tree(self.db, root)
        result = list(map(lambda x: str(x), result))
        response = "out: results "+ " ".join(result)
      except:
        error_id = Error_ID.ERR_INVALID_EXP
        response = "out: "+self.errors.strn(error_id)

    return response, error_id

  def run(self):
    while (True):
      response, error_id = get_and_validate_input()
      if response == "q":
        print("\nexiting")
        sys.exit(0)
      
      if (error_id == Error_ID.ERR_EMPTY_INPUT):
        continue

      if (error_id != Error_ID.OK):
        print("out: ", self.errors.strn(error_id))
      else:
        response, error_id = self.handle_input(response)
        print(response)

def main():
  simple_search_engine().run()

if __name__ == "__main__":
  main()
