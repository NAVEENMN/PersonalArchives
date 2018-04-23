import sys
import math
from utils import *
from functools import reduce

class node():
  '''
  __init__ method of node
  Args:
    op(str): operation & or |
    left_child(str or node) : child containing node or a token
    right_child(str or node) : child containing node or a token
  '''
  def __init__(self, op, left_child, right_child):
    self.op = op
    self.left = left_child
    self.right = right_child

class op_tree():
  '''
  __init__ method of op_tree
  Note:
    op_tree provides access to build execution tree
  '''
  def __init__(self):
    self.root = None
  
  '''
    function: create_node(self, op, left, right)
    Args:
      op(str): operation & or |
      left(str or node) : child containing node or a token
      right(str or node) : child containing node or a token
    Returns:
      node: A node containing & or | as parent and tokens 
            or node as childrens 
  '''
  def create_node(self, op, left, right):
    _node = node(op, left, right)
    dangling_node = (type(left) == str and type(right) == str)
    if not ((self.root != None) and (dangling_node)):
      self.root = _node
    return _node

'''
  function: apply_op(lst_a, lst_b, op)
  Args:
    lst_a(list) : list containing document ids
    lst_b(list) : list containing document ids
    op(str) : operation to apply & or |
  Returns:
    list : A list reduced based on operation op
'''
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


'''
  function: _query(db, term)
  Args:
    db(database) : Refrence to database defined in utils.py
    term(string or list): token
  Returns:
    list : list of documents matching a token
'''
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


'''
  function: run_query(db, operation, left, right)
  Args:
    db(database) : Refrence to database defined in utils.py
    operation(str): operation to apply & or |
    left(node|str) : Reference to left sub tree
    right(node|str): Reference to right sub tree
  Returns:
    list : list of documents after merging results from
           left and right subtree
'''
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

'''
  function: reduce_tree(db, node)
  Note:
    Inorder based tree traversal
  Args:
    db(database) : Refrence to database defined in utils.py
    node : Reference to left or right sub tree
  Returns:
    list: list of documents after traversing execution tree
'''
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

'''
  function: build_query_tree(query)
  Note:
    Builds execution tree based expression from query.
    Usage of stacks op_stack, term_stack are 
    explained in documentation.
  Args:
    query(str): query expression
  Returns:
    node : Reference to root node
'''
def build_query_tree(query):
  op_stk, term_stk = [], []
  term = ""
  _op_tree = op_tree()

  if query.isalpha():
    node = _op_tree.create_node("&", query, query)
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
        node = _op_tree.create_node(op_stk.pop(), left, right)
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
    node = _op_tree.create_node(op_stk.pop(), left, right)
    term_stk.append(node)

  return _op_tree.root

'''
   function: get_and_validate_input
   Note:
     gets commands from user and validates them
   Args:
     None
   Returns:
     response(dict): dictonary containing following 
                     information
                     response["action"] : (str) & or |
                     response["doc_idx"] : (int) document id
                     response["expression"] : (str) query expression
                     response["tokens"] : (str) tokens for index
     error (Enum.Error_ID) : Enum for corresponding error
'''
def get_and_validate_input():
  error = Error_ID.OK
  response = {}
  command = str(input("in: "))
  
  if command == "q":
    return command, error

  error, tokens = validate_input(command)

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
      response["tokens"] = tokens
  
  return response, error

'''
  function: term_frequency(terms)
  Note:
    computes term frequency as 
    explained in documentation
  Args:
    terms (list): list of tokens
  Returns:
    list : [[term, tf], [term, tf], ...]
'''
def term_frequency(terms):
  N = len(terms) # number of terms
  tf = map(lambda term: [term, 1+math.log(terms.count(term))], terms)
  return list(tf)

'''
  function: doc_frequency(terms)
  Note:
    computes document frequency as 
    explained in documentation
  Args:
    terms (list): list of tokens
    results(list): list of document id satisfying 
                   query expression
    db(database) : Refrence to database defined in utils.py
  Returns:
    list : [[term, df], [term, df], ...]
'''
def doc_frequency(term, results, db):
  f = lambda t, dcid: db.has_token(t, dcid)
  df = len(list(filter(lambda doc_id: f(term, doc_id), results)))
  return df

'''
  function: idf(doc_id, results, db)
  Note:
    computes inverse document frequency as 
    explained in documentation
  Args:
    doc_id(int): For a given document id
    results(list): list of document id satisfying 
                   query expression
    db(database) : Refrence to database defined in utils.py
  Returns:
    list : [[term, idf], [term, idf], ...]
'''
def idf(doc_id, results, db):
  D = len(results) # number of documents
  terms = db.get_entry(doc_id)
  _map = map(lambda term: [term, math.log(0.001+(D/doc_frequency(term, results, db)))], terms)
  return list(_map)

'''
  function: tfidf(tf, idf)
  Note:
    computes term frequency and 
    inverse document frequency as 
    explained in documentation
  Args:
    tf(list): list of term frequencies
    idf(list): list of inverse document frequencies
  Returns:
    list : [tf-idf, tf-idf, ...]
'''
def tfidf(tf, idf):
  return list(map(lambda tf_,idf_: [tf_[0], tf_[1]*idf_[1]], tf[1], idf[1]))

'''
  function: value(query, term_values)
  Note:
    compare and return all tf-idfs matching a query term
  Args:
    query(str): A given query
    term_values(list): list of all tf-idfs
  Returns:
    list : [tf-idf, tf-idf, ...]
'''
def value(query, term_values):
  values = map(lambda qv: qv[1] if qv[0] == query else 0.0, term_values)
  return list(set(values))

'''
  function: query_value(query, tf-idfs)
  Note:
    For a given query term grab tf-idfs across all results
    and add them
  Args:
    query(str): A given query
    tf-idfs(list): list of all tf-idfs
  Returns:
    list : [query_value, query_value, ...]
'''
def query_value(query, tfidfs):
  qv = map(lambda x: [x[0], sum(value(query, x[1]))], tfidfs)
  return list(qv) 

'''
  function: rank_results(db, results, query_tokens)
  Note:
    Compute tf-idfs, sort based on score and return results
  Args:
    db(database) : Refrence to database defined in utils.py
    results(list): list of document ids satisfying expression
    query_tokens(list): list of tokens from expression
  Returns:
    list : [document_id, document_id, ...]
'''
def rank_results(db, results, query_tokens):

  # compute tfidf
  _tf = list(map(lambda doc_id: [doc_id, term_frequency(db.get_entry(doc_id))], results))
  _idf = list(map(lambda doc_id: [doc_id, idf(doc_id, results, db)], results))
  _tfidf = list(map(lambda tf, idf: [tf[0], tfidf(tf, idf)], _tf, _idf))
  
  # values of query terms with respect to results
  q_values = list(map(lambda query_token: query_value(query_token, _tfidf), query_tokens))

  # value of documents with respect query terms
  net_scores = list(reduce(lambda lst1, lst2: map(lambda x, y: [x[0], x[1]+y[1]], lst1, lst2), q_values))

  # rank documents based on net scores
  ranked = [doc[0] for doc in sorted(net_scores, key=lambda x: x[1], reverse=True)]

  return ranked

'''
  function: simple_search_engine()
  Note:
    Main wrapper to handle all functions
    and print results to console.
  Args:
    None
  Returns:
    None
'''
class simple_search_engine():
  def __init__(self):
    print("simple search engine")
    print("--------------------")
    print(" press q anytime to quit")
    print(" example add index:")
    print(" index 1 salt pepper fish")
    print(" example query:")
    print(" query ((salt&pepper)|butter)")
    print("--------------------\n")
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
        result = rank_results(self.db, result, payload["tokens"])
        result = map(lambda x: str(x), result)
        response = "out: results "+ " ".join(list(result))
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
