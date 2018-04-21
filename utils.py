import re
from functools import reduce
from enum import Enum

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
  ERR_UNKNOWN = 13

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
    self.err_str[Error_ID.ERR_UNKNOWN] = "An unkown error occured"
  def strn(self, err_id):
    return self.err_str[err_id]

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

def validate_parenthesis_and_tokens(expression):
  error = Error_ID.OK
  stack = list()
  tokens = []
  term = ""
  
  for x in range(0, len(expression)):
    char = expression[x]
    if char == "(" or char == ")":
      if char == "(":
        stack.append(char)
      else:
        stack.pop()
        if term != "":
          tokens.append(term)
          term = ""
    else:
      # char must be A-Z,a-z,&,|
      if char == "&" or char == "|":
        if term != "":
          tokens.append(term)
          term = ""
      else:
        term += char

  # check if query has mismatching parenthesis
  if len(stack) != 0:
    error = Error_ID.ERR_INVALID_EXP
    return error

  if term != "":
    tokens.append(term)
    term = ""
  
  # check if tokens contain illegal characters
  validate_tokens = lambda token: re.match('^[\w-]+$', token) is not None
  valid = reduce(lambda x,y: x and y, map(validate_tokens, tokens))
  if not valid:
    error = Error_ID.ERR_TOKENS_NON_ALPHA
  
  return error

def validate_input(query):
  error = Error_ID.OK
  if len(query) == 0:
    error = Error_ID.ERR_EMPTY_INPUT
  else:
    parts = query.split(" ")
    action = parts[0]
    # check if command begins with index or query
    if (action == "index") or (action == "query"):

      # validate index command
      if (action == "index"):
        
        # check if command contains document id and terms
        if len(parts) < 3:
          error = Error_ID.ERR_NO_DOC_ID
          return error

        # check if document id is a valid integer
        document_id = parts[1]
        if not document_id.isdigit():
          error = Error_ID.ERR_INVALID_DOC_INDEX
          return error

        # check if tokens are valid characters
        tokens = parts[2:]
        validate_tokens = lambda token: re.match('^[\w-]+$', token) is not None
        valid = reduce(lambda x,y: x and y, map(validate_tokens, tokens))
        if not valid:
          error = Error_ID.ERR_TOKENS_NON_ALPHA
          return error

      else:
        # validate query command
        expression = parts[1]
        # check if expression contains valid parenthesis and tokens
        error = validate_parenthesis_and_tokens(expression)

    else:
      error = Error_ID.ERR_INVALID_COMMAND
  return error      
