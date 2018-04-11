import re
from functools import reduce
from enums import *


def tokenize_expression(expression):
  error = Error_ID.OK
  stk = list()
  token = ""
  cnt = 0
  exp = str(expression[0])
  for x in range(0, len(exp)):
    if exp[x] == "(":
      cnt += 1
    else:
      if exp[x].isdigit():
        error = Error_ID.ERR_INVALID_TOKENS_EXP
        return token, error
      if exp[x] == "&" or exp[x] == "|":
        token += exp[x]
        continue
      if exp[x] == ")":
        stk.append(token)
        token = ""
      else:
        if not validate_tokens(exp[x]):
          error = Error_ID.ERR_INVALID_TOKENS_EXP
          return token, error
        token += exp[x]
  return stk, error

def validate_query_parenthesis(expression):
  error = Error_ID.OK
  stk = list()
  exp = str(expression[0])
  cnt = 0
  for x in range(0, len(exp)):
    if exp[x] == "(":
      cnt += 1
      stk.append(exp[x])
    else:
      if exp[x] == ")":
        stk.pop()
        cnt -= 1
  if (cnt != 0):
    error = Error_ID.ERR_INVALID_EXP
  return error


def validate_tokens(tokens):
  valid = re.match('^[\w-]+$', tokens) is not None
  return valid

def get_input():
  error = Error_ID.OK
  response = dict()
  command = str(input("in: "))
  
  if command == "q":
    return command, error

  if command == "dsp":
    response["action"] = ACTION.DISPLAY
    return response, error
  
  if len(command) == 0:
    error = Error_ID.ERR_EMPTY_INPUT
  else:
    parts = command.split(" ")
    if (parts[0] == "index") or (parts[0] == "query"):
      # ---- validate index input ----
      if (parts[0] == "index"):
        response["action"] = ACTION.INDEX
        
        if len(parts) < 2:
          error = Error_ID.ERR_NO_DOC_ID
          return response, error
        
        if parts[1].isdigit():
          response["doc_idx"] = int(parts[1])
        else:
          error = Error_ID.ERR_INVALID_DOC_INDEX
          return response, error
        
        if len(parts) < 3:
          error = Error_ID.ERR_NO_TOKENS
          return response, error

        # validate tokens
        valid = reduce(lambda x,y: x and y, map(validate_tokens, parts[2:]))
        
        if not valid:
          error = Error_ID.ERR_TOKENS_NON_ALPHA
          return response, error

        response["tokens"] = parts[2:]
      
      else:
      # ---- validate query input ----
        action = ACTION.QUERY
        expression = parts[1:]

        response["action"] = ACTION.QUERY
        
        # validate expression and handle error
        if len(expression) == 0:
          error_id = Error_ID.ERR_NO_EXP
          return response, error_id

        f = lambda x: True if ((ord(x)>=65 and ord(x)<90)or(ord(x)>=97 and ord(x)<122)) else False
        is_char = reduce(lambda x,y: x and y, map(f,list(expression[0])))

        if is_char:
          response["single"] = True
          response["expression"] = expression
        else:
          response["single"] = False
          error_id = validate_query_parenthesis(expression)

          if error_id != Error_ID.OK:
            return response, error_id

          tokens, error_id = tokenize_expression(expression)
          tokens = list(filter(None, tokens))
          if error_id != Error_ID.OK:
            return response, error_id
          response["expression"] = tokens
    else:
      error = Error_ID.ERR_INVALID_COMMAND
  return response, error
