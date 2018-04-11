import re
from functools import reduce
from enums import *


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
          error = Error_ID.ERR_NO_EXP
          return response, error
        
        response["expression"] = parts[1:]
    else:
      error = Error_ID.ERR_INVALID_COMMAND
  return response, error
