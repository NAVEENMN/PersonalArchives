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

    

