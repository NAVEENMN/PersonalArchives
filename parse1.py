cmd = str(input("cmd: "))

class node():
  def __init__(self):
    self.op = None
    self.term = None
    self.left = None
    self.right = None

  def set_op(self, op):
    self.op = op

  def set_term(self, term):
    self.term = term
  
  def set_left(self, child):
    self.left = child
  
  def set_right(self, child):
    self.right = child

class op_tree():
  def __init__(self):
    self.root = None
    
  def insert(self, op, left, right):
    _node = node()
    _node.set_left(left)
    _node.set_op(op)
    _node.set_right(right)
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

def parse(cmd):
  op_stk = []
  term_stk = []
  cnt = 0
  term = ""
  _op_tree = op_tree()
  for x in range(0, len(cmd)):
    if cmd[x] == "(" or cmd[x] == ")":
      if cmd[x] == "(":
        cnt += 1
      else:
        cnt -= 1
        if term != "":
          term_stk.append(term)
          term = ""
        print(cmd[x], cnt, term_stk, op_stk)
        right = term_stk.pop()
        left = term_stk.pop()
        node = _op_tree.insert(op_stk.pop(), left, right)
        term_stk.append(node)
    else:
      if cmd[x] == "&" or cmd[x] == "|":
        if term != "":
          term_stk.append(term)
          term = ""
        op_stk.append(cmd[x])
      else:
        term += cmd[x]
    print(cmd[x], cnt, term_stk, op_stk)
  print(_op_tree.root)
  _op_tree.print_tree(_op_tree.root)
  
def main():
  parse(cmd)

if __name__ == "__main__":
  main()
