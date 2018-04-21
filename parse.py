import re
from functools import reduce

cmd = str(input("cmd: "))

def ope(cmd, op):
  print("operate", op)
  if op == "&":
    res = "&".join(cmd)
    print(cmd, "&")
    return res
  else:
    res = "|".join(cmd)
    print(cmd, "|")
  return res

def parse(cmd):
  print("parse:", cmd)
  if ("(" in cmd):
    result= []
    reh = re.findall(r'((.*?))', cmd)
    print(reh)
    exit()
    pattern = r"\(([^(\)]*)\)"
    while '(' in cmd:
      result.extend(re.findall(pattern, cmd))
      cmd = re.sub(pattern, '', cmd)
    result = list(filter(None, (t.strip() for t in result)))
    operations = list(result.pop())
    print("result", result, operations)
    to_reduce = list()
    for x in range(1, len(result)):
      red = [result[x-1], result[x], operations.pop()]
      to_reduce.append(red)
    reduced = reduce(lambda x,y: parse(x)+parse(y), result)
    print("reduced", list(reduced))
    res = map(lambda tr, op: ope(tr, op),reduced, operations)
    print("mapped", list(res))
  if ("&" in cmd):
    res = []
    parts = cmd.split("&")
    resa = parse(parts[0])
    resb = parse("&".join(parts[1:]))
    res.extend(resa)
    res.extend(resb)
    return res
  elif ("|" in cmd):
    res = []
    parts = cmd.split("|")
    resa = parse(parts[0])
    resb = parse("|".join(parts[1:]))
    res.extend(resa)#process
    res.extend(resb)
    return res
  else:
    return [cmd]

def main():
  result = parse(cmd)
  print(result)

if __name__ == "__main__":
  main()
