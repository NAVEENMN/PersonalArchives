## simple search engine
```
Tested on Python 3.6.1
Usage: python main.py

# example to add index:

in: index 1 salt pepper fish
out: index ok  1
in: index 2 salt butter sugar
out: index ok  2
in: index 3 pepper oil salt
out: index ok  3

# example to query:

in: query salt
out: results  {1, 2, 3}
in: query (salt&pepper)
out: results  {1, 3}
in: query ((salt|pepper)&(salt&butter))
out: results  {2}
in: query (butter&pepper)
out: results {}
in: query ((salt&pepper)|(sugar))
out: results  {1, 2, 3}
in: query ((salt|pepper)&butter)
out: results  {2}
in: query ((salt|pepper)&garlic)
out: results  {}
in: q

exiting

# Errors

in: query {}
out:  error no tokens were provided
in: query (salt|p7eer)
out:  error expression contains invalid tokens
in: query
out:  error no expression provided
in: in
out:  error invalid input (valid: index | query)
in: index hsad sada
out:  error invalid document index
in: index 4 g%#$
out:  error tokens with non alphanumeric provided

```
