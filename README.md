## simple search engine
```
Tested on Python 3.6.1
Usage: python main.py

# example to add index:

in: index 1 salt pepper fish salt salt salt
out: index ok 1
in: index 2 salt butter sugar sugar milk butter water water
out: index ok 2
in: index 3 pepper oil salt salt water corn mint cream
out: index ok 3
in: index 4 oil butter fish water water water soup lemon
out: index ok 4
in: index 5 ginger salt sugar pepper salt pepper pepper soup
out: index ok 5

# example to query:

in: query salt
out: results 1 3 5 2
in:
in: query (salt|pepper)
out: results 5 1 3 2
in:
in: query ((salt|water)&(sugar|pepper)
out:  error invalid expression provided,  [use brackets concisely]
in: query ((salt|water)&(sugar|pepper))
out: results 2 5 3 1
in:
in: query (((mint)))
out: error invalid expression provided,  [use brackets concisely]
in:
in: query mint
out: results 3
in:
in: query (((salt|mint)|(mint&water)|fish)|pepper)
out: results 3 4 1 5 2
in:
in: query salt|water|oil|pepper
out: results 3 4 5 2 1
in:
in: query avacado
out: results
in:
in: q

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
out:  error invalid tokens provided

```
