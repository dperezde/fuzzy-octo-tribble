# import the library 
import urllib

# import bar from the foo_package package, in the subdirectory foo_package. In
# order to make it a package we need to have the __init__.py file
from foo_package import bar
#or import foo.bar -> With this othe syntax we need to call foo.bar everytime we
#want to use it.

import re

# use it: dir(urllib) works when on a python prompt
dir(urllib)

help(urllib.urlopen)

functions = dir(re)
find_functions = []
for f in functions:
    if "find" not in f:
        continue
    else:
        find_functions.append(f)

print find_functions

print sorted(find_functions)

