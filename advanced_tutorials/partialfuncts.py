from functools import partial

# create a new function that multiplies by 2
def multiply(a, b):
    return a*b

dbl = partial(multiply,2)

print dbl(4)

def func(u, v, w, x):
    return u*4 + v*3 + w*2 + x

p = partial(func, 5, 6, 7)

p(8)
