import types
import random
import collections

# lottery is the generator
def lottery():
    # returns 6 numbers between 1 and 40
    for i in xrange(6):
        yield random.randint(1,40)

    # returns a 7th number between 1 and 15
    yield random.randint(1,15)


for random_number in lottery():
    print "And the next number is... %d" % random_number


# Exercise: Write a generator function which returns the Fibonacci series

def fibonacci():
    a_b = collections.deque([0]*2,2)
    while True:
        if 0 in a_b:
            a_b.append(1)
            yield 1
        else:
            c = a_b[0] + a_b[1]
            yield c
            a_b.append(c)
            

if type(fibonacci()) == types.GeneratorType:
    print "Good, the fibonacci function is a generator"

    counter = 0
    for n in fibonacci():
        print n
        counter +=1
        if counter == 20:
            break


