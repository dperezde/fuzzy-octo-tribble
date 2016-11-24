def decorator(some_f):
    pass

@decorator
def function(arg):
    return "Return"

# Otra manera de escribirlo es:
# def function(arg):
#   return "Return"
# function = decorator(function)

def repeater(old_f):
    def new_f(*args, **kwds):
        old_function(*args, **kwds)
        old_function(*args, **kwds)
    return new_f


@repeater
def Multiply(n1, n2):
    print n1*n2

def my_decorator(some_f):
    def wrapper():
        print ("Something happens before some_f() is called")

        some_f()

        print ("Something happens after some_f() is called")

    return wrapper




@my_decorator
def just_some_function():
    print("Whee!")


just_some_function()

# Exercise Make a decorator factory which returns a decorator that decorates
# functions with one argument. The factory should take one argument, a type, and
# then returns a decorator that makes function should check if the input is the
# correct type. If it is wrong, it should print "Bad Type". (In reality, it
# should raise an error, but error raising isn't in this tutorial.) Look at the
# tutorial code and expected output to see what it is if you are confused (I
# know I would be.) Using isinstance(object, type_of_object) or type(object)
# might help.

def type_check(correct_type):
    def wrapper(some_f):
        def check_arg(arg):
            if isinstance(arg, correct_type):
                return some_f(arg)
            else:
                print ("Bad Type")
        return check_arg
    return wrapper




@type_check(int)
def times2(num):
    return num*2

print times2(2)
times2('Not a Number')

@type_check(str)
def first_letter(word):
    return word[0]

print first_letter('Hello World')
first_letter(['Not', 'A', 'String'])
