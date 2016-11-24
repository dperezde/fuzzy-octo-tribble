def transmit_to_space(message):
    "This is the enclosing function"
    def data_transmitter():
        "The nested Function"
        print(message)


    data_transmitter()

def print_msg(number):
    def printer():
        "here we are using the nonlocal keyword"
#        nonlocal number # Without the nonlocal keyword the output would be 3 9
        number = 3
        print(number)
        
    printer()
    print(number)

print_msg(9)


#fun2 = transmit_to_space("Burn the sun!")

#fun2()

# Exercise: Make a nested loop and a python closure to make functions to get
# multiple multiplication functions using closures. That is using closures, one
# could make functions to create multiply_with_5() or multiply_with_4()
# functions using closures.

def multiplier_of(n):
    def multiply(i):
        return n*i
    return multiply

multiplywith5 = multiplier_of(5)
print(multiplywith5(9))
