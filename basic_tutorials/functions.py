def my_function():
    print "Hello from my_function()!"

def my_arg_function(username, greeting):
    print "Hello %s , from My function, I wish you %s"%(username,greeting)

def sum_two(a, b):
    return a + b


my_function()
my_arg_function("bitherm", "go fuck yourself")
x = sum_two(2, 1)

# Modify this function to return a list of strings as defined above
def list_benefits():
    list = ["More organized code", "More readable code", "Easier code reuse",
            "Allowing programmers to share and connect code together"]
    return list

# Modify this function to concatenate to each benefit - " is a benefit of
# functions!"
def build_sentence(benefit):
    sentence = benefit + " is a benefit of functions!"
    return sentence

def name_the_benefits_of_functions():
    list_of_benefits = list_benefits()
    for benefit in list_of_benefits:
        print build_sentence(benefit)

name_the_benefits_of_functions()

