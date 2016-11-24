def foo(first, second, third, *therest):
    print "First: %s" % first
    print "Second: %s" % second
    print "Third: %s" % third
    print "And all the rest... %s" % list(therest)


foo(1,2,3,4,5,6,7,8)


# Args by keyword
def bar(first, second, third, **options):
    if options.get("action") == "sum":
        print "The sum is: %d" % (first + second + third)

    if options.get("number") == "first":
        return first

result = bar(1, 2, 3, action = "sum", number = "first")

print "Result is %d" % result



# Exercise: Fill in the foo and bar functions so they can receive a variable
# amount of arguments (3 or more) The foo function must return the amount of
# extra arguments received. The bar must return True if the argument with the
# keyword magicnumber is worth 7, and False otherwise.
def foo_ex(a, b, c, *therest):
    return len(therest)

def bar_ex(a, b, c, **therest):
    if therest.get("magicnumber") == 7:
        return True
    else:
        return False
    


if foo_ex(1, 2, 3, 4) == 1:
    print "Good"

if foo_ex(1, 2, 3, 4, 5) == 2:
    print "Better"
if bar_ex(1, 2, 3, magicnumber = 6) == False:
    print "Great"
if bar_ex(1, 2, 3, magicnumber = 7) == True:
    print "Awesome"

