def do_stuff_with_number(n):
    print n

the_list = (1, 2, 3, 4, 5)

for i in range(20):
    try:
        do_stuff_with_number(the_list[i])

    except IndexError: # Raised when accessing a non-existing index of a list
        do_stuff_with_number(0)

# Exercise 
actor= {"name": "John Cleese", "rank":"awesome"}

def get_last_name():
    name = actor["name"].split()
    return name[1]
#    return actor["last name"]

get_last_name()
print "All exceptions are caught! Good job!"
print "The actor's last name is %s" % get_last_name()
