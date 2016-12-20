phonebook = {}
phonebook["John"] = 938477566
phonebook["Jack"] = 938377264
phonebook["Jill"] = 947662781

phonebook2 = {
        "John" : 938477566,
        "Jack" : 938377264,
        "Jill" : 947662781
        }

for name, number in phonebook.iteritems():
        print "Phone number of %s is %d" % (name, number)

del phonebook2["Jack"]

phonebook2.pop("John")

phonebook["Jake"] = 938273443
phonebook.pop("Jill")

if "Jake" in phonebook:
    print "Jake is listed in the phonebook."

if "Jill" not in phonebook:
    print "Jill is not listed in the phonebook."
