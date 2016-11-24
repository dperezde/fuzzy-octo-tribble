import json
import cPickle

json_string = json.dumps([1, 2, 3, "a", "b", "c"])

print json.loads(json_string)

pickled_string = cPickle.dumps([1, 2, 3, "a", "b", "c"])
print cPickle.loads(pickled_string)



# Exercise The aim of this exercise is to print out the JSON string with
# key-value pair "Me" : 800 added to it.

def add_employee(salaries_json, name, salary):
    salaries_dict = json.loads(salaries_json)
    salaries_dict[name] = salary
    salaries_json = json.dumps(salaries_dict)

    return salaries_json

#test code
salaries = '{"Alfred" : 300, "Jane" : 400}'

new_salaries = add_employee(salaries, "Me", 800)

decoded_salaries = json.loads(new_salaries)
print decoded_salaries["Alfred"]
print decoded_salaries["Jane"]
print decoded_salaries["Me"]

