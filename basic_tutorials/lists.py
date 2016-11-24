numbers = []
strings = []
names = ["John", "Eric", "Jessica"]

numbers.append(1)
numbers.append(2)
numbers.append(3)

strings.append("hello")
strings.append("world")

second_name = names[1]

print(numbers)
print(strings)

for x in numbers:
    print x
for y in strings:
    print y
print("The second name in the names list is %s" % second_name)
