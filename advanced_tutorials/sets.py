print set("my name is Eric and Eric is my name".split())

a = set(["Jake", "John", "Eric"])
b = set(["John", "Jill"])

print(a.intersection(b))

print(b.intersection(a))

print(a.symmetric_difference(b)) # a xor b

print(b.symmetric_difference(a))

print(b.difference(a)) # (a xor b) and a

print(a.difference(b))


print(a.union(b)) # a or b


# Exercise print all participants of A that didn't attend B
print(a.difference(b))

