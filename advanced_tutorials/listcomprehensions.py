sentence = "the quick brown fox jumps over the lazy dog"

words = sentence.split()

word_length = []

# without list comprehensions
for word in words:
    if word != "the":
        word_length.append(len(word))


# with list comprehensions
word_lengths_compr = [len(word) for word in words if word != "the"]

# Exercise Using a list comprehension, create a new list called "newlist" out of
# the list "numbers", which contains only the positive numbers from the list, as
# integers.

numbers = [34.6, -203.4, 44.9, 68.3, -12.2, 44.6, 12.7]
newlist = [int(n) for n in numbers if n >= 0]

print newlist




