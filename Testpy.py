myFruits = dict(apple="red", durian="green", grape="purple")

# Remove apple
myFruits.pop("apple")
print("After removing apple: ", myFruits, "\n")
# Add orange
myFruits["orange"] = "orange"
print("After adding orange: ", myFruits, "\n")
# How many elements?
print("Number of elements: ", len(myFruits), "\n")
# Create a shallow copy
shallow_copy = myFruits
print("Shallow copy: ", shallow_copy, "\n")
# Create a deep copy
deep_copy = myFruits.copy()
print("Deep copy: ", deep_copy)
