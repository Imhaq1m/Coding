#Exercise 1
print("Hello, World!")
print("\n")
#Exercise 2

# Define variables
name = "Alice" # String
age = 25 # Integer
height = 5.5 # Float
is_student = True # Boolean
# Print variables
print("Name:", name)
print("Age:", age)
print("Height:", height)
print("Is Student:", is_student)
print("\n")

#Exercise 3
# Perform arithmetic operations
a = 10
b = 3
print("Addition:", a + b)
print("Subtraction:", a - b)
print("Multiplication:", a * b)
print("Division:", a / b)
print("Modulus (remainder):", a % b)
print("\n")

#Exercise 4

#Get input from user
number = int(input("Enter a number: "))

#Conditional statements
if number > 0:
  print("The number is positive.")
elif number < 0:
  print("The number is negative.")
else:
  print("The number is zero.")
print("\n")

#Exercise 5

for i in range(1,6):
  print(i)
print("\n")

#Exercise 6
def add_numbers(a, b):
    return a + b
# Call the function
result = add_numbers(10, 5)
print("Sum:", result)
print("\n")

#Exercise 7
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)
