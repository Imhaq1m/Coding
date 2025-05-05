#Exercise 1
print ("Hello, Python!\n")

#Excersise 2
a = int(input('Enter first number: '))
b = int(input('Enter second number: '))
sum = a+b
print("Sum:",sum,"\n")

#Exercise 3
r = int(input("Enter radius of the circle: "))
area = 3.14 * pow(r, 2)
print("Area of the cirlce:",area,"\n")

#Exercise 4
a = int(input("Enter a number: "))
if(a%2==0):
    print("The number is even")
else:
    print("The number is odd")
print("\n")

#Exercise 5
a = int(input("Enter first number: "))
b = int(input("Enter second number: "))
c = int(input("Enter third number: "))
print("The largest number:", max(a, b ,c))

#Excercise 6
n = int(input("Enter an number: "))
for i in range(1,11):
    print(n,"x",i,"=",n*i)
print("\n")

#Exercise 7
vowels = ['a', 'e', 'i', 'o', 'u']
c = int(0)
s = str(input("Enter a string: "))
for i in range(len(s)):
    if(s[i] in vowels):
        c+=1
print("Number of vowels:",c,"\n")

#Exercise 8
l = map(int, input("Enter numbers separated by space: ").split())
l = list(l)
print("Original list:",l)
rev = l[::-1]
print("Reverse list:",rev)
print("\n")

a = int(input("Enter a number: "))
sum = int(1)
for i in range(a, 0, -1):
    sum*=i
print("Factorial:",sum,"\n")


#Exercise 10
l = list(map(int, input("Enter numbers separated by space: ").split()))
sum = int(0)
for i in range(len(l)):
    sum+=l[i]
print("Sum of list items:",sum)