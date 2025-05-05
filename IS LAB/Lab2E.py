import math
import random
import numpy as np
import sys
'''
#Exercise 1
def create_list():
    l = list(map(int, input("Enter numbers separated by space: ").split()))
    return l
res = create_list()
print("List:",res,"\n")

#Exercise 2
def sum_of_list(lst):
    sum = int(0)
    for i in range(len(lst)):
        sum += lst[i]
    return sum
input = list(map(int, input().split()))
print("Sum of list elements:",sum_of_list(input),"\n")

#Exercise 3
def reverse_list(lst):
    rev = lst[::-1]
    return rev
input = list(map(int, input().split()))
print("Reversed list:", reverse_list(input),"\n")

#Exercise 4
def squares_list(n):
    lst = []
    for i in range(1, n+1, 1):
        lst.append(pow(i, 2))
    return lst
n = int(input())
print("List of squares:",squares_list(n), "\n")

#Exercise 5
def find_max_min(lst):
    h = max(lst)
    l = min(lst)
    return h, l
lst = list(map(int, input().split()))
high, low = find_max_min(lst)
print("Maximum:",high)
print("Minimum:",low)
print("\n")

#Exercise 6
def count_occurrences(lst, element):
    count = int(0)
    for i in range(len(lst)):
        if lst[i] == element:
            count+=1
    return count
lst = list(map(int, input().split()))
element = int(input("Enter the element to count: "))
ans = count_occurrences(lst, element)
print(element,"occurs",ans,"times\n")

#Exercise 7
def remove_all(lst, element):
    lst.remove(element)
    return lst
lst = list(map(int, input().split()))
element = int(input("Enter the element to remove: "))
print("List after removal:",remove_all(lst, element),"\n")

#Exercise 8
def is_sorted(lst):
    sort = lst==sorted(lst)
    return sort
lst = list(map(int, input().split()))
print("Is the list sorted?:",is_sorted(lst),"\n")

#Exercise 9
def uppercase_list(lst):
    for i in range(len(lst)):
       lst[i] = lst[i].upper()
    return lst
lst = list(map(str, input().split()))
result = uppercase_list(lst)
print("Uppercase List:", result,"\n")

#Exercise 10
def factorial_list(lst):
    for i in range(len(lst)):
        lst[i] = math.factorial(lst[i])
    return lst
lst = list(map(int, input().split()))
res = factorial_list(lst)
print("Factorial of list elements:",res,"\n")

#Exercise 11
def random_list(n):
    lst = []
    for i in range(n):
        lst.append(random.randint(1,100))
    return lst
n = int(input())
print("Random list:",random_list(n),"\n")

#Exercise 12
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
def find_primes(lst):
    return [x for x in lst if is_prime(x)]
lst = list(map(int, input().split()))
prime = find_primes(lst)
print("Prime numbers in the list:", prime,"\n")

#Exercise 13
def unique_elements(lst):
    unique_set = set(lst)
    return list(unique_set)
lst = list(map(int, input().split()))
res = unique_elements(lst)
print("Unique elements:", res,"\n")

#Exercise 14
def flatten_2d_list(lst):
    newlst = []
    for i in lst:
        for j in i:
            newlst.append(j)
    return newlst
lst = [[1, 2], [3, 4], [5, 6]]
res = flatten_2d_list(lst)
print("Flattened list:", res,"\n")

#Exercise 15
def sort_tuples(lst):
    for i in range(len(lst)):
        for j in range(0, len(lst) - i - 1):
            if lst[j][1] > lst[j + 1][1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
    return lst
lst = [(1, 3), (4, 1), (5, 2)]
print("Sorted tuples:", sort_tuples(lst))
'''

#Exercise 16
arr = np.array([1, 2, 3, 4, 5])
add = arr + 2
sub = arr - 2
mul = arr * 2
div = arr / 2
print("Original array:", arr)
print("Array + 2:", add)
print("Array - 2:", sub)
print("Array * 2:", mul)
print("Array / 2:", div)

#Exercise 17
