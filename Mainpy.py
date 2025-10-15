import sys
import threading
import collections
import heapq
import bisect
import math
import itertools
import functools

# Fast input


def input(): return sys.stdin.readline().strip()
# Fast output
def print(x): return sys.stdout.write(str(x) + '\n')


# Constants
INF = float('inf')
MOD = 10**9 + 7
YES, NO = "YES", "NO"
Yes, No = "Yes", "No"

# --------------------- Solution ---------------------


def solve():
    x1, y1, x2, y2 = map(int, input().split())
    if ((x1 >= x2 and y1 >= y2) or (x1 <= x2 and y1 <= y2)):
        print(NO)
        return
    yes = True
    if (x1 < x2):
        while (x1 != x2):
            x1 += y1
            x2 += y2
            if (x1 > x2):
                yes = False
                break
    else:
        while (x1 != x2):
            x1 += y1
            x2 += y2
            if (x1 < x2):
                yes = False
                break
    print(YES if yes else NO)


def main():
    # Uncomment if you need to read number of test cases
    # t = int(input())
    # for _ in range(t):
    #    solve()

    # Or just call solve() once
    solve()

# Use threading to increase recursion limit (if needed for DFS, etc.)


# Increase recursion limit (only if needed)
# sys.setrecursionlimit(1 << 25)

# Multithreaded execution (optional, useful for deep recursion)
# threading.Thread(target=main).start()

if __name__ == "__main__":
    main()
