# _*_ coding:utf-8 _*_

import numpy as np
import os, sys


def firstCharacterHash(tstr):
    counts = {}
    order = []
    for s in tstr:
        if s in counts:
            counts[s] += 1
        else:
            counts[s] = 1
            order.append(s)
    for i in order:
        if counts[i] == 1:
            return i


st = np.array([-2, 0, 9,20,-2, 0, 9])
print('input:',st)
print(firstCharacterHash(st))

st = [bin(i).replace('0b','').zfill(32) for i in st]
find = 0
count=0
for i in range(31,0,-1):
    sum_cur = sum([int(j[i]) for j in st])
    find += (sum_cur % 2) * (2 ** count)
    count += 1
print(find)
ss=list(range(1,10))
print(ss)
print(ss[:-1])
print(ss[1:])

class Solution:
    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        length=len(nums)
        if k >length:
            step=k % length
        else:
            step=k
        nums[length-step:]+nums[:length-step]

a=[1,2]
print(a[1:]+a[:1])
print(3%7)

a.pop(0)
print(a)
nums1=[1,2,3,4,4,5,6]
nums2=[7,4,4,9,999]
import collections
print(list((collections.Counter(nums1) & collections.Counter(nums2)).elements()))
print(list((collections.Counter(nums1) & collections.Counter(nums2)).values()))
print(list((collections.Counter(nums1) & collections.Counter(nums2)).keys()))
# dict1={'jj':1}
# print(list(dict1.elements()))