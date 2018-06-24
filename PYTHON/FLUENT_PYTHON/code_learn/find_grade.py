import bisect


def grade(score, break_points=[60, 70, 80, 90], grades='FDCBA'):
    i = bisect.bisect(break_points, score)
    return grades[i]


row_fm = '[score:{}]-----[grade:{}]'
scores = [33, 44, 55, 66, 77, 88, 99, 77, 100]
for score in scores:
    print(row_fm.format(score, grade(score)))


import random
size = 10
random.seed(1729)
my_list = []
for i in range(size):
    new_item = random.randrange(size * 2)
    bisect.insort(my_list, new_item)
    print('%2d-->' % new_item, my_list)
