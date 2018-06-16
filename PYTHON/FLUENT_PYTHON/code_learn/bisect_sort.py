import bisect
import sys

hystack = [1, 4, 6, 8, 12, 15, 20, 21, 23, 23, 26, 29, 30]
needles = [0, 1, 2, 5, 8, 10, 22, 23, 29, 30, 31]
row_format = '{0:2d}@{1:2d}       {2}{0:<2d}'


def bisect_model(bisect_fn):
    for needle in reversed(needles):
        position = bisect_fn(hystack, needle)
        offset = position * '  |'
        print(row_format.format(needle, position, offset))
        


if __name__ == '__main__':
    if sys.argv[-1] == 'left':
        bisect_fn = bisect.bisect_left
    else:
        bisect_fn = bisect.bisect

    print('DEMO:', bisect_fn.__name__)
    

    main()
