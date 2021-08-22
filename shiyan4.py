import cv2
# import numpy as np
# def list_of_groups(init_list,n):
#     list_of_groups=zip(*(iter(init_list),)*n)
#     for j in list_of_groups:
#         print(j)
#     end_list=[list(i) for i in list_of_groups] # i is a tuple
#     count = len(init_list)%n
#     end_list.append(init_list[-count:]) if count != 0 else end_list
#     return end_list
#
# a = [6,6,6,7,7,7]
# a = np.array(a)
# b = np.zeros((2,2,2))
# c = list_of_groups(a,2)
# print(c)
a = [2,3,4,5]
c = [1,1,1]
e = iter(a)
d = (iter(a),)*2
r = (2,)*2
o = ([1,2],[3,4])
b = zip(*o)
for i in b:
    print(i)