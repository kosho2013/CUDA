a = [-1,-1,6,1,9,3,2,-1,4,-1]
print(a)


myset = set()

for ele in a:
    myset.add(ele)

for i in range(len(a)):
    if i in myset:
        a[i] = i
    else:
        a[i] = -1

print(a)
