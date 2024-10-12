import heapq
n=3
lis=[2,4,5,1,7]
re1 = map(lis.index, heapq.nlargest(n, lis)) #求最大的n个索引    nsmallest 求最小  nlargest求最大
re2 = heapq.nlargest(n, lis) #求最大的三个元素
print(list(re1)) #因为re1由map()生成的不是list，直接print不出来，添加list()就行了
print(re2) 
