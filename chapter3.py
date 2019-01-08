tup=4,5,6

tup

nested_tup=(4,5,6),(7,8)
nested_tup

tuple([4,0,2])

tup=tuple('string')
tup

tup[0]

tup=tuple(['foo',[1,2],True])

tup[1].append(3)

(4,None,'foo')+(6,0)+('bar',)

('foo','bar')*4

tup=(4,5,6)
a,b,c=tup

tup=(4,5,(6,7))
a,b,(c,d)=tup

values=1,2,3,4,5
a,b,*rest=values

a,b,*_=values

a=(1,2,3,4,3,2,3)
a.count(2)

#-------------------------------------------
a_list=[2,3,7,None]
tup=('foo','bar','baz')
b_list=list(tup)
b_list[1]='peekaboo'
b_list

gen=range(10)
gen
list(gen)

b_list.append('dwarf')

b_list

b_list.insert(1,'red')

b_list

b_list.pop(2)
b_list

b_list.append('foo')
b_list

b_list.remove('foo')
b_list

'dwarf' in b_list

[4,None,'foo']+[7,8,(2,3)]

x=[4,None,'foo']
x.extend([7,8,(2,3)])
x

a=[7,2,5,1,3]
a.sort()
a

b=['saw','small','He','foxes','six']
b.sort(key=len)
b

import bisect
c=[1,2,2,2,3,4,7]
bisect.bisect(c,2)
bisect.bisect(c,5)
bisect.insort(c,6)

seq=[7,2,3,7,5,6,0,1]
seq[1:5]

seq[3:4]=[6,3]
seq

seq[:5]
seq[3:]

seq[::2]
seq[::-1]

some_list=['foo','bar','baz']
mapping={}
for i,v in enumerate(some_list):
    mapping[v]=i

mapping

sorted([7,1,2,6,0,3,2])

sorted('horse race')

seq1=['foo','bar','baz']
seq2=['one','two','three']
zipped=zip(seq1,seq2)
list(zipped)


seq3=[False,True]
list(zip(seq1,seq2,seq3))

for i,(a,b) in enumerate(zip(seq1,seq2)):
    print('{0}:{1},{2}'.format(i,a,b))

pitchers=[('Nolan','Ryan'),('Roger','Clemens'),('Schilling','Curt')]
first_names,last_names=zip(*pitchers)
first_names
last_names

list(reversed(range(10)))

#-------------------------------------------------
empty_dict={}
d1={'a':'some value','b':[1,2,3,4]}
d1

d1[7]='an integer'

d1
d1['b']

'b' in d1

d1[5]='some value'

d1
d1['dummy']='another value'

d1

del d1[5]

d1

ret=d1.pop('dummy')

ret

d1

list(d1.keys())

list(d1.values())

d1.update({'b':'foo','c':12})
d1


mapping={}
for key,value in zip(key_list,value_list):
    mapping[key]=value


mapping=dict(zip(range(5),reversed(range(5))))
mapping

words=['apple','bat','bar','atom','book']
by_letter={}

for word in words:
    letter=word[0]
    if letter not in by_letter:
        by_letter[letter]=[word]
    else:
        by_letter[letter].append(word)

by_letter

hash('string')
hash((1,2,(2,3)))
#hash((1,2,[2,3]))

#---------------------------------
set([2,2,2,1,3,3])

{2,2,2,1,3,3}

a={1,2,3,4,5}
b={3,4,5,6,7,8}
a|b
a&b
#-------------------------------








