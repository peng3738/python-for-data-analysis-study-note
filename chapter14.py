path='E:python_study/python for data analysis/pydata-book-2nd-edition/datasets/bitly_usagov/example.txt'
open(path).readline()
import json
records=[json.loads(line) for line in open(path)]
records[0]
import matplotlib.pyplot as plt

#time_zones=[rec['tz'] for rec in records]
time_zones=[rec['tz']for rec in records if 'tz' in rec]
time_zones[:10]

def get_counts(sequence):
    counts={}
    for x in sequence:
        if x in counts:
            counts[x]+=1
        else:
            counts[x]=1
    return counts

from collections import defaultdict

def get_counts2(sequence):
    counts=defaultdict(int)
    for x in sequence:
        counts[x]+=1
    return counts

counts=get_counts(time_zones)
counts['America/New_York']
len(time_zones)

def top_counts(count_dict,n=10):
    value_key_pairs=[(count,tz) for tz, count in count_dict.items()]
    value_key_pairs.sort()
    return value_key_pairs[-n:]

top_counts(counts)

from collections import Counter
counts=Counter(time_zones)
counts.most_common(10)

import pandas as pd
frame=pd.DataFrame(records)
frame.info()
frame['tz'][:10]
tz_counts=frame['tz'].value_counts()
tz_counts[:10]

clean_tz=frame['tz'].fillna('Missing')
clean_tz[clean_tz=='']='Unknown'
tz_counts=clean_tz.value_counts()

import seaborn as sns
subset=tz_counts[:10]
sns.barplot(y=subset.index,x=subset.values)

plt.figure()

frame['a'][1]
frame['a'][50]
frame['a'][51][:50]

results=pd.Series([x.split()[0]for x in frame.a.dropna()])
results[:5]
results.value_counts()[:8]

cframe=frame[frame.a.notnull()]
cframe=cframe.copy()
cframe['os']=np.where(cframe['a'].str.contains('Windows'),'Windows','Not Windows')

cframe['os'][:5]

by_tz_os=cframe.groupby(['tz','os'])

agg_counts=by_tz_os.size().unstack().fillna(0)
agg_counts[:10]

indexer=agg_counts.sum(1).argsort()
indexer[:10]

count_subset=agg_counts.take(indexer[-10:])
count_subset

agg_counts.sum(1).nlargest(10)
count_subset=count_subset.stack()
count_subset.name='total'
count_subset=count_subset.reset_index()
count_subset[:10]
sns.barplot(x='total',y='tz',hue='os',data=count_subset)
plt.show()

def norm_total(group):
    group['normed_total']=group.total/group.total.sum()
    return group

results=count_subset.groupby('tz').apply(norm_total)
sns.barplot(x='normed_total',y='tz',hue='os',data=results)
plt.show()


g=count_subset.groupby('tz')
results2=count_subset.total/g.total.transform('sum')

#--------------------------------
import pandas as pd
pd.options.display.max_rows=10
unames=['user_id','gender','age','occupation','zip']
users=pd.read_table('E:python_study/python for data analysis/pydata-book-2nd-edition/datasets/movielens/users.dat',
                    sep='::',header=None,names=unames)

rnames=['user_id','movie_id','rating','timestamp']
ratings=pd.read_table('E:python_study/python for data analysis/pydata-book-2nd-edition/datasets/movielens/ratings.dat',
                      sep='::',header=None,names=rnames)

mnames=['movie_id','title','genres']
movies=pd.read_table('E:python_study/python for data analysis/pydata-book-2nd-edition/datasets/movielens/movies.dat',
                     sep='::',header=None,names=mnames)


users[:5]
ratings[:5]
movies[:5]
ratings

data=pd.merge(pd.merge(ratings,users),movies)

mean_ratings=data.pivot_table('rating',index='title',columns='gender',aggfunc='mean')
mean_ratings[:5]

ratings_by_title=data.groupby('title').size()
ratings_by_title[:10]

active_titles=ratings_by_title.index[ratings_by_title>=250]
active_titles

mean_ratings=mean_ratings.loc[active_titles]
mean_ratings

top_female_ratings=mean_ratings.sort_values(by='F',ascending=False)
top_female_ratings[:10]

#------------------------------
mean_ratings['diff']=mean_ratings['M']-mean_ratings['F']
sorted_by_diff=mean_ratings.sort_values(by='diff')
sorted_by_diff[:10]

sorted_by_diff[::-1][:10]

rating_std_by_title=data.groupby('title')['rating'].std()
rating_std_by_title=rating_std_by_title.loc[active_titles]
rating_std_by_title.sort_values(ascending=False)[:10]

#-------------------------------------

import pandas as pd
names1880 = pd.read_csv('E:python_study/python for data analysis/pydata-book-2nd-edition/datasets/babynames/yob1880.txt',
                        names=['name', 'sex', 'births'])
names1880

names1880.groupby('sex').births.sum()

years=range(1880,2011)
pieces=[]
columns=['name','sex','births']
for year in years:
    path='E:python_study/python for data analysis/pydata-book-2nd-edition/datasets/babynames/yob%d.txt'%year
    frame=pd.read_csv(path,names=columns)

    frame['year']=year
    pieces.append(frame)

names=pd.concat(pieces,ignore_index=True)

total_births=names.pivot_table('births',index='year',columns='sex',aggfunc='sum')


total_births.tail()

def add_prop(group):
    group['prop']=group.births/group.births.sum()
    return group
names=names.groupby(['year','sex']).apply(add_prop)

names.groupby(['year','sex']).prop.sum()

def get_top1000(group):
    return group.sort_values(by='births',ascending=False)[:1000]

grouped=names.groupby(['year','sex'])
top1000=grouped.apply(get_top1000)
top1000.reset_index(inplace=True,drop=True)

pieces=[]
for year,group in names.groupby(['year','sex']):
    pieces.append(group.sort_values(by='births',ascending=False)[:1000])
top1000=pd.concat(pieces,ignore_index=True)
top1000

boys=top1000[top1000.sex=='M']
girls=top1000[top1000.sex=='F']

total_births=top1000.pivot_table('births',index='year',columns='name',
                                 aggfunc=sum)
total_births.info()

subset=total_births[['John','Harry','Mary','Marilyn']]
subset.plot(subplots=True,figsize=(12,10),grid=False,
            title="Number of births per year")

plt.show()
#---------------------------------------
table=top1000.pivot_table('prop',index='year',columns='sex',aggfunc=sum)
table.plot(title='Sum of table1000.prop by year and sex',
           yticks=np.linspace(0,1.2,13),xticks=range(1880,2020,10))
plt.show()

df=boys[boys.year==2010]
df

prop_cumsum=df.sort_values(by='prop',ascending=False).prop.cumsum()
prop_cumsum[:10]

prop_cumsum.values.searchsorted(0.5)


df=boys[boys.year==1900]
in1900=df.sort_values(by='prop',ascending=False).prop.cumsum()
in1900.values.searchsorted(0.5)+1

def get_quantile_count(group,q=0.5):
    group=group.sort_values(by='prop',ascending=False)
    return group.prop.cumsum().values.searchsorted(q)+1

diversity=top1000.groupby(['year','sex']).apply(get_quantile_count)
diversity=diversity.unstack('sex')

diversity.plot(title="Number of popular names in top 50%")

#----------------------------
get_last_letter=lambda x:x[-1]
last_letters=names.name.map(get_last_letter)
last_letters.name='last_letter'
table=names.pivot_table('births',index=last_letters,columns=['sex','year'],
                        aggfunc=sum)
subtable=table.reindex(columns=[1910,1960,2010],level='year')
subtable.head()
subtable.sum()

letter_prop=subtable/subtable.sum()
letter_prop

import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,1,figsize=(10,8))
letter_prop['M'].plot(kind='bar',rot=0,ax=axes[0],title='Male')
letter_prop['F'].plot(kind='bar',rot=0,ax=axes[1],title='Female',legend=False)

plt.show()

letter_prop=table/table.sum()
dny_ts=letter_prop.loc[['d','n','y'],'M'].T
dny_ts.head()

dny_ts.plot()
plt.show()

all_names=pd.Series(top1000.name.unique())
lesley_like=all_names[all_names.str.lower().str.contains('lesl')]
lesley_like

filtered=top1000[top1000.name.isin(lesley_like)]
filtered.groupby('name').births.sum()

table=filtered.pivot_table('births',index='year',columns='sex',aggfunc='sum')
table=table.div(table.sum(1),axis=0)
table.tail()

table.plot(style={'M':'k-','F':'k--'})
plt.show()

#-------------------
import json
db=json.load(open('E:python_study/python for data analysis/pydata-book-2nd-edition/datasets/usda_food/database.json'))
len(db)

db[0].keys()

db[0]['nutrients'][0]

nutrients=pd.DataFrame(db[0]['nutrients'])
nutrients[:7]

info_keys=['description','group','id','manufacturer']
info=pd.DataFrame(db,columns=info_keys)
info[:5]

info.info()

pd.value_counts(info.group)[:10]

nutrients = []

for rec in db:
    fnuts = pd.DataFrame(rec['nutrients'])
    fnuts['id'] = rec['id']
    nutrients.append(fnuts)

nutrients = pd.concat(nutrients, ignore_index=True)

nutrients

#-------------
nutrients.duplicated().sum()
nutrients=nutrients.drop_duplicates()

col_mapping={'description':'food','group':'fgroup'}
info=info.rename(columns=col_mapping,copy=False)
info.info()

col_mapping={'description':'nutrient','group':'nutgroup'}
nutrients=nutrients.rename(columns=col_mapping,copy=False)
nutrients

ndata=pd.merge(nutrients,info,on='id',how='outer')
ndata.info()

ndata.iloc[3000]

result=ndata.groupby(['nutrient','fgroup'])['value'].quantile(0.5)
result['Zinc, Zn'].sort_values().plot(kind='barh')
plt.show()

by_nutrient=ndata.groupby(['nutgroup','nutrient'])
get_maximum=lambda x:x.loc[x.value.idxmax()]
get_minimum=lambda x:x.loc[x.value.idxmin()]

max_foods=by_nutrient.apply(get_maximum)[['value','food']]
max_foods.food=max_foods.food.str[:50]
max_foods.loc['Amino Acids']['food']


#-------------------------------------------
fec=pd.read_csv('E:python_study/python for data analysis/pydata-book-2nd-edition/datasets/fec/P00000001-ALL.csv')
fec.info()

fec.iloc[123456]

unique_cands=fec.cand_nm.unique()
unique_cands

unique_cands[2]


parties = {'Bachmann, Michelle': 'Republican',
           'Cain, Herman': 'Republican',
           'Gingrich, Newt': 'Republican',
           'Huntsman, Jon': 'Republican',
           'Johnson, Gary Earl': 'Republican',
           'McCotter, Thaddeus G': 'Republican',
           'Obama, Barack': 'Democrat',
           'Paul, Ron': 'Republican',
           'Pawlenty, Timothy': 'Republican',
           'Perry, Rick': 'Republican',
           "Roemer, Charles E. 'Buddy' III": 'Republican',
           'Romney, Mitt': 'Republican',
           'Santorum, Rick': 'Republican'}

fec.cand_nm[123456:123461]
fec.cand_nm[123456:123461].map(parties)

fec['party']=fec.cand_nm.map(parties)
fec['party'].value_counts()

(fec.contb_receipt_amt>0).value_counts()

fec=fec[fec.contb_receipt_amt>0]

fec_mrbo=fec[fec.cand_nm.isin(['Obama, Barack','Romney, Mitt'])]

fec.contbr_occupation.value_counts()[:10]


occ_mapping = {
   'INFORMATION REQUESTED PER BEST EFFORTS' : 'NOT PROVIDED',
   'INFORMATION REQUESTED' : 'NOT PROVIDED',
   'INFORMATION REQUESTED (BEST EFFORTS)' : 'NOT PROVIDED',
   'C.E.O.': 'CEO'
}

f=lambda x:occ_mapping.get(x,x)
fec.contbr_occupation=fec.contbr_occupation.map(f)

fec.contbr_occupation.value_counts()[:10]

emp_mapping = {
   'INFORMATION REQUESTED PER BEST EFFORTS' : 'NOT PROVIDED',
   'INFORMATION REQUESTED' : 'NOT PROVIDED',
   'SELF' : 'SELF-EMPLOYED',
   'SELF EMPLOYED' : 'SELF-EMPLOYED',
}

f=lambda x:emp_mapping.get(x,x)
fec.contbr_employer=fec.contbr_employer.map(f)


by_occupation=fec.pivot_table('contb_receipt_amt',index='contbr_occupation',
                              columns='party',aggfunc='sum')



over_2mm=by_occupation[by_occupation.sum(1)>2000000]
over_2mm
over_2mm.plot(kind='barh')
plt.show()

def get_top_amounts(group,key,n=5):
    totals=group.groupby(key)['contb_receipt_amt'].sum()
    return totals.nlargest(n)

grouped=fec_mrbo.groupby('cand_nm')
grouped.apply(get_top_amounts,'contbr_occupation',n=7)

grouped.apply(get_top_amounts,'contbr_employer',n=10)

bins=np.array([0,1,10,100,1000,10000,100000,1000000,10000000])
labels=pd.cut(fec_mrbo.contb_receipt_amt,bins)
labels
grouped=fec_mrbo.groupby(['cand_nm',labels])
grouped.size().unstack(0)

bucket_sums=grouped.contb_receipt_amt.sum().unstack(0)
normed_sums=bucket_sums.div(bucket_sums.sum(axis=1),axis=0)
normed_sums

normed_sums[:-2].plot(kind='barh')
plt.show()


grouped=fec_mrbo.groupby(['cand_nm','contbr_st'])
totals=grouped.contb_receipt_amt.sum().unstack(0).fillna(0)
totals=totals[totals.sum(1)>100000]
totals[:10]

percent=totals.div(totals.sum(1),axis=0)
percent[:10]












        
