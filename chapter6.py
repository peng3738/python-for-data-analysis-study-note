from IPython import get_ipython
!type E:/python_study/python for data analysis/examples/ex1.csv
'''
a,b,c,d,message
1,2,3,4,hello
5,6,7,8,world
9,10,11,12,foo
'''

path='E:/python_study/python for data analysis'
path='E:/python_study/python for data analysis/examples'

filename='/ex1.csv'

import pandas as pd
df=pd.read_csv('E:/python_study/python for data analysis/examples/ex1.csv')
df
df1=pd.read_csv(path+filename)

pd.read_table(path+filename,sep=',')

filename2='/ex2.csv'
pd.read_csv(path+filename2,header=None)

pd.read_csv(path+filename2,names=['a','b','c','d','message'])

names=['a','b','c','d','message']
pd.read_csv(path+filename2,names=names,index_col='message')

filename3='/csv_mindex.csv'
parsed=pd.read_csv(path+filename3,index_col=['key1','key2'])
parsed

filename4='/ex3.txt'
list(open(path+filename4))

result=pd.read_table(path+filename4,sep='\s+')
result

pd.read_csv(path+'/ex4.csv',skiprows=[0,2,3])
result=pd.read_csv(path+'/ex5.csv')

pd.isnull(result)

result=pd.read_csv(path+'/ex5.csv',na_values=['Null'])
result

sentinels={'message':['foo','NA'],'something':['two']}
pd.read_csv(path+'/ex5.csv',na_values=sentinels)


pd.options.display.max_rows=10

result=pd.read_csv(path+'/ex6.csv')
result

pd.read_csv(path+'/ex6.csv',nrows=5)

chunker=pd.read_csv(path+'/ex6.csv',chunksize=1000)
chunker

tot=pd.Series([])
for piece in chunker:
    tot=tot.add(piece['key'].value_counts(),fill_value=0)
tot=tot.sort_values(ascending=False)
tot[:10]

data=pd.read_csv(path+'/ex5.csv')
data

data.to_csv(path+'/out.csv')

#!type path+'/out.csv'
import sys
data.to_csv(sys.stdout,sep='|')

data.to_csv(sys.stdout,na_rep='NULL')

data.to_csv(sys.stdout,index=False,header=False)

dates=pd.date_range('1/1/2000',periods=7)
ts=pd.Series(np.arange(7),index=dates)
ts.to_csv(path+'/tseries.csv')

import csv
f=open(path+'/ex7.csv')
reader=csv.reader(f)

for line in reader:
    print(line)

with open(path+'/ex7.csv') as f:
    lines=list(csv.reader(f))


header,values=lines[0],lines[1:]

data_dict={h:v for h,v in zip(header,zip(*values))}
data_dict

class my_dialect(csv.Dialect):
    lineterminator='\n'
    delimiter=';'
    quotechar='"'
    quoting=csv.QUOTE_MINIMAL

#reader=csv.reader(f,dialect=my_dialect)
obj="""
{"name":"wes",
"places_lived":["United States","Spain","Germany"],
"pet":null,
"siblings":[{"name":"Scott","age":30,"pets":["Zeus","Zuko"]},
{"name":"Katie","age":38,"pets":["Sixes","Stache","Cisco"]}]
}
"""

import json
result=json.loads(obj)
result

asjson=json.dumps(result)

siblings=pd.DataFrame(result['siblings'],columns=['name','age'])
siblings

data=pd.read_json(path+'/example.json')
data

print(data.to_json())
print(data.to_json(orient='records'))


path='E:/python_study/python for data analysis/examples'
tables=pd.read_html(path+'/fdic_failed_bank_list.html')
tables
len(tables)
failures=tables[0]
failures.head()

close_timestamps=pd.to_datetime(failures['Closing Date'])
close_timestamps.dt.year.value_counts()

from lxml import objectify
path1='E:/python_study/python for data analysis/pydata-book-2nd-edition/datasets/mta_perf/Performance_MNR.xml'
parsed=objectify.parse(open(path1))
root=parsed.getroot()

data=[]
skip_fields=['OARENT_SEQ','INDICATOR_SEQ',
             'DESIRED_CHANGE','DECIMAL+PLACES']
for elt in root.INDICATOR:
    el_data={}
    for child in elt.getchildren():
        if child.tag in skip_fields:
            continue
        el_data[child.tag]=child.pyval
    data.append(el_data)

perf=pd.DataFrame(data)
perf.head()


from io import StringIO
tag='<a href="http://google.com">Google</a>'
root=objectify.parse(StringIO(tag)).getroot()

root
root.get('href')
root.text

frame=pd.read_csv(path+'/ex1.csv')
frame
frame.to_pickle(path+'/frame_pickle')

pd.read_pickle(path+'/frame_pickle')


import numpy as np
import pandas as pd
from pandas import HDFStore
path='E:/python_study/python for data analysis'
frame=pd.DataFrame({'a':np.random.randn(100)})
store=pd.HDFStore(path+'/examples/mydata.h5')
store['obj1']=frame
store['obj1_col']=frame['a']
store

path='E:/python_study/python for data analysis/examples'
xlsx=pd.ExcelFile(path+'/ex1.xlsx')

pd.read_excel(xlsx,'Sheet1')

frame=pd.read_excel(path+'/ex1.xlsx','Sheet1')
frame
writer=pd.ExcelWriter(path+'/ex2.xlsx')
frame.to_excel(writer,'Sheet1')

frame.to_excel(path+'/ex2.xlsx')

import requests
url='https://api.github.com/repos/pandas-dev/pandas/issues'
resp=requests.get(url)
resp

data=resp.json()
data[0]['title']
issues=pd.DataFrame(data,columns=['number','title','labels','state'])
issues


import sqlite3

query="""CREATE TABLE TEST (a VARCHAR(20),b VARCHAR(20)),c REAL, d INTEGER);"""

con=sqlite3.connect('mydata.sqlite')
con.execute(query)
con.commit()

data=[('Atlanta','Georgia',1.25,6),('Tallahassee','Florida',2.6,3),
      ('Sacramento','California',1.7,5)]

stmt="INSERT INTO test VALUES(?,?,?,?)"
con.executemany(stmt,data)
con.commit()

cursor=con.execute('select * from test')
rows=cursor.fetchall()

rows

cursor.description

pd.DataFrame(rows,columns=[x[0] for x in cursor.description])


import sqlalchemy as sqla
db=sqla.create_engine('sqlite:///mydata.sqlite')
pd.read_sql('select * from test',db)
























