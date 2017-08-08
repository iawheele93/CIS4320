"""

    @Author:    Ian Wheeler; 2/14/17; <wheeler.i5971@gmail.com>
    ------------------------------------------------------------------------
    IDE:        PyCharm CE
    ------------------------------------------------------------------------
    Title:      Assignment #2
    ------------------------------------------------------------------------
    Dataset:    res.csv
    ------------------------------------------------------------------------
    Source:     Xuqing Wu; University of Houston

"""
import pandas as pd
import json
import numpy as np
import collections

resdf = pd.read_csv('res.csv', encoding='utf8')
resdf.apply(lambda x: pd.lib.infer_dtype(x.values))

zipCode = resdf['zipCode'].values.tolist()
neighborhoodraw = resdf['neighborhood'].values
councilDistrict = resdf['councilDistrict'].values
policeDistrictraw = resdf['policeDistrict'].values
location1raw = resdf['Location 1'].values
resNameraw = resdf['name'].values

neighborhood=[elm.strip().upper() for elm in neighborhoodraw]
policeDistrict=[elm.strip() for elm in policeDistrictraw]
location1=[elm.strip() for elm in location1raw]
resName=[elm.strip() for elm in resNameraw]

"""
question-1: Find how many different zipCode in the table
"""
q1 = len(set(zipCode))

"""
question-2: Find how many different councilDistrict in the table
"""
q2 = len(set(councilDistrict))

"""
question-3: Find how many different zipCode in the table
"""
q3 = len(set(zipCode))

"""
question-4: Find how many different policeDistrict in the table
"""
q4 = len(set(policeDistrict))

"""
question-5: Find out which policeDistrict has the largest number of
restaurants. If you got more than one policeDistricts, put them in a list,
e.g. ['SOUTHERN','NORTHERN']
"""
resInPoliceD = [(dname, rcnt) for dname, rcnt in
              collections.Counter(policeDistrict).items()]
dname, rcnt = zip(*resInPoliceD)
maxcnt = max(rcnt)
idxs = [idx for idx, val in enumerate(rcnt) if val == maxcnt]
q5 = [dname[idx] for idx in idxs]

print(rcnt)
print(enumerate(rcnt))
print(collections.Counter(policeDistrict).items())

"""
question-6: Find out which policeDistrict has the largest number of
restaurants. If you got more than one policeDistricts, put them in a list,
e.g. ['SOUTHERN','NORTHERN']
"""
resInPoliceD = [(dname, rcnt) for dname, rcnt in
              collections.Counter(policeDistrict).items()]
dname, rcnt = zip(*resInPoliceD)
maxcnt = max(rcnt)
idxs = [idx for idx, val in enumerate(rcnt) if val == maxcnt]
q6 = [dname[idx] for idx in idxs]

print(list(collections.Counter(policeDistrict).items()))

"""
question-7: Find out which zipCode has the largest number of
restaurants. If you got more than one zipCode, put them in a list,
e.g. [21215,21217]
"""
resInzipCode = [(zcode, rcnt) for zcode, rcnt in
              collections.Counter(zipCode).items()]
zcode, rcnt = zip(*resInzipCode)
maxcnt = max(rcnt)
idxs = [idx for idx, val in enumerate(rcnt) if val == maxcnt]
q7 = [zcode[idx] for idx in idxs]

"""
question-8: List all different neighborhood in the SOUTHERN policeDistrict.
Put your answer in a list, e.g. ['Cherry Hill', 'Curtis Bay', 'Federal Hill']
"""
nbs_pd = zip(neighborhood, policeDistrict)
allnbsraw = [nb for nb, pd in nbs_pd if pd == 'SOUTHERN']
allnbs = set(allnbsraw)
q8 = list(allnbs)

"""
question-9: After finding out all BURGER KING restaurants. Find out in which
policeDistrict, it has more than one BURGER KING restaurants. If you got
more than one policeDistricts, put them in a list,
e.g. ['SOUTHERN','NORTHERN'].

NOTE: the name like BURGER KING # 10293 is also a BURGER KING restaurant,
"""
idxs = [i for i, v in enumerate(resName) if 'BURGER KING' in v]
pdraw = [policeDistrict[i] for i in idxs]
q9 = [pd for pd, cnt in collections.Counter(pdraw).items() if cnt > 1]

"""
question-10: Are there any relplicated names in the location 1 column? If
the answer is yes, put them in a list,
e.g. ['Hopkins Pl Baltimore, MD','Hayward Ave Baltimore, MD']. If not, assign
empty list to it.
"""
q10 = [aloc for aloc, cnt in collections.Counter(location1).items() if cnt > 1]

"""
question-11: How many different zipCodes are used in the CENTRAL
policeDistrict?
"""
zip_pd=zip(zipCode,policeDistrict)
allzipsraw=[azip for azip,pd in zip_pd if pd=='CENTRAL']
q11=len(set(allzipsraw))

"""
Fill in you answer into the follow structure. e.g.
suppose a=[3 4 5] and b=39
answer={'question-1':a,'question-2':b}
"""

answer = {'question-1': q1, 'question-2': q2,
          'question-3': q3, 'question-4': q4,
          'question-5': q5, 'question-6': q6,
          'question-7': q7, 'question-8': q8,
          'question-9': q9, 'question-10': q10,
          'question-11': q11}

print(answer)

""" 
DON'T CHANGE THE FOLLOWING CODE
"""
#with open('myoutput.txt','w') as outfile:
    #json.dump(answer,outfile)
    
#end