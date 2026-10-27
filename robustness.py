from pathlib import Path
import json
import numpy as np,pandas as pd
from scipy.stats import chi2_contingency
D=pd.read_csv('project/data/raw/dataset.csv');M=json.load(open('project/data/raw/mapping.json'));O=Path('full_results')
cols=[x for x in D if x!='Unnamed: 0'];ded=D.drop_duplicates(cols);G=ded.groupby('timestamp').agg(y=('class','max'),dominant=('inspection_type',lambda s:s.mode().min()));tab=pd.crosstab(G.dominant,G.y);chi,p,df,exp=chi2_contingency(tab);tab.to_csv(O/'dedup_type_counts.csv');json.dump({'chi2':chi,'p':p,'V':float(np.sqrt(chi/len(G))),'min_expected':float(exp.min())},open(O/'dedup_type_test.json','w'),indent=2)
rows=[]
for typ,fs in M.items():
 for f in fs:
  v=D.loc[D.inspection_type==int(typ),f];q=v.quantile([.25,.5,.75]);iq=q.loc[.75]-q.loc[.25];rows.append([typ,f,len(v),v.nunique(),v.min(),q.loc[.25],q.loc[.5],q.loc[.75],v.max(),((v<q.loc[.25]-1.5*iq)|(v>q.loc[.75]+1.5*iq)).sum()])
pd.DataFrame(rows,columns=['type','feature','n','unique','min','q1','median','q3','max','iqr_flags']).to_csv(O/'measurement_by_type.csv',index=False)
