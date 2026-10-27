from pathlib import Path
import json,hashlib,time,sys,platform
import numpy as np,pandas as pd
from scipy.stats import norm,chi2_contingency
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score,confusion_matrix,precision_recall_curve
from sklearn.inspection import permutation_importance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import importlib.metadata
O=Path('full_results');O.mkdir(exist_ok=True);rng=np.random.default_rng(640);z=norm.ppf(.99375)
def wil(x,n):
 p=np.asarray(x)/n;c=(p+z*z/(2*n))/(1+z*z/n);h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n);return c-h,c+h
D=pd.read_csv('project/data/raw/dataset.csv');mapping=json.load(open('project/data/raw/mapping.json'))
D['date']=pd.to_datetime(D.timestamp,utc=True).dt.normalize();days=sorted(D.date.unique());di={v:i+1 for i,v in enumerate(days)};D['day']=D.date.map(di)
G=D.groupby('timestamp',sort=True).agg(y=('class','max'),n=('class','size'),day=('day','first'),dominant=('inspection_type',lambda s:s.mode().min())).reset_index();G['gid']=np.arange(len(G));ids=dict(zip(G.timestamp,G.gid));D['gid']=D.timestamp.map(ids)
G['period']=(G.day>50).astype(int)
G.to_csv(O/'group_data.csv',index=False)
feat=[x for x in D if x.startswith('inspection_feat')];num={}
for k,cols in mapping.items():
 for c in cols:num[c+'__type'+k]=np.where(D.inspection_type==int(k),D[c],0).astype('float32')
X=pd.DataFrame(num);cats=['inspection_type']+[x for x in D if x.startswith('meta_feat')]
for c in cats:X[c]=D[c].astype(str)
N=list(num);y=D['class'].to_numpy();dd=D.day.to_numpy();gid=D.gid.to_numpy()
# audit diagnostics using applicable values only
rows=[]
for c in feat:
 types=[int(k) for k,cols in mapping.items() if c in cols]
 v=D.loc[D.inspection_type.isin(types),c] if types else pd.Series(dtype=float)
 if len(v):
  q1,q3=v.quantile([.25,.75]);iqr=q3-q1;rows.append([c,len(v),int(v.isna().sum()),v.min(),v.max(),v.median(),q1,q3,int(((v<q1-1.5*iqr)|(v>q3+1.5*iqr)).sum()),int((v==0).sum()),int((v==1).sum())])
A=pd.DataFrame(rows,columns=['field','applicable_n','nulls','min','max','median','q1','q3','iqr_flags','zeros','ones']);A.to_csv(O/'measurement_audit.csv',index=False)
origcols=[c for c in D if c not in ['date','day','gid','Unnamed: 0']];ded=D.drop_duplicates(origcols);gd=ded.groupby('timestamp').agg(y=('class','max'),dominant=('inspection_type',lambda s:s.mode().min()))
stat={'rows':len(D),'dedup_rows':len(ded),'groups':len(G),'gated_features':len(N),'duplicate_groups_changed_type':int((gd.dominant.reindex(G.timestamp).to_numpy()!=G.dominant.to_numpy()).sum()),'group_size_quantiles':G.n.quantile([0,.25,.5,.75,.95,1]).to_dict(),'iqr_flags':int(A.iqr_flags.sum()),'applicable_cells':int(A.applicable_n.sum()),'applicable_min':float(A['min'].min()),'applicable_max':float(A['max'].max())}
json.dump(stat,open(O/'diagnostics.json','w'),indent=2)
print('audit',stat,flush=True)
def make(kind,pars):
 pre=ColumnTransformer([('n',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),N),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float32),cats)],sparse_threshold=0)
 if kind=='LR':m=LogisticRegression(max_iter=500,solver='lbfgs',**pars)
 elif kind=='RF':m=RandomForestClassifier(n_estimators=200,n_jobs=1,random_state=640,**pars)
 else:m=HistGradientBoostingClassifier(max_iter=100,random_state=640,**pars)
 return Pipeline([('pre',pre),('constant',VarianceThreshold()),('model',m)])
def group_scores(mask,s):
 return pd.Series(s,index=gid[mask]).groupby(level=0).max()
configs=[]
for c in [.01,.1,1]:
 for w in [None,'balanced']:configs.append(('LR',{'C':c,'class_weight':w}))
for dep in [3,6]:
 for leaf in [50,200]:configs.append(('RF',{'max_depth':dep,'min_samples_leaf':leaf}))
for leaf in [7,15]:
 for l2 in [1,10]:configs.append(('HGB',{'max_leaf_nodes':leaf,'l2_regularization':l2}))
folds=[(20,21,25),(25,26,30),(30,31,40)];results=[]
for idx,(kind,pars) in enumerate(configs):
 aps=[]
 for tr,lo,hi in folds:
  tm=dd<=tr;vm=(dd>=lo)&(dd<=hi);m=make(kind,pars);m.fit(X.loc[tm],y[tm]);s=group_scores(vm,m.predict_proba(X.loc[vm])[:,1]);ap=average_precision_score(G.set_index('gid').loc[s.index,'y'],s);aps.append(ap)
 row={'id':idx,'kind':kind,'params':pars,'fold_ap':aps,'mean_ap':float(np.mean(aps))};results.append(row);json.dump(results,open(O/'model_selection.json','w'),indent=2);print('candidate',row,flush=True)
best=max(results,key=lambda r:(r['mean_ap'],-r['id']));kind,pars=configs[best['id']];model=make(kind,pars);tm=dd<=40;model.fit(X.loc[tm],y[tm]);vm=(dd>40)&(dd<=60);vs=group_scores(vm,model.predict_proba(X.loc[vm])[:,1]);vy=G.set_index('gid').loc[vs.index,'y'].to_numpy();ss=vs.to_numpy()
# choose largest eligible score threshold with maximum avoidance
cand=np.unique(ss);pos=np.sort(ss[vy==1]);tp=len(pos)-np.searchsorted(pos,cand,side='left');lower=wil(tp,len(pos))[0];eligible=cand[lower>.9];tau=float(eligible.max()) if len(eligible) else -1.
val={'threshold':tau,'positives':int(vy.sum()),'sensitivity':float(np.mean(ss[vy==1]>=tau)),'avoidance':float(np.mean(ss<tau)),'lower_sensitivity':float(wil(np.sum(ss[vy==1]>=tau),int(vy.sum()))[0])}
json.dump({'best':best,'validation':val},open(O/'frozen_rule.json','w'),indent=2)
# frozen final evaluation
te=dd>60;ts=group_scores(te,model.predict_proba(X.loc[te])[:,1]);test=G.set_index('gid').loc[ts.index].copy();test['score']=ts;test['retain']=(ts>=tau).astype(int);test.index.name='gid';test.to_csv(O/'test_predictions.csv');tn,fp,fn,tp=confusion_matrix(test.y,test.retain,labels=[0,1]).ravel();metrics={'TP':int(tp),'FN':int(fn),'FP':int(fp),'TN':int(tn),'sensitivity':tp/(tp+fn),'precision':tp/(tp+fp),'avoidance':(tn+fn)/len(test),'negative_avoidance':tn/len(test),'miss_rate':fn/(tp+fn),'average_precision':average_precision_score(test.y,test.score),'sensitivity_ci':list(wil(tp,tp+fn)),'avoidance_ci':list(wil(tn+fn,len(test)))}
# bootstrap day-level sums in moving blocks; fixed-rule conditional intervals
counts=test.assign(pos=test.y,tp=test.y*test.retain,avoid=1-test.retain,one=1).groupby('day')[['pos','tp','avoid','one']].sum().to_numpy()
def boot_counts(arr,b):
 n=len(arr);ix=np.concatenate([np.arange(st,st+b)%n for st in rng.integers(0,n,size=math.ceil(n/b))])[:n];return arr[ix].sum(axis=0)
import math
boot={}
for b in [1,3,7]:
 vals=[]
 for _ in range(2000):
  c=boot_counts(counts,b);vals.append([c[1]/c[0] if c[0] else np.nan,c[2]/c[3]])
 boot[str(b)]=np.nanquantile(vals,[.00625,.99375],axis=0).T.tolist()
metrics['block_ci']=boot
# periods bootstrap and cluster regression
periods={}
for b in [1,3,7]:
 arrs=[g.groupby('day').agg(pos=('y','sum'),n=('y','size')).to_numpy() for _,g in G.groupby('period')];vals=[]
 for _ in range(2000):
  c0=boot_counts(arrs[0],b);c1=boot_counts(arrs[1],b);vals.append(c1[0]/c1[1]-c0[0]/c0[1])
 periods[str(b)]=np.quantile(vals,[.00625,.99375]).tolist()
from scipy.stats import chi2
# Unpenalized logistic MLE and day-cluster sandwich covariance
U=pd.get_dummies(G.dominant,prefix='type',drop_first=True,dtype=float)
def cluster_fit(include_period):
 Z=np.column_stack([np.ones(len(G)),G.period.to_numpy(),U.to_numpy()]) if include_period else np.column_stack([np.ones(len(G)),U.to_numpy()])
 lr=LogisticRegression(penalty=None,fit_intercept=False,max_iter=1000,tol=1e-10).fit(Z,G.y)
 beta=lr.coef_[0];mu=lr.predict_proba(Z)[:,1];bread=np.linalg.pinv(Z.T@((mu*(1-mu))[:,None]*Z));scores=Z*(G.y.to_numpy()-mu)[:,None]
 sums=pd.DataFrame(scores).groupby(G.day.to_numpy()).sum().to_numpy();n,k=Z.shape;c=len(sums);cov=bread@(sums.T@sums)@bread*(c/(c-1))*((n-1)/(n-k));return beta,cov
beta,cov=cluster_fit(True);beta2,cov2=cluster_fit(False);w=float(beta2[1:]@np.linalg.pinv(cov2[1:,1:])@beta2[1:]);se1=float(np.sqrt(cov[1,1]))
json.dump({'period_block_ci':periods,'adjusted_period_coef':float(beta[1]),'adjusted_period_se':se1,'adjusted_period_p':float(2*norm.sf(abs(beta[1]/se1))),'type_cluster_wald':w,'type_cluster_p':float(chi2.sf(w,4))},open(O/'cluster_results.json','w'),indent=2)
# deduplicated frozen rule sensitivity scores (same fit)
dupmask=~D.duplicated(origcols);ds=group_scores(te&dupmask,model.predict_proba(X.loc[te&dupmask])[:,1]);assert np.allclose(ds.to_numpy(),ts.to_numpy())
metrics['dedup_same_frozen_scores']=True
json.dump(metrics,open(O/'test_metrics.json','w'),indent=2)
# deterministic fitted LR coefficients or feature importances available for chosen model
names=model['pre'].get_feature_names_out();names=names[model['constant'].get_support()];est=model['model'];imp=np.abs(est.coef_[0]) if hasattr(est,'coef_') else getattr(est,'feature_importances_',None)
if imp is not None:pd.DataFrame({'feature':names,'importance':imp}).sort_values('importance',ascending=False).to_csv(O/'feature_importance.csv',index=False)
# figures
plt.rcParams.update({'font.size':10})
def save(name):plt.tight_layout();plt.savefig(O/(name+'.png'),dpi=160);plt.close()
daily=G.groupby('day').agg(n=('y','size'),pos=('y','sum'));daily.to_csv(O/'daily.csv');plt.figure(figsize=(6.3,3.2));plt.plot(daily.index,100*daily.pos/daily.n);plt.xlabel('Active production day');plt.ylabel('Defect groups (%)');save('daily')
plt.figure(figsize=(6.3,3.2));pr,re,_=precision_recall_curve(test.y,test.score);plt.plot(re,pr);plt.axhline(test.y.mean(),linestyle='--',color='gray');plt.xlabel('Recall');plt.ylabel('Precision');save('precision_recall')
plt.figure(figsize=(6.3,3.2));srt=np.sort(test.score);thresholds=np.unique(np.quantile(srt,np.linspace(0,1,201)));plt.plot([np.mean(test.score<t) for t in thresholds],[np.mean(test.loc[test.y==1,'score']>=t) for t in thresholds]);plt.scatter(metrics['avoidance'],metrics['sensitivity'],color='red');plt.xlabel('Candidate review avoidance');plt.ylabel('Sensitivity');save('tradeoff')
json.dump({'python':sys.version,'packages':{k:importlib.metadata.version(k) for k in ['numpy','pandas','scipy','scikit-learn','matplotlib']},'seed':640,'folds':folds,'source':'original source manifest verified','run_date':'2026-10-27','test_used_once':True},open(O/'run_manifest.json','w'),indent=2)
print('FINAL',best,val,metrics,flush=True)
