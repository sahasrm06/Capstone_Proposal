# Refit the frozen selected specification deterministically; verify it reproduces saved test scores.
from pathlib import Path
source=Path('analyze_full.py').read_text();exec(source[:source.index('configs=[]')])
from sklearn.metrics import average_precision_score
from scipy.stats import chi2_contingency
import joblib
cfg=json.load(open(O/'frozen_rule.json'));best=cfg['best'];tm=dd<=40;model=make(best['kind'],best['params']);model.fit(X.loc[tm],y[tm]);te=dd>60;new=group_scores(te,model.predict_proba(X.loc[te])[:,1]);old=pd.read_csv(O/'test_predictions.csv').set_index('gid');assert np.allclose(new.loc[old.index],old.score,rtol=1e-7,atol=1e-10)
joblib.dump(model,O/'frozen_model.joblib')
names=model['pre'].get_feature_names_out();names=names[model['constant'].get_support()];est=model['model'];kind=best['kind']
if hasattr(est,'coef_'):imp=np.abs(est.coef_[0]);method='absolute standardized logistic coefficient'
elif hasattr(est,'feature_importances_'):imp=est.feature_importances_;method='tree impurity importance'
else:
 imp=np.zeros(len(names))
 for predictors in est._predictors:
  for tree in predictors:
   for node in tree.nodes:
    if not node['is_leaf']:imp[int(node['feature_idx'])]+=max(float(node['gain']),0)
 method='summed positive training split gain from pinned histogram boosting implementation'
if imp.sum()>0:imp=imp/imp.sum()
pd.DataFrame({'feature':names,'relative_importance':imp}).sort_values('relative_importance',ascending=False).to_csv(O/'feature_importance.csv',index=False)
# Metadata-only development diagnostic; never used to select the primary rule.
meta=Pipeline([('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False)),('model',LogisticRegression(C=.1,max_iter=500))]);meta.fit(X.loc[tm,cats],y[tm]);vm=(dd>40)&(dd<=60);ms=group_scores(vm,meta.predict_proba(X.loc[vm,cats])[:,1]);ap=average_precision_score(G.set_index('gid').loc[ms.index,'y'],ms)
manifest=json.load(open(O/'run_manifest.json'));manifest['model_sha256']=hashlib.sha256((O/'frozen_model.joblib').read_bytes()).hexdigest();manifest['reproduced_test_scores']=True;manifest['importance_method']=method;manifest['metadata_only_validation_ap']=ap;manifest['joblib']=importlib.metadata.version('joblib');json.dump(manifest,open(O/'run_manifest.json','w'),indent=2)
print('Export verified; metadata-only validation AP',ap,flush=True)
