"""Reproduce four prospective planning calculations; not observed model results."""
from pathlib import Path
import json,math
from scipy.stats import norm,chi2,ncx2
from scipy.optimize import brentq
ROOT=Path(__file__).resolve().parents[1]
a=json.loads((ROOT/'results/audit.json').read_text())
alpha=.05/4;z=norm.ppf(1-alpha/2);zb=norm.ppf(.8)
c=chi2.ppf(1-alpha,4)
n1=math.ceil(brentq(lambda n:ncx2.sf(c,4,n*.1**2)-.8,1,100000))
m2=math.ceil(z*z*.9*.1/.05**2);pi=a['positive_groups']/a['groups'];n2=math.ceil(m2/pi)
n3=math.ceil(z*z*.5*.5/.02**2)
p1=.02;p2=.03;pb=(p1+p2)/2
n4each=math.ceil((z*(2*pb*(1-pb))**.5+zb*(p1*(1-p1)+p2*(1-p2))**.5)**2/(p2-p1)**2)
values={'alpha_per_RQ':alpha,'confidence':1-alpha,'power':.8,'z_two_sided':z,'z_power':zb,'chi_critical':c,
    'N1':n1,'RQ2_positive_events':m2,'group_prevalence_for_projection':pi,'N2_cohort_equivalent':n2,
    'N3_test_groups':n3,'RQ4_per_period':n4each,'N4':2*n4each,'Nminimum':max(n1,n2,n3,2*n4each),'Nstudy':a['groups'],
    'event_and_allocation_checks':{'RQ2':a['test']['positives']>=m2,'RQ3':a['test']['groups']>=n3,
    'RQ4_early':a['early']['groups']>=n4each,'RQ4_late':a['late']['groups']>=n4each},
    'design_effect_two_sensitivity':{'RQ2_events':2*m2,'RQ2_available':a['test']['positives'],'RQ3_groups':2*n3,
    'RQ4_each':2*n4each,'interpretation':'Sensitivity assumption, not an estimated design effect.'}}
assert [n1,n2,n3,2*n4each]==[1611,10220,3900,10870]
(ROOT/'results/sample_sizes.json').write_text(json.dumps(values,indent=2)+'\n');print(json.dumps(values,indent=2))
