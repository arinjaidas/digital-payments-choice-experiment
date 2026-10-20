"""Runnable synthetic demonstration of the payment-choice analysis workflow.

Archived original scripts and aggregate results are separate from these
synthetic responses. No original participant-level data are included.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLUMNS=['UPIChoice','Default','PriorUPI','SwitchingEffort','PlatformReliance']


def synthetic_data(seed=20261019,n=800):
    rng=np.random.default_rng(seed)
    default=rng.binomial(1,.5,n);prior=rng.binomial(1,.65,n)
    effort=rng.normal(size=n);trust=rng.normal(size=n)
    score=-.65+.7*default+.85*prior+.35*effort+.2*trust
    choice=rng.binomial(1,1/(1+np.exp(-score)))
    return pd.DataFrame(dict(UPIChoice=choice,Default=default,PriorUPI=prior,
                             SwitchingEffort=effort,PlatformReliance=trust))


def validate(frame):
    if not set(COLUMNS).issubset(frame.columns):raise ValueError('Required columns: '+', '.join(COLUMNS))
    clean=frame[COLUMNS].apply(pd.to_numeric,errors='raise').replace([np.inf,-np.inf],np.nan).dropna()
    for col in ['UPIChoice','Default','PriorUPI']:
        if not set(clean[col].unique()).issubset({0,1}):raise ValueError(col+' must be binary.')
    if len(clean)<30 or clean.Default.nunique()!=2 or clean.UPIChoice.nunique()!=2:
        raise ValueError('Need at least 30 complete observations with both groups and outcomes.')
    design=sm.add_constant(clean.drop(columns='UPIChoice'),has_constant='add')
    if np.linalg.matrix_rank(design)<design.shape[1]:raise ValueError('Design matrix is rank deficient.')
    return clean


def analyse(frame):
    data=validate(frame)
    X=sm.add_constant(data.drop(columns='UPIChoice'),has_constant='add')
    lpm=sm.OLS(data.UPIChoice,X).fit(cov_type='HC3')
    logit=sm.Logit(data.UPIChoice,X).fit(disp=False,cov_type='HC3')
    d1=X.copy();d0=X.copy();d1['Default']=1;d0['Default']=0
    group=data.groupby('Default').UPIChoice.agg(['count','mean'])
    return data,group,lpm,logit,float((logit.predict(d1)-logit.predict(d0)).mean())


def run(frame,output,synthetic=False):
    output=Path(output);output.mkdir(exist_ok=True,parents=True)
    data,group,lpm,logit,ame=analyse(frame)
    group.to_csv(output/'group_summary.csv')
    rows=[]
    for name,fit in [('LPM_HC3',lpm),('Logit_HC3',logit)]:
        ci=fit.conf_int()
        for variable in fit.params.index:
            rows.append({'model':name,'variable':variable,'coefficient':float(fit.params[variable]),
                'hc3_se':float(fit.bse[variable]),'p_value':float(fit.pvalues[variable]),
                'ci_low':float(ci.loc[variable,0]),'ci_high':float(ci.loc[variable,1])})
    pd.DataFrame(rows).to_csv(output/'model_coefficients.csv',index=False)
    (output/'model_summaries.txt').write_text(lpm.summary().as_text()+'\n\n'+logit.summary().as_text())
    result={'synthetic':synthetic,'input_rows':len(frame),'complete_case_rows':len(data),
        'unadjusted_difference':float(group.loc[1,'mean']-group.loc[0,'mean']),
        'lpm_default_coefficient':float(lpm.params['Default']),
        'logit_average_discrete_change':ame,
        'interpretation':'Simulated workflow demonstration, not an estimate from the original study.' if synthetic else 'Conditional associations; causal interpretation depends on design and allocation.',
        'archived_results':'Original study results are excluded from this public edition.'}
    (output/'run_metadata.json').write_text(json.dumps(result,indent=2))
    if synthetic:data.to_csv(output/'synthetic_anonymous_responses.csv',index=False)
    fig,ax=plt.subplots(figsize=(6.5,4.5))
    ax.bar(['Active-choice group','UPI-default group'],group['mean'],color=['#355c7d','#28958b'])
    ax.set(ylim=(0,1),ylabel='UPI choice share',title=('SYNTHETIC DEMONSTRATION: ' if synthetic else '')+'Payment choice')
    fig.tight_layout();fig.savefig(output/'choice.svg');plt.close(fig)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--demo',action='store_true');g.add_argument('--data',type=Path)
    p.add_argument('--output',default='demo_output');a=p.parse_args()
    frame=synthetic_data() if a.demo else pd.read_csv(a.data)
    print(json.dumps(run(frame,a.output,a.demo),indent=2))
