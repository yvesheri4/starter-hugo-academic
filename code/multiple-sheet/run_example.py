"""Reproduce a compact drift-space example using the archived Multiple-sheet core."""
import argparse, csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiple_sheet import Simulation, pulse_profile

def verify():
    # Zero space charge must give constant velocity and ballistic motion.
    n=8; steps=101; time=np.linspace(0,.4,steps)
    p,v,e=Simulation(.4,[.01,np.ones(n),np.zeros(n),1000,1],[np.zeros(n),np.ones(n)],steps)
    np.testing.assert_allclose(p[:,0],1.5*time,atol=1e-12)
    np.testing.assert_allclose(v[:,0],1,atol=1e-12)
    np.testing.assert_allclose(e,0,atol=1e-12)
    for kind in ['square','trapezoidal','gauss']:
        q=pulse_profile(.1,80,Type=kind)
        assert np.all(q>=0)
        np.testing.assert_allclose(q.sum(),1,atol=1e-12)
    print('Ballistic limit and charge-normalized profile checks passed.')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=Path('results'))
    ap.add_argument('--sheets',type=int,default=80)
    ap.add_argument('--steps',type=int,default=1200)
    ap.add_argument('--check',action='store_true')
    a=ap.parse_args();verify()
    if a.check:return
    if a.sheets<8 or a.steps<100:ap.error('Use at least 8 sheets and 100 time samples.')
    a.output.mkdir(parents=True,exist_ok=True)
    times=np.linspace(0,4,a.steps)
    fig,axes=plt.subplots(1,3,figsize=(10.5,3.8),sharex=True,sharey=True,layout='constrained')
    names=['Square-top','Trapezoidal','Gaussian']; kinds=['square','trapezoidal','gauss']
    colors=['#237665','#357ca0','#a2468d']; factors=[1,1.15,2.1276595]; critical=[.825,.825,.885]
    rows=[];arrivals=[]; summaries=[]
    for name,kind,factor,limit,color,ax in zip(names,kinds,factors,critical,colors,axes):
        weights=pulse_profile(.1,a.sheets,Type=kind);charge=.5*limit*weights
        p,v,e=Simulation(4,[factor*.1/a.sheets,charge,charge,1000,1],[np.zeros(a.sheets),np.ones(a.sheets)],a.steps)
        assert np.isfinite(p).all() and np.isfinite(v).all()
        injection=np.arange(a.sheets)*factor*.1/a.sheets
        for i,ti in enumerate(times):
            inside=(ti>=injection)&(p[i]>=0)&(p[i]<1)&(charge>0)
            ordered=p[i,inside]
            if len(ordered)>1 and np.any(np.diff(ordered)>1e-12):
                raise ValueError('Sheet crossing detected: outside the archived solver regime')
        for j in range(a.sheets):
            valid=(times>=injection[j])&(p[:,j]>=0)&(p[:,j]<=1)
            ax.plot(times,np.where(valid,p[:,j],np.nan),color=color,alpha=.6,lw=.65)
            idx=np.flatnonzero(p[:,j]>=1)
            arrivals.append([name,j,float(charge[j]),float(times[idx[0]]) if len(idx) else '',float(v[idx[0],j]) if len(idx) else ''])
        for i in np.linspace(0,a.steps-1,61,dtype=int):
            for j in np.unique(np.linspace(0,a.sheets-1,8,dtype=int)):
                if times[i]>=injection[j] and 0<=p[i,j]<=1:rows.append([name,int(j),float(times[i]),float(p[i,j]),float(v[i,j])])
        ax.set(title=name,xlabel='Normalized time',xlim=(0,3.5),ylim=(0,1));ax.grid(alpha=.18)
        summaries.append({'profile':name,'total_normalized_sheet_charge':float(charge.sum()),'minimum_saved_velocity':float(v.min()),'sheets_reaching_anode':sum(1 for row in arrivals if row[0]==name and row[3]!='')})
    last_arrival=max(row[3] for row in arrivals if row[3]!='')
    for ax in axes: ax.set_xlim(0,min(4,last_arrival*1.05))
    axes[0].set_ylabel('Normalized position')
    fig.suptitle(f'Multiple-sheet drift-space example | {a.sheets} sheets per profile',fontsize=13)
    fig.savefig(a.output/'trajectories.png',dpi=130);plt.close(fig)
    for filename,header,data in [('trajectory_samples.csv',['profile','sheet','time_normalized','position_normalized','velocity_normalized'],rows),('arrivals.csv',['profile','sheet','normalized_sheet_charge','arrival_time_normalized','arrival_velocity_normalized'],arrivals)]:
        with (a.output/filename).open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(header);w.writerows(data)
    result={'sheets':a.sheets,'time_samples':a.steps,'end_time':4,'pulse_duration_parameter':.1,'charge_fraction_of_archived_limit':.5,'profiles':summaries,'note':'Illustrative rerun of the archived numerical core; not a convergence study or a claim to regenerate the published figures exactly.'}
    (a.output/'run.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    table=np.genfromtxt(Path(__file__).parent/'data/current_limits.csv',delimiter=',',names=True)
    fig,ax=plt.subplots(figsize=(7,4.5),layout='constrained')
    for name,key,color in zip(names,['square','trapezoidal','gaussian'],colors):
        ax.loglog(table['k'],table[key+'_sheet'],'o-',color=color,label=name+' · sheets')
        ax.errorbar(table['k'],table[key+'_pic'],yerr=table[key+'_pic_error'],fmt='x',color=color,capsize=3,label=name+' · PIC')
    ax.set(xlabel='Normalized pulse duration k',ylabel='Normalized limiting current density',title='Archived current-limit comparison')
    ax.grid(which='both',alpha=.2);ax.legend(fontsize=8,ncol=2)
    fig.savefig(a.output/'current-limits.png',dpi=150);plt.close(fig)
    print(json.dumps(result))

if __name__=='__main__':main()
