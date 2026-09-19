"""Build portfolio charts from frozen derived tables only (no model fitting)."""
from pathlib import Path
import hashlib, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'assets'
TEAL='#137C8B'; NAVY='#263D57'; ORANGE='#C47B35'; GRAY='#8794A3'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelcolor':NAVY,
 'text.color':NAVY,'axes.edgecolor':'#BAC5CF','axes.spines.top':False,'axes.spines.right':False,
 'axes.titleweight':'bold','savefig.facecolor':'white','figure.facecolor':'white',
 'axes.grid':False,'svg.fonttype':'none'})
inputs=[]; outputs=[]

def read(path):
    p=ROOT/path;inputs.append(p);return pd.read_csv(p)

def save(fig,name):
    OUT.mkdir(exist_ok=True)
    for ext in ['png','svg']:
        p=OUT/f'{name}.{ext}';fig.savefig(p,dpi=200,bbox_inches='tight');outputs.append(p)
    plt.close(fig)

def main():
    d=read('projects/microgrid-2026/data/F04_q1_dispatch.csv')
    x=np.arange(len(d))/6
    fig,axs=plt.subplots(2,1,figsize=(9,4.5),sharex=True,gridspec_kw={'height_ratios':[1.1,1]},layout='constrained')
    axs[0].plot(x,d.load_kw/1000,color=NAVY,label='Load',lw=1.8)
    axs[0].plot(x,d.pv_kw/1000,color=ORANGE,label='PV forecast',lw=1.8)
    axs[0].step(x,d.grid_kwh*6/1000,color=TEAL,label='Grid purchase',where='post',lw=1.5)
    axs[0].set(ylabel='Power (MW)',ylim=(0,10));axs[0].legend(ncol=3,frameon=False,loc='upper left',fontsize=9)
    axs[1].plot(np.arange(145)/6,np.r_[d.soc_start_kwh.iloc[0],d.soc_end_kwh]/1000,color=TEAL,lw=2)
    axs[1].axhspan(1.2,10.8,color=TEAL,alpha=.06);axs[1].set(ylabel='Stored energy (MWh)',xlabel='Hour',xlim=(0,24),ylim=(0,12),xticks=np.arange(0,25,4))
    for ax in axs:ax.grid(axis='y',alpha=.18)
    save(fig,'microgrid-dispatch')

    t=read('projects/microgrid-2026/data/strategy_summary.csv').set_index('strategy')
    labels=['D','S','F-fast','Q3'];sel=t.loc[['q2/D','q2/S','q2/F_fast','q3/Z2_111']]
    fig,ax=plt.subplots(figsize=(9,4.4),layout='constrained');bottom=np.zeros(4);x=np.arange(4)
    for key,label,color in [('plan_cost_yuan','Plan',NAVY),('adjustment_cost_yuan','Amendment',ORANGE),('emergency_cost_yuan','Emergency',TEAL)]:
        y=sel[key].to_numpy()/1e6;ax.bar(x,y,bottom=bottom,color=color,width=.52,label=label);bottom+=y
    for i,(_,r) in enumerate(sel.iterrows()):ax.text(i,r.total_cash_yuan/1e6+.22,f'{r.total_cash_yuan/1e6:.3f}',ha='center',fontweight='bold')
    ax.set(xticks=x,xticklabels=labels,ylabel='Cash cost (million CNY)',ylim=(0,18.2));ax.legend(ncol=3,frameon=False,loc='upper right',fontsize=9);ax.grid(axis='y',alpha=.18)
    ax.set_axisbelow(True);save(fig,'microgrid-costs')

    p=read('projects/agriculture-2024/data/q3_common_paths_profits.csv')
    fig,ax=plt.subplots(figsize=(9,4.3),layout='constrained')
    for field,label,color,style in [('q2_frozen_profit_yuan','Frozen Q2',TEAL,'-'),('q3_final_profit_yuan','Q3 candidate',ORANGE,'--')]:
        a=np.sort(p[field].to_numpy())/1e6;ax.plot(a,np.arange(1,len(a)+1)/len(a),label=label,color=color,ls=style,lw=2)
    ax.axhline(.1,color=GRAY,lw=1,ls=':');ax.text(65.5,.13,'10th percentile',color=GRAY,fontsize=9)
    ax.set(xlabel='Seven-year nominal profit (million CNY)',ylabel='Cumulative probability',ylim=(0,1.02))
    ax.legend(frameon=False,loc='lower right');ax.grid(alpha=.16);save(fig,'agriculture-distribution')

    s=read('projects/agriculture-2024/data/q3_sensitivity_summary.csv')
    s=s[(s.surplus_rule=='half_price_surplus')&(s.elasticity_scale=='baseline')]
    levels=['weak','baseline','strong']
    fig,axs=plt.subplots(1,2,figsize=(9,4.3),layout='constrained')
    for plan,label,color,marker in [('q2_frozen','Frozen Q2',TEAL,'o'),('q3_final','Q3 candidate',ORANGE,'s')]:
        a=s[s.plan==plan].set_index('dependence_strength').loc[levels]
        for ax,key in zip(axs,['mean_profit_yuan','cvar_90_yuan']):
            ax.plot(range(3),a[key]/1e6,color=color,marker=marker,lw=2,label=label)
    for ax,title in zip(axs,['Mean profit','Lower-tail CVaR (90%)']):
        ax.set(title=title,xticks=range(3),xticklabels=['Weak','Base','Strong'],xlabel='Assumed dependence',ylim=(66.9,70));ax.grid(axis='y',alpha=.18)
    axs[0].set_ylabel('Seven-year nominal profit (million CNY)');axs[1].legend(frameon=False,loc='upper right',fontsize=9)
    save(fig,'agriculture-risk')
    manifest=dict(classification='Derived visualizations of historical model outputs and Simulated agricultural scenarios',
      inputs=[dict(path=p.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in inputs],
      outputs=[dict(path=p.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in outputs])
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(f'Built {len(outputs)} chart files from {len(inputs)} frozen tables.')

if __name__=='__main__':main()
