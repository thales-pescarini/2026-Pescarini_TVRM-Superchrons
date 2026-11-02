# Fig. 6: blocking ages of the 31 ensemble specimens at 2 C/Myr (a) and sensitivity
# of the ensemble blocking age to the cooling rate (b). The ensemble age is the mean
# of the per-specimen weighted medians.
# Usage: python figures/fig6_blocking_ages.py [run directory] [output stem]
import os, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.insert(0, os.path.join(ROOT, 'model'))
from blocking import C_P, C_M, MRS_START, MRS_END, PEAK_START, RATE, elements, load_spec, wquant, gumbel_age_density

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'data', 'trout_run')
OUT = sys.argv[2] if len(sys.argv) > 2 else 'Fig6'
RATE_GRID = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

assert abs(C_P-285.33)<0.01 and abs(C_M-394.35)<0.01, (C_P,C_M)
mem=open(os.path.join(D,"ensemble_membership.txt")).read().split()
assert len(mem)==31
def short(s):
    # label as in Fig. 3: core letter + specimen number (D57, B121, G18, A07)
    for pre,L in (("1G_TT_","G"),("1B_TT_set1_","B"),("1B_TT_set2_","B"),
                  ("1A_T_1_","A"),("1A_T_2_","A"),("Dt_","D")):
        if s.startswith(pre): return f"{L}{int(s.split('_')[-1]):02d}"
    return s

spec=[]; rate_rows={r:[] for r in RATE_GRID}
for s in mem:
    path=os.path.join(D,"C2_SPECTRA",f"{s}__C2_spectrum.csv")
    T,w=load_spec(path,dec=5)
    for r in RATE_GRID:
        A,W,_=elements(T,w,r); rate_rows[r].append(wquant(A,W,0.5))
    A,W,TH=elements(T,w,RATE)
    inm=(A>=MRS_END)&(A<=MRS_START)
    d=pd.read_csv(path)
    spec.append(dict(spec=s,lab=short(s),Tsp=d.temperature_C.values,rsp=d.C2_spectrum_raw.values,
                     p10=wquant(A,W,.10),q1=wquant(A,W,.25),med=wquant(A,W,.50),
                     q3=wquant(A,W,.75),p90=wquant(A,W,.90),
                     reset=100*W.sum(),inm=100*W[inm].sum()/W.sum(),A=A,W=W,TH=TH))
spec.sort(key=lambda x: -x['med'])
med=np.array([s['med'] for s in spec]); n=len(spec)
MEAN,SD,SE=med.mean(),med.std(ddof=1),med.std(ddof=1)/np.sqrt(n)
NIN=int(((med>=MRS_END)&(med<=MRS_START)).sum())
assert abs(MEAN-475.4)<0.05 and abs(SD-8.8)<0.05 and NIN==22, (MEAN,SD,NIN)
rate_curve=[]
for r in RATE_GRID:
    mm=np.array(rate_rows[r]); rate_curve.append((r,mm.mean(),mm.std(ddof=1),int(((mm>=MRS_END)&(mm<=MRS_START)).sum())))
print(f"n={n} | mean of medians {MEAN:.2f} ± {SD:.2f} (SE {SE:.2f}) | delay {PEAK_START-MEAN:.2f} Myr | in MRS {NIN}/{n}")
for r,m_,s_,k in rate_curve: print(f"  {r:.1f} °C/Myr -> {m_:.2f} ± {s_:.2f} Ma | {k}/{n}")
pd.DataFrame([{k:v for k,v in s.items() if k not in ('Tsp','rsp','lab')} for s in spec]).round(3)\
  .drop(columns=['A','W','TH']).to_csv(OUT+"_per_specimen.csv",index=False)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.5,'axes.linewidth':.7,
                     'mathtext.fontset':'dejavusans','xtick.direction':'in','ytick.direction':'in',
                     'xtick.top':True,'ytick.right':True,'xtick.major.size':3,'ytick.major.size':3,
                     'xtick.minor.size':1.6,'xtick.major.width':.7,'ytick.major.width':.7,
                     'legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none'})
INK,SUB='0.15','0.45'
MRS_C='#1b7837'
RID_E,RID_F='#5f5470','#8c7fa0'
IN_C='#5b8f6a'
ADOPT='#9c7a3c'

AGE_HI,AGE_LO=501.0,410.0
fig=plt.figure(figsize=(7.2,4.15))
gs=GridSpec(1,2,width_ratios=[1,1],wspace=.17,left=.105,right=.985,top=.928,bottom=.114)
axA=fig.add_subplot(gs[0]); axB=fig.add_subplot(gs[1],sharex=axA)
for ax in (axA,axB):
    ax.axvspan(MRS_END,MRS_START,color=MRS_C,alpha=.10,lw=0,zorder=0)
    ax.set_xlim(AGE_HI,AGE_LO)
    ax.xaxis.set_major_locator(MultipleLocator(10)); ax.xaxis.set_minor_locator(MultipleLocator(2))
    ax.set_xlabel('Blocking age (Ma)' if ax is axA else 'Ensemble mean blocking age (Ma)')
    ax.text(.5*(MRS_END+MRS_START),1.012,'MRS',transform=ax.get_xaxis_transform(),
            ha='center',va='bottom',fontsize=7.5,color=MRS_C,fontweight='bold')
    ax.text(.5*(MRS_START+AGE_HI),1.012,'Pre-MRS',transform=ax.get_xaxis_transform(),
            ha='center',va='bottom',fontsize=7,color=SUB)
    ax.text(.5*(MRS_END+AGE_LO),1.012,'Post-MRS',transform=ax.get_xaxis_transform(),
            ha='center',va='bottom',fontsize=7,color=SUB)

# (a)
RIDGE_H=1.4
GA=np.linspace(AGE_LO-5,AGE_HI,2000)
for i,s in enumerate(spec):
    y=n-1-i
    dens=gumbel_age_density(GA,s['A'],s['W'],s['TH'])
    dens=RIDGE_H*dens/dens.max(); z=10+i
    axA.fill_between(GA,y,y+dens,facecolor=RID_F,alpha=.27,lw=0,zorder=z)
    axA.plot(GA,y+dens,color=RID_E,lw=.6,zorder=z+.2)
    axA.plot([AGE_LO,AGE_HI],[y,y],color='0.6',lw=.4,zorder=z+.1)
    axA.text(-0.012,y+.3,s['lab'],fontsize=6,va='center',ha='right',color=SUB,
             transform=axA.get_yaxis_transform())
axA.set_ylim(-.4,n+.9); axA.set_yticks([]); axA.tick_params(right=False)
axA.set_ylabel('Blocking-age distribution',labelpad=30)

axA.text(.03,.975,'a',transform=axA.transAxes,fontsize=11.5,fontweight='bold',va='top',ha='left',zorder=95,
         bbox=dict(boxstyle='square,pad=0.3',fc=(1,1,1,0.5),ec='0.25',lw=.7))

# (b)
rt=np.array([x[0] for x in rate_curve]); mu=np.array([x[1] for x in rate_curve])
sd=np.array([x[2] for x in rate_curve]); ins=np.array([x[3] for x in rate_curve])
BAND='#3A5876'
axB.fill_betweenx(rt,mu-2*sd,mu+2*sd,color=BAND,alpha=.16,lw=0,zorder=1.5)
axB.plot(mu,rt,color=BAND,lw=1.2,zorder=2)
for x,y,e,k in zip(mu,rt,sd,ins):
    good=MRS_END<=x<=MRS_START
    axB.plot(x,y,'o',ms=5.2,mfc=IN_C if good else 'white',mec=INK if good else '0.45',mew=.7,zorder=5,clip_on=False)
    lowest=y==RATE_GRID[0]; highest=y==RATE_GRID[-1]
    axB.text(x-2.6,y+(.06 if lowest else (-.06 if highest else 0)),f'{k}/{n}',ha='left',
             va='bottom' if lowest else ('top' if highest else 'center'),
             fontsize=6,color=INK if good else SUB,zorder=6)
axB.set_ylim(RATE_GRID[0],RATE_GRID[-1])
axB.yaxis.set_major_locator(MultipleLocator(1)); axB.yaxis.set_minor_locator(MultipleLocator(.5))
axB.tick_params(left=True,labelleft=True,right=True,labelright=False)
axB.set_ylabel('Cooling rate (°C Myr$^{-1}$)')
axB.legend(handles=[Line2D([],[],marker='o',ls='',ms=5,mfc=IN_C,mec=INK,mew=.7,label='Ensemble mean in MRS'),
                    Line2D([],[],marker='o',ls='',ms=5,mfc='white',mec='0.45',mew=.7,label='Ensemble mean outside MRS'),
                    Patch(facecolor=BAND,alpha=.16,edgecolor='none',label='± 2σ (specimen medians)'),
                    Line2D([],[],ls='',label=f'k/{n}: medians in MRS')],
           loc='upper right',ncol=1,fontsize=6.3,handlelength=1.6,labelspacing=.4,handletextpad=.5,
           borderaxespad=1.1,frameon=True)
axB.text(.03,.975,'b',transform=axB.transAxes,fontsize=11.5,fontweight='bold',va='top',ha='left',zorder=95,
         bbox=dict(boxstyle='square,pad=0.3',fc=(1,1,1,0.5),ec='0.25',lw=.7))

for ext in ('pdf','png','svg'):
    fig.savefig(f'{OUT}.{ext}',dpi=400 if ext=='png' else None,facecolor='white',bbox_inches='tight',pad_inches=0.02)
print('saved',OUT)
