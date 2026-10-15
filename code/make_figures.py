from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import ListedColormap

BASE=Path('/mnt/data/ascvts_tbad_optimal_stopping_package')
FIG=BASE/'figures'; FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.titlesize':13,'axes.labelsize':10,'figure.dpi':160})

def save(name):
    plt.savefig(FIG/name,bbox_inches='tight',dpi=240)
    plt.close()

# Figure 1 - architecture
fig,ax=plt.subplots(figsize=(13,7)); ax.axis('off')
boxes=[
(.03,.72,.20,.14,'CTA Morphology','Entry tear diameter\nFalse lumen size\nSerial aortic diameter'),
(.30,.72,.22,.14,'Continuous Dynamics','False lumen F(t)\nAortic diameter D(t)\nGrowth momentum G(t)'),
(.59,.72,.23,.14,'Reserve and Burden','Morphologic burden B(t)\nStability reserve R(t)\nLatent regime state'),
(.30,.40,.22,.14,'Competing Hazards','Waiting hazard λW(t)\nProcedural hazard λT(t)\nPhase-dependent risk'),
(.59,.40,.23,.14,'Optimal Stopping','Bellman recursion\nContinuation region C\nIntervention region S'),
(.39,.10,.31,.15,'Robustness and Validation','Free-boundary continuation | LHS-PRCC\nStochastic perturbation | synthetic AIC\nFuture registry calibration')]
colors=['#dbeafe','#e0f2fe','#dcfce7','#fce7f3','#fef3c7','#ede9fe']
for (x,y,w,h,t,s),c in zip(boxes,colors):
    p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.015,rounding_size=.02',fc=c,ec='#334155',lw=1.4); ax.add_patch(p)
    ax.text(x+w/2,y+h*.68,t,ha='center',va='center',weight='bold',fontsize=11); ax.text(x+w/2,y+h*.31,s,ha='center',va='center',fontsize=9)
for a,b in [((.23,.79),(.30,.79)),((.52,.79),(.59,.79)),((.41,.72),(.41,.54)),((.70,.72),(.70,.54)),((.52,.47),(.59,.47)),((.41,.40),(.50,.25)),((.70,.40),(.60,.25))]:
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=14,lw=1.5,color='#475569'))
ax.text(.5,.95,'Optimal-Stopping Architecture for Preemptive TEVAR',ha='center',weight='bold',fontsize=17)
ax.text(.5,.90,'Literature-parameterized mathematical proof-of-concept — no patient-level calibration',ha='center',fontsize=10,style='italic')
save('Figure_1_Model_Architecture.png')

# Figure 2 - reference morphology trajectory
r=pd.read_csv(BASE/'Reference_Trajectory.csv')
fig,axs=plt.subplots(2,2,figsize=(12.5,8.5)); axs=axs.ravel()
axs[0].plot(r.day,r.false_lumen_mm,lw=2.3,label='False lumen diameter'); axs[0].axhline(21,ls='--',lw=1.3,label='Guideline anchor 20–22 mm'); axs[0].set_ylabel('mm'); axs[0].set_title('False Lumen Expansion')
axs[1].plot(r.day,r.aortic_diameter_mm,lw=2.3,label='Total aortic diameter'); axs[1].axhline(40,ls='--',lw=1.3,label='High-risk anchor 40 mm'); axs[1].set_ylabel('mm'); axs[1].set_title('Serial Aortic Enlargement')
axs[2].plot(r.day,r.growth_momentum,lw=2.2,label='Growth momentum'); axs[2].plot(r.day,r.morphologic_burden,lw=2.0,label='Morphologic burden'); axs[2].plot(r.day,r.stability_reserve,lw=2.0,label='Stability reserve'); axs[2].set_ylim(0,1); axs[2].set_title('Dynamic Disease State')
axs[3].plot(r.day,r.p_stable,lw=1.8,label='P(stable)'); axs[3].plot(r.day,r.p_progressive,lw=1.8,label='P(progressive)'); axs[3].plot(r.day,r.p_highrisk,lw=1.8,label='P(high-risk)'); axs[3].plot(r.day,r.p_intervention_favored,lw=1.8,label='P(intervention-favored)'); axs[3].set_ylim(0,1); axs[3].set_title('Latent Regime Occupancy')
for ax in axs: ax.set_xlabel('Time since dissection (days)'); ax.grid(alpha=.22); ax.legend(frameon=False,fontsize=8)
fig.suptitle('Reference Morphologic Trajectory Under Continued Surveillance',fontsize=15.5,weight='bold'); fig.tight_layout(); save('Figure_2_Reference_Trajectory.png')

# Figure 3 - competing hazards and decision path
D=pd.read_csv(BASE/'Reference_Optimal_Stopping_Path.csv')
fig,axs=plt.subplots(1,2,figsize=(13,5.5))
axs[0].plot(D.day,D.wait_hazard,lw=2.3,label='Waiting/deterioration intensity'); axs[0].plot(D.day,D.procedure_hazard,lw=2.3,label='Procedure-associated intensity'); axs[0].axvline(33,ls='--',lw=1.4,label='Reference model stopping day'); axs[0].set_yscale('log'); axs[0].set_xlabel('Time since dissection (days)'); axs[0].set_ylabel('Theoretical intensity'); axs[0].set_title('Competing Hazards')
axs[1].plot(D.day,D.operate_now_cost,lw=2.2,label='Operate-now value'); axs[1].plot(D.day,D.wait_cost,lw=2.2,label='Wait-one-step value'); axs[1].axvline(33,ls='--',lw=1.4,label='First stopping point'); axs[1].fill_between(D.day,0,1,where=(D.operate==0),alpha=.10,transform=axs[1].get_xaxis_transform(),label='Continuation region'); axs[1].fill_between(D.day,0,1,where=(D.operate==1),alpha=.10,transform=axs[1].get_xaxis_transform(),label='Intervention region'); axs[1].set_xlabel('Time since dissection (days)'); axs[1].set_ylabel('Model value'); axs[1].set_title('Finite-Horizon Optimal-Stopping Path')
for ax in axs: ax.grid(alpha=.2); ax.legend(frameon=False,fontsize=8)
fig.tight_layout(); save('Figure_3_Competing_Hazards_and_Stopping_Path.png')

# Figure 4 - entry tear x false lumen stopping surface
S=pd.read_csv(BASE/'EntryTear_FalseLumen_Stopping_Surface.csv')
piv=S.pivot(index='false_lumen_mm',columns='entry_tear_mm',values='stopping_day')
fig,ax=plt.subplots(figsize=(10.8,7.2)); im=ax.imshow(piv.values,origin='lower',aspect='auto',extent=[piv.columns.min(),piv.columns.max(),piv.index.min(),piv.index.max()],cmap='viridis')
cs=ax.contour(piv.columns,piv.index,piv.values,levels=[28,32,36,40,44],colors='white',linewidths=1.2); ax.clabel(cs,fmt=lambda x:f'{int(x)} d',fontsize=8)
ax.axvline(10,ls='--',lw=1.3,color='black'); ax.axhline(21,ls='--',lw=1.3,color='black'); ax.scatter([12],[22],marker='*',s=180,c='black',label='Reference')
ax.set_xlabel('Entry tear diameter (mm)'); ax.set_ylabel('Initial false lumen diameter (mm)'); ax.set_title('Model-Derived Preemptive TEVAR Stopping Surface',weight='bold'); fig.colorbar(im,ax=ax,label='Optimal stopping day (proof-of-concept)'); ax.legend(frameon=False)
save('Figure_4_Stopping_Surface.png')

# Figure 5 - continuation curves
C=pd.read_csv(BASE/'Stopping_Boundary_Continuation.csv')
fig,ax=plt.subplots(figsize=(10.5,6.5))
for fl,g in C.groupby('initial_false_lumen_mm'):
    ax.plot(g.entry_tear_mm,g.stopping_day,marker='o',ms=3.5,lw=1.8,label=f'FL {fl:.0f} mm')
ax.axvline(10,ls='--',lw=1.2,color='black',label='Entry-tear high-risk anchor'); ax.axhspan(14,90,alpha=.07,label='Subacute interval 14–90 d')
ax.set_xlabel('Entry tear diameter (mm)'); ax.set_ylabel('Model-derived stopping day'); ax.set_title('Free-Boundary Continuation Across Entry Tear and False Lumen Size',weight='bold'); ax.grid(alpha=.2); ax.legend(frameon=False,ncol=2)
save('Figure_5_Free_Boundary_Continuation.png')

# Figure 6 - scenarios
sc=pd.read_csv(BASE/'Scenario_Results.csv')
fig,ax=plt.subplots(figsize=(11,6.5)); y=np.arange(len(sc)); ax.barh(y,sc.stopping_day); ax.set_yticks(y,sc.scenario); ax.invert_yaxis(); ax.axvline(14,ls=':',lw=1.2,color='black'); ax.axvline(90,ls=':',lw=1.2,color='black');
for i,v in enumerate(sc.stopping_day): ax.text(v+.4,i,f'{v:.0f} d',va='center',fontsize=9)
ax.set_xlabel('Model-derived stopping day'); ax.set_title('Scenario-Specific Shift in Candidate Preemptive TEVAR Timing',weight='bold'); ax.grid(axis='x',alpha=.2); save('Figure_6_Scenario_Stopping_Days.png')

# Figure 7 - sensitivity
pr=pd.read_csv(BASE/'Global_Sensitivity_PRCC.csv').sort_values('PRCC_stopping_day')
fig,ax=plt.subplots(figsize=(10.5,7.5)); vals=pr.PRCC_stopping_day; cols=['#dc2626' if x<0 else '#16a34a' for x in vals]; ax.barh(pr.parameter,vals,color=cols,alpha=.86); ax.axvline(0,color='black',lw=1); ax.set_xlim(-1,1); ax.grid(axis='x',alpha=.2); ax.set_xlabel('PRCC with model-derived stopping day'); ax.set_title('Global Sensitivity Across 500 Latin-Hypercube Parameter Sets',weight='bold'); save('Figure_7_Global_Sensitivity.png')

# Figure 8 - stochastic distribution
st=pd.read_csv(BASE/'Stochastic_Perturbation_Results.csv')
fig,ax=plt.subplots(figsize=(10.5,6.2)); ax.hist(st.stopping_day,bins=np.arange(st.stopping_day.min()-0.5,st.stopping_day.max()+1.5,1),alpha=.78); med=st.stopping_day.median(); lo=st.stopping_day.quantile(.025); hi=st.stopping_day.quantile(.975); ax.axvline(med,ls='--',color='black',label=f'Median {med:.0f} d'); ax.axvspan(lo,hi,alpha=.12,label=f'2.5–97.5% {lo:.0f}–{hi:.0f} d'); ax.set_xlabel('Model-derived stopping day'); ax.set_ylabel('Simulation count'); ax.set_title('Stochastic Perturbation of Morphologic Inputs',weight='bold'); ax.grid(alpha=.18); ax.legend(frameon=False); save('Figure_8_Stochastic_Stopping_Distribution.png')

# Figure 9 - AIC
mc=pd.read_csv(BASE/'Synthetic_Model_Comparison_AIC.csv').sort_values('AIC',ascending=False)
fig,ax=plt.subplots(figsize=(10.5,6)); bars=ax.barh(mc.model,mc.AIC)
for b,a,acc in zip(bars,mc.AIC,mc.internal_accuracy): ax.text(a+8,b.get_y()+b.get_height()/2,f'AIC {a:.1f} | internal acc {acc:.3f}',va='center',fontsize=8)
ax.set_xlabel('Synthetic AIC (lower is better)'); ax.set_title('Internal Model-Recovery Comparison — Not Clinical Validation',weight='bold'); ax.grid(axis='x',alpha=.2); save('Figure_9_Synthetic_AIC.png')

# Figure 10 - eigenvalues
E=pd.read_csv(BASE/'Post_TEVAR_Eigenvalues.csv'); fig,ax=plt.subplots(figsize=(8,6)); ax.scatter(E.real,E.imag,s=110); ax.axvline(0,color='black',ls='--'); ax.axhline(0,color='#64748b',lw=.8); ax.set_xlabel('Real part'); ax.set_ylabel('Imaginary part'); ax.set_title('Post-TEVAR Remodeling Equilibrium Eigenvalues',weight='bold'); ax.grid(alpha=.2); save('Figure_10_Post_TEVAR_Eigenvalues.png')

# Figure 11 - validation roadmap
fig,ax=plt.subplots(figsize=(14,3.2)); ax.axis('off'); steps=['Evidence audit','Structural\nidentifiability','Retrospective\nCTA calibration','Observed-data\nmodel comparison','Causal\nanalysis','External\nvalidation','Prospective\nimpact study']
for i,s in enumerate(steps):
    x=.02+i*.14; p=FancyBboxPatch((x,.34),.12,.30,boxstyle='round,pad=.01',fc='#dbeafe' if i<3 else '#e2e8f0',ec='#334155'); ax.add_patch(p); ax.text(x+.06,.49,s,ha='center',va='center',fontsize=9,weight='bold')
    if i<len(steps)-1: ax.add_patch(FancyArrowPatch((x+.12,.49),(x+.14,.49),arrowstyle='-|>',mutation_scale=12,color='#475569'))
ax.text(.5,.84,'Roadmap From Mathematical Stopping Rule to Clinically Evaluated Decision Support',ha='center',fontsize=15,weight='bold'); ax.text(.5,.16,'No patient-level TEVAR timing threshold is claimed before calibration and external validation',ha='center',fontsize=9,style='italic'); save('Figure_11_Validation_Roadmap.png')

# Table figures helper

def table_image(df,title,name,fontsize=7.5,figsize=(13,5.5)):
    fig,ax=plt.subplots(figsize=figsize); ax.axis('off'); ax.set_title(title,fontsize=14,weight='bold',pad=12)
    tab=ax.table(cellText=df.values,colLabels=df.columns,loc='center',cellLoc='left',colLoc='center'); tab.auto_set_font_size(False); tab.set_fontsize(fontsize); tab.scale(1,1.45)
    for (r,c),cell in tab.get_celld().items():
        if r==0: cell.set_text_props(weight='bold',color='white'); cell.set_facecolor('#334155')
        elif r%2==0: cell.set_facecolor('#f1f5f9')
    save(name)

states=pd.DataFrame([
['F(t)','False lumen diameter','mm','Longitudinal false-lumen expansion'],['D(t)','Total aortic diameter','mm','Serial aortic enlargement'],['G(t)','Growth momentum','0–1','Latent acceleration of morphologic progression'],['B(t)','Morphologic burden','0–1','Accumulated adverse morphology'],['R(t)','Stability reserve','0–1','Capacity to remain in surveillance-favorable state']
],columns=['Symbol','State','Scale','Interpretation']); table_image(states,'Table 1 as Figure — Dynamic State Variables','Figure_12_Table1_State_Variables.png',8,(13,4.8))

anchors=pd.DataFrame([
['Entry tear','>10 mm','ACC/AHA and ESC high-risk imaging feature'],['False lumen diameter','>20–22 mm','ACC/AHA high-risk imaging feature'],['Maximum aortic diameter','>40 mm','ACC/AHA/ESC high-risk imaging feature'],['Serial enlargement','>5 mm between scans','ACC/AHA high-risk imaging feature'],['Timing phase','Acute <14 d; subacute 14–90 d','Guideline/meta-analysis phase anchor']
],columns=['Anchor','Literature-informed value','Role']); table_image(anchors,'Table 2 as Figure — Literature-Informed Clinical Anchors','Figure_13_Table2_Clinical_Anchors.png',8,(13,4.7))

scsmall=sc[['scenario','entry_tear_mm','false_lumen0_mm','aorta0_mm','stopping_day']].copy(); scsmall.columns=['Scenario','Entry tear','FL baseline','Aorta baseline','Stopping day']; table_image(scsmall.round(2),'Table 3 as Figure — Scenario Outputs','Figure_14_Table3_Scenario_Outputs.png',7,(14,6))
pars=pd.read_csv(BASE/'Model_Parameters.csv').head(18) if (BASE/'Model_Parameters.csv').exists() else pd.DataFrame()
if len(pars): table_image(pars.round(4),'Table 4 as Figure — Parameter Provenance','Figure_15_Table4_Parameter_Provenance.png',6.4,(14,8))
