from pathlib import Path
import math, json
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.optimize import root
from scipy.stats import qmc, rankdata, pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

OUT = Path('/mnt/data/ascvts_tbad_optimal_stopping_package')
OUT.mkdir(exist_ok=True)
(OUT/'figures').mkdir(exist_ok=True)
SEED = 20260816
rng = np.random.default_rng(SEED)

# Literature anchors are observable clinical/morphologic quantities, not ODE coefficients.
ANCHORS = {
    'entry_tear_highrisk_mm': 10.0,
    'false_lumen_highrisk_mm': 21.0,  # midpoint of guideline 20-22 mm range
    'aorta_highrisk_mm': 40.0,
    'serial_enlargement_highrisk_mm': 5.0,
    'acute_phase_day': 14.0,
    'subacute_end_day': 90.0,
}

# Provisional structural coefficients for proof-of-concept only.
BASE = dict(
    # untreated morphology dynamics, per day unless dimensionless
    k_fl_base=0.030, k_fl_tear=0.042, k_fl_momentum=0.020, k_fl_relax=0.012,
    k_da_base=0.015, k_da_fl=0.020, k_da_momentum=0.016, k_da_relax=0.008,
    k_mom_tear=0.055, k_mom_fl=0.045, k_mom_aorta=0.040, k_mom_relax=0.090,
    k_burden_tear=0.060, k_burden_fl=0.070, k_burden_growth=0.085, k_burden_clear=0.045,
    k_reserve_recover=0.030, k_reserve_burden=0.060, k_reserve_growth=0.045,
    # post-TEVAR remodeling dynamics
    k_tevar_fl=0.022, k_tevar_aorta=0.010, k_tevar_mom=0.140,
    k_tevar_burden=0.075, k_tevar_reserve=0.045,
    fl_target_mm=17.0, aorta_target_mm=39.0,
    # latent state / hazards
    q01=0.010, q12=0.012, q23=0.010, q10=0.008, q21=0.007, q32=0.005,
    hwait0=0.0005, hproc0=0.012,
    bw_tear=0.65, bw_fl=0.85, bw_aorta=0.75, bw_growth=0.95, bw_reserve=0.75,
    bp_fl=0.35, bp_aorta=0.35, bp_reserve=0.55,
    acute_penalty=20.0, acute_decay=18.0, chronic_penalty=0.45, chronic_mid=95.0, chronic_scale=12.0,
    # decision weights
    wait_penalty=0.00005, morph_penalty=0.00025, reserve_penalty=0.00020,
    post_penalty=0.008, terminal_penalty=0.050, discount=0.995,
)

RANGES = {
    'k_fl_tear':(0.025,0.060), 'k_fl_momentum':(0.010,0.032),
    'k_da_fl':(0.010,0.032), 'k_da_momentum':(0.008,0.026),
    'k_mom_tear':(0.030,0.080), 'k_mom_fl':(0.025,0.070), 'k_mom_aorta':(0.020,0.065),
    'k_burden_tear':(0.035,0.090), 'k_burden_fl':(0.040,0.100), 'k_burden_growth':(0.050,0.120),
    'k_reserve_recover':(0.018,0.050), 'k_reserve_burden':(0.035,0.090), 'k_reserve_growth':(0.025,0.070),
    'hwait0':(0.00025,0.0010), 'hproc0':(0.007,0.020),
    'acute_penalty':(10.0,30.0), 'chronic_penalty':(0.20,0.75),
}


def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))


def norm_tear(E):
    return float(sigmoid((E-ANCHORS['entry_tear_highrisk_mm'])/2.0))


def norm_fl(F):
    return float(np.clip((F-15.0)/20.0,0,1.5))


def norm_aorta(D):
    return float(np.clip((D-35.0)/25.0,0,1.5))


def rhs_pre(t,y,p,E):
    F,D,G,B,R = y
    e = norm_tear(E); f = norm_fl(F); d = norm_aorta(D)
    dF = p['k_fl_base'] + p['k_fl_tear']*e*(1-F/50.0) + p['k_fl_momentum']*G - p['k_fl_relax']*max(F-18.0,0)/30.0
    dD = p['k_da_base'] + p['k_da_fl']*f + p['k_da_momentum']*G - p['k_da_relax']*max(D-38.0,0)/30.0
    dG = (p['k_mom_tear']*e + p['k_mom_fl']*f + p['k_mom_aorta']*d)*(1-G) - p['k_mom_relax']*G
    dB = (p['k_burden_tear']*e + p['k_burden_fl']*f + p['k_burden_growth']*G)*(1-B) - p['k_burden_clear']*R*B
    dR = p['k_reserve_recover']*(1-R) - p['k_reserve_burden']*B*R - p['k_reserve_growth']*G*R
    return np.array([dF,dD,dG,dB,dR])


def rhs_post(t,y,p,E):
    F,D,G,B,R = y
    dF = -p['k_tevar_fl']*(F-p['fl_target_mm'])
    dD = -p['k_tevar_aorta']*(D-p['aorta_target_mm'])
    dG = -p['k_tevar_mom']*G
    dB = -p['k_tevar_burden']*B*max(R,0.15)
    dR = p['k_tevar_reserve']*(1-R) - 0.015*B*R
    return np.array([dF,dD,dG,dB,dR])


def integrate_pre(E=12,F0=22,D0=41,G0=0.20,B0=0.15,R0=0.78,p=None,t_end=120,step=0.25):
    p = BASE if p is None else p
    y0=np.array([F0,D0,G0,B0,R0],float)
    ts=np.arange(0,t_end+1e-9,step)
    sol=solve_ivp(lambda t,y:rhs_pre(t,y,p,E),(0,t_end),y0,t_eval=ts,rtol=1e-9,atol=1e-11,method='RK45')
    df=pd.DataFrame(sol.y.T,columns=['false_lumen_mm','aortic_diameter_mm','growth_momentum','morphologic_burden','stability_reserve'])
    df.insert(0,'day',sol.t)
    df['entry_tear_mm']=E
    df['serial_enlargement_mm']=df.aortic_diameter_mm-D0
    return df


def integrate_post(state,E=12,p=None,t_end=180,step=0.5):
    p=BASE if p is None else p
    y0=np.array(state,float)
    ts=np.arange(0,t_end+1e-9,step)
    sol=solve_ivp(lambda t,y:rhs_post(t,y,p,E),(0,t_end),y0,t_eval=ts,rtol=1e-9,atol=1e-11,method='RK45')
    return pd.DataFrame(sol.y.T,columns=['false_lumen_mm','aortic_diameter_mm','growth_momentum','morphologic_burden','stability_reserve']).assign(day=sol.t)


def latent_score(row,E):
    e=norm_tear(E); f=norm_fl(row.false_lumen_mm); d=norm_aorta(row.aortic_diameter_mm)
    return 0.22*e+0.24*f+0.20*d+0.20*row.growth_momentum+0.14*(1-row.stability_reserve)


def generator(row,E,p=None):
    p=BASE if p is None else p
    z=latent_score(row,E); R=row.stability_reserve
    q01=p['q01']*math.exp(2.0*z)
    q12=p['q12']*math.exp(2.4*(z-0.20))
    q23=p['q23']*math.exp(2.8*(z-0.38))
    q10=p['q10']*math.exp(1.2*max(R-0.55,0))
    q21=p['q21']*math.exp(1.1*max(R-0.50,0))
    q32=p['q32']*math.exp(1.0*max(R-0.45,0))
    Q=np.array([[-q01,q01,0,0],[q10,-q10-q12,q12,0],[0,q21,-q21-q23,q23],[0,0,q32,-q32]])
    return Q,z


def attach_markov(df,E,p=None):
    p=BASE if p is None else p
    probs=np.array([0.82,0.14,0.035,0.005],float)
    out=[]; dt=float(df.day.iloc[1]-df.day.iloc[0])
    for _,r in df.iterrows():
        Q,z=generator(r,E,p)
        probs=probs@expm(Q*dt); probs=np.maximum(probs,0); probs/=probs.sum()
        out.append((*probs,z))
    cols=['p_stable','p_progressive','p_highrisk','p_intervention_favored','latent_score']
    for j,c in enumerate(cols): df[c]=[x[j] for x in out]
    return df


def procedural_phase_factor(day,p):
    return 1 + p['acute_penalty']*math.exp(-day/p['acute_decay']) + p['chronic_penalty']*float(sigmoid((day-p['chronic_mid'])/p['chronic_scale']))


def hazards(row,E,p=None):
    p=BASE if p is None else p
    e=norm_tear(E); f=norm_fl(row.false_lumen_mm); d=norm_aorta(row.aortic_diameter_mm); g=row.growth_momentum; R=row.stability_reserve
    wait=p['hwait0']*math.exp(p['bw_tear']*e+p['bw_fl']*f+p['bw_aorta']*d+p['bw_growth']*g+p['bw_reserve']*(1-R))
    proc=p['hproc0']*procedural_phase_factor(float(row.day),p)*math.exp(p['bp_fl']*f+p['bp_aorta']*d+p['bp_reserve']*(1-R))
    return wait,proc


def attach_hazards(df,E,p=None):
    waits=[]; procs=[]
    for _,r in df.iterrows():
        w,o=hazards(r,E,p); waits.append(w); procs.append(o)
    df['wait_hazard']=waits; df['procedure_hazard']=procs
    return df


def decision_costs_from_path(df,E,p=None,horizon_day=120):
    p=BASE if p is None else p
    # daily states from the continuous pre-TEVAR path
    days=np.arange(0,horizon_day+1)
    dd=pd.DataFrame({'day':days})
    for c in ['false_lumen_mm','aortic_diameter_mm','growth_momentum','morphologic_burden','stability_reserve']:
        dd[c]=np.interp(days,df.day,df[c])
    dd['entry_tear_mm']=E
    dd=attach_hazards(dd,E,p)
    V=np.zeros(len(dd)); action=np.zeros(len(dd),int); opcost=np.zeros(len(dd)); waitcost=np.zeros(len(dd))
    # terminal cost if still waiting at horizon
    r=dd.iloc[-1]
    terminal=p['terminal_penalty']*(norm_fl(r.false_lumen_mm)+norm_aorta(r.aortic_diameter_mm)+r.growth_momentum+(1-r.stability_reserve))
    V[-1]=terminal; action[-1]=1
    for i in range(len(dd)-2,-1,-1):
        r=dd.iloc[i]
        # operate-now cost: procedure intensity plus a small post-remodeling residual cost
        postres=p['post_penalty']*(norm_fl(r.false_lumen_mm)+norm_aorta(r.aortic_diameter_mm)+0.5*r.growth_momentum)
        cop=r.procedure_hazard+postres
        cwait=r.wait_hazard+p['wait_penalty']+p['morph_penalty']*(norm_fl(r.false_lumen_mm)+norm_aorta(r.aortic_diameter_mm)+r.growth_momentum)+p['reserve_penalty']*(1-r.stability_reserve)+p['discount']*V[i+1]
        opcost[i]=cop; waitcost[i]=cwait
        if cop<=cwait:
            V[i]=cop; action[i]=1
        else:
            V[i]=cwait; action[i]=0
    dd['value']=V; dd['operate_now_cost']=opcost; dd['wait_cost']=waitcost; dd['operate']=action
    stopdays=dd.loc[dd.operate==1,'day']
    tau=float(stopdays.iloc[0]) if len(stopdays) else np.nan
    return dd,tau




def integrate_pre_fast(E=12,F0=22,D0=41,G0=0.20,B0=0.15,R0=0.78,p=None,t_end=120,step=1.0):
    p=BASE if p is None else p
    y=np.array([F0,D0,G0,B0,R0],float)
    rows=[]
    t=0.0
    n=int(round(t_end/step))
    for i in range(n+1):
        rows.append((t,*y))
        if i==n: break
        h=step
        k1=rhs_pre(t,y,p,E); k2=rhs_pre(t+h/2,y+h*k1/2,p,E); k3=rhs_pre(t+h/2,y+h*k2/2,p,E); k4=rhs_pre(t+h,y+h*k3,p,E)
        y=y+h*(k1+2*k2+2*k3+k4)/6
        y[0]=max(y[0],0); y[1]=max(y[1],0); y[2:]=np.clip(y[2:],0,1)
        t+=h
    df=pd.DataFrame(rows,columns=['day','false_lumen_mm','aortic_diameter_mm','growth_momentum','morphologic_burden','stability_reserve'])
    df['entry_tear_mm']=E; df['serial_enlargement_mm']=df.aortic_diameter_mm-D0
    return df

def run_reference_fast(E=12,F0=22,D0=41,G0=0.20,B0=0.15,R0=0.78,p=None):
    p=BASE if p is None else p
    df=integrate_pre_fast(E,F0,D0,G0,B0,R0,p,step=1.0)
    df=attach_hazards(df,E,p)
    dec,tau=decision_costs_from_path(df,E,p)
    return df,dec,tau


def fast_tau(E=12,F0=22,D0=41,G0=0.20,B0=0.15,R0=0.78,p=None,horizon=120):
    p=BASE if p is None else p
    y=np.array([F0,D0,G0,B0,R0],float)
    states=np.empty((horizon+1,5),float); states[0]=y
    for day in range(horizon):
        h=1.0; t=float(day)
        k1=rhs_pre(t,y,p,E); k2=rhs_pre(t+.5,y+.5*k1,p,E); k3=rhs_pre(t+.5,y+.5*k2,p,E); k4=rhs_pre(t+1,y+k3,p,E)
        y=y+(k1+2*k2+2*k3+k4)/6
        y[0]=max(y[0],0); y[1]=max(y[1],0); y[2:]=np.clip(y[2:],0,1)
        states[day+1]=y
    V=np.zeros(horizon+1); act=np.zeros(horizon+1,dtype=np.int8)
    F,D,G,B,R=states[-1]; V[-1]=p['terminal_penalty']*(norm_fl(F)+norm_aorta(D)+G+(1-R)); act[-1]=1
    for day in range(horizon-1,-1,-1):
        F,D,G,B,R=states[day]
        e=norm_tear(E); f=norm_fl(F); d=norm_aorta(D)
        wait=p['hwait0']*math.exp(p['bw_tear']*e+p['bw_fl']*f+p['bw_aorta']*d+p['bw_growth']*G+p['bw_reserve']*(1-R))
        proc=p['hproc0']*procedural_phase_factor(day,p)*math.exp(p['bp_fl']*f+p['bp_aorta']*d+p['bp_reserve']*(1-R))
        cop=proc+p['post_penalty']*(f+d+0.5*G)
        cwait=wait+p['wait_penalty']+p['morph_penalty']*(f+d+G)+p['reserve_penalty']*(1-R)+p['discount']*V[day+1]
        if cop<=cwait: V[day]=cop; act[day]=1
        else: V[day]=cwait
    idx=np.flatnonzero(act==1)
    tau=float(idx[0]) if len(idx) else np.nan
    return tau,states

def run_reference(E=12,F0=22,D0=41,G0=0.20,B0=0.15,R0=0.78,p=None):
    p=BASE if p is None else p
    df=integrate_pre(E,F0,D0,G0,B0,R0,p)
    df=attach_markov(df,E,p); df=attach_hazards(df,E,p)
    dec,tau=decision_costs_from_path(df,E,p)
    return df,dec,tau

# Reference scenario
ref,dec,ref_tau=run_reference()
ref.to_csv(OUT/'Reference_Trajectory.csv',index=False)
dec.to_csv(OUT/'Reference_Optimal_Stopping_Path.csv',index=False)

# Post-TEVAR equilibrium using root of post-treatment system
x_guess=np.array([17.0,39.0,0.0,0.0,0.95])
sol_eq=root(lambda y:rhs_post(0,y,BASE,12),x_guess)
eq=sol_eq.x
pd.DataFrame([eq],columns=['false_lumen_mm','aortic_diameter_mm','growth_momentum','morphologic_burden','stability_reserve']).to_csv(OUT/'Post_TEVAR_Equilibrium.csv',index=False)
# Jacobian finite differences

def jacobian(fun,x,eps=1e-6):
    x=np.asarray(x,float); n=len(x); J=np.zeros((n,n))
    for j in range(n):
        h=eps*max(1,abs(x[j])); xp=x.copy(); xm=x.copy(); xp[j]+=h; xm[j]-=h
        J[:,j]=(fun(xp)-fun(xm))/(2*h)
    return J
J=jacobian(lambda y:rhs_post(0,y,BASE,12),eq)
np.savetxt(OUT/'Post_TEVAR_Jacobian.csv',J,delimiter=',')
eigs=np.linalg.eigvals(J)
pd.DataFrame({'real':eigs.real,'imag':eigs.imag}).to_csv(OUT/'Post_TEVAR_Eigenvalues.csv',index=False)

# Entry tear continuation: model-derived stopping day
rows=[]
for E in np.arange(6,16.01,0.5):
    for F0 in [18,21,24,27]:
        D0=39 + 0.35*(F0-18)
        tau,_=fast_tau(E=E,F0=F0,D0=D0)
        rows.append({'entry_tear_mm':E,'initial_false_lumen_mm':F0,'initial_aortic_mm':D0,'stopping_day':tau})
pd.DataFrame(rows).to_csv(OUT/'Stopping_Boundary_Continuation.csv',index=False)

# 2D free-boundary surface at baseline D=41, varying E and F0
surf=[]
for E in np.linspace(6,16,21):
    for F0 in np.linspace(16,30,29):
        D0=39.5+0.25*(F0-18)
        tau,_=fast_tau(E=float(E),F0=float(F0),D0=float(D0))
        surf.append({'entry_tear_mm':E,'false_lumen_mm':F0,'initial_aortic_mm':D0,'stopping_day':tau})
pd.DataFrame(surf).to_csv(OUT/'EntryTear_FalseLumen_Stopping_Surface.csv',index=False)

# Scenarios
scenarios=[
    ('Reference high-risk morphology',12,22,41,0.20,0.15,0.78),
    ('Smaller tear and false lumen',8,18,39,0.12,0.10,0.85),
    ('Large entry tear',15,22,41,0.24,0.18,0.75),
    ('Expanded false lumen',12,27,43,0.28,0.20,0.72),
    ('Serial enlargement burden',12,24,45,0.34,0.24,0.68),
    ('Low stability reserve',12,22,41,0.25,0.22,0.55),
    ('Combined high-risk morphology',15,28,45,0.38,0.30,0.58),
    ('Borderline morphology',10,20,40,0.16,0.12,0.82),
]
srows=[]
for name,E,F,D,G,B,R in scenarios:
    df,dec,tau=run_reference_fast(E,F,D,G,B,R)
    df=attach_markov(df,E,BASE)
    d30=df.iloc[(np.abs(df.day-30)).argmin()]
    d90=df.iloc[(np.abs(df.day-90)).argmin()]
    srows.append(dict(scenario=name,entry_tear_mm=E,false_lumen0_mm=F,aorta0_mm=D,stopping_day=tau,
                      false_lumen_day30=d30.false_lumen_mm,aorta_day30=d30.aortic_diameter_mm,
                      enlargement_day90=d90.aortic_diameter_mm-D, reserve_day30=d30.stability_reserve,
                      highrisk_prob_day30=d30.p_highrisk, wait_hazard_day30=d30.wait_hazard,procedure_hazard_day30=d30.procedure_hazard))
pd.DataFrame(srows).to_csv(OUT/'Scenario_Results.csv',index=False)

# LHS global sensitivity of stopping day, 2000 parameter sets
keys=list(RANGES)
sampler=qmc.LatinHypercube(d=len(keys),seed=SEED)
U=sampler.random(500)
X=np.zeros_like(U)
for j,k in enumerate(keys):
    lo,hi=RANGES[k]; X[:,j]=lo+(hi-lo)*U[:,j]
outs=[]
for i in range(len(X)):
    p=BASE.copy(); p.update({k:X[i,j] for j,k in enumerate(keys)})
    tau,_=fast_tau(p=p)
    outs.append(tau)
outs=np.asarray(outs,float)
pd.DataFrame(X,columns=keys).assign(stopping_day=outs).to_csv(OUT/'LHS_Parameter_Samples.csv',index=False)
# PRCC residualized ranks
RX=np.column_stack([rankdata(X[:,j]) for j in range(X.shape[1])]); y_rank=rankdata(outs)
sens=[]
for j,k in enumerate(keys):
    others=[m for m in range(X.shape[1]) if m!=j]
    Z=np.column_stack([np.ones(len(X)),RX[:,others]])
    bx=np.linalg.lstsq(Z,RX[:,j],rcond=None)[0]; by=np.linalg.lstsq(Z,y_rank,rcond=None)[0]
    rx=RX[:,j]-Z@bx; ry=y_rank-Z@by
    r,pv=pearsonr(rx,ry); sens.append({'parameter':k,'PRCC_stopping_day':r,'p_value':pv})
pd.DataFrame(sens).to_csv(OUT/'Global_Sensitivity_PRCC.csv',index=False)

# Local elasticity +/- 1% around reference
local=[]
for k in keys:
    p1=BASE.copy(); p2=BASE.copy(); p1[k]*=1.01; p2[k]*=0.99
    t1,_=fast_tau(p=p1); t2,_=fast_tau(p=p2)
    if ref_tau and not np.isnan(ref_tau):
        elast=((t1-t2)/(2*0.01*BASE[k]))*(BASE[k]/ref_tau) if BASE[k]!=0 else np.nan
    else: elast=np.nan
    local.append({'parameter':k,'tau_plus1pct':t1,'tau_minus1pct':t2,'elasticity_stopping_day':elast})
pd.DataFrame(local).to_csv(OUT/'Local_Elasticity.csv',index=False)

# Stochastic input perturbation: computational scenarios, not empirical distribution
st=[]
for i in range(500):
    E=float(np.clip(rng.normal(12,2.0),6,18))
    F=float(np.clip(rng.normal(22,3.5),15,32))
    D=float(np.clip(rng.normal(41.5,3.0),36,52))
    G=float(np.clip(rng.normal(0.20,0.07),0.03,0.55))
    R=float(np.clip(rng.normal(0.76,0.09),0.40,0.95))
    tau,_=fast_tau(E,F,D,G,0.15,R)
    st.append({'entry_tear_mm':E,'false_lumen_mm':F,'aortic_diameter_mm':D,'growth_momentum':G,'stability_reserve':R,'stopping_day':tau})
pd.DataFrame(st).to_csv(OUT/'Stochastic_Perturbation_Results.csv',index=False)

# Synthetic model-recovery AIC. Outcome is model-generated operate-now label at day 30.
syn=[]
for i in range(800):
    E=float(rng.uniform(6,18)); F=float(rng.uniform(15,32)); D=float(rng.uniform(36,52)); G=float(rng.uniform(0.05,0.50)); R=float(rng.uniform(0.45,0.95))
    tau,states=fast_tau(E,F,D,G,0.15,R)
    op=int(tau<=30)
    serial90=float(states[90,1]-D)
    syn.append({'E':E,'F':F,'D':D,'G':G,'R':R,'serial90':serial90,'operate30':op})
syn=pd.DataFrame(syn); syn.to_csv(OUT/'Synthetic_Recovery_Data.csv',index=False)
models={
    'Entry tear only':['E'],
    'Entry tear + false lumen':['E','F'],
    'Morphology triad':['E','F','D'],
    'Triad + serial enlargement':['E','F','D','serial90'],
    'Full dynamic state':['E','F','D','serial90','G','R'],
}
mc=[]
for name,cols in models.items():
    Xs=syn[cols].values; y=syn.operate30.values
    mdl=LogisticRegression(max_iter=3000).fit(Xs,y)
    pprob=np.clip(mdl.predict_proba(Xs)[:,1],1e-9,1-1e-9)
    ll=np.sum(y*np.log(pprob)+(1-y)*np.log(1-pprob)); k=len(cols)+1
    aic=2*k-2*ll; acc=accuracy_score(y,(pprob>=0.5).astype(int))
    mc.append({'model':name,'AIC':aic,'internal_accuracy':acc})
pd.DataFrame(mc).to_csv(OUT/'Synthetic_Model_Comparison_AIC.csv',index=False)

# Numerical robustness across solvers
rob=[]
for method in ['RK45','DOP853']:
    for step in [0.1,0.25,0.5,1.0]:
        ts=np.arange(0,120+1e-9,step); y0=np.array([22,41,0.20,0.15,0.78],float)
        sol=solve_ivp(lambda t,y:rhs_pre(t,y,BASE,12),(0,120),y0,t_eval=ts,rtol=1e-9,atol=1e-11,method=method)
        dfr=pd.DataFrame(sol.y.T,columns=['F','D','G','B','R']); dfr.insert(0,'day',sol.t)
        # decision path uses interpolation from these states
        temp=dfr.rename(columns={'F':'false_lumen_mm','D':'aortic_diameter_mm','G':'growth_momentum','B':'morphologic_burden','R':'stability_reserve'})
        deci,tau=decision_costs_from_path(temp,12,BASE)
        rob.append({'solver':method,'output_step_day':step,'F_day120':temp.false_lumen_mm.iloc[-1],'D_day120':temp.aortic_diameter_mm.iloc[-1],'R_day120':temp.stability_reserve.iloc[-1],'stopping_day':tau})
pd.DataFrame(rob).to_csv(OUT/'Numerical_Robustness.csv',index=False)

# Summary log
pr=pd.read_csv(OUT/'Global_Sensitivity_PRCC.csv').sort_values('PRCC_stopping_day',key=lambda s:abs(s),ascending=False)
stdf=pd.read_csv(OUT/'Stochastic_Perturbation_Results.csv')
scdf=pd.read_csv(OUT/'Scenario_Results.csv')
summary={
    'seed':SEED,
    'reference_stopping_day':ref_tau,
    'reference_day30':{
        'false_lumen_mm':float(ref.iloc[(np.abs(ref.day-30)).argmin()].false_lumen_mm),
        'aortic_diameter_mm':float(ref.iloc[(np.abs(ref.day-30)).argmin()].aortic_diameter_mm),
        'serial_enlargement_mm':float(ref.iloc[(np.abs(ref.day-30)).argmin()].serial_enlargement_mm),
        'stability_reserve':float(ref.iloc[(np.abs(ref.day-30)).argmin()].stability_reserve),
    },
    'post_tevar_equilibrium':dict(zip(['F','D','G','B','R'],map(float,eq))),
    'eigenvalues':[{'real':float(x.real),'imag':float(x.imag)} for x in eigs],
    'stochastic_stopping_day':{'median':float(stdf.stopping_day.median()),'p2_5':float(stdf.stopping_day.quantile(.025)),'p97_5':float(stdf.stopping_day.quantile(.975))},
    'top_prcc':pr.head(6).to_dict(orient='records'),
    'scenarios':scdf[['scenario','stopping_day']].to_dict(orient='records'),
}
(OUT/'run_output.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
