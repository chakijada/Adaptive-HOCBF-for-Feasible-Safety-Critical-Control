#!/usr/bin/env python3
"""v3_e2: v3_e restarted from its first strictly admissible sample.

The common physical state is the fixed-gain/LCS state stored by v3_e at its
reported t=0.31 s (absolute reference time 1.81 s). All controllers restart
with kappa_1=2.3, kappa_2=2.0, dot(kappa_1)=0 and eta=0.35. The proposed first
QP holds eta at 0.35 exactly; subsequent terminal gains remain adaptive.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle
import numpy as np
import quadprog
from scipy.optimize import linprog
import os, sys, time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import current_qp as cq                      # the repaired (current) QP, shared with Application 1-a
MODE = os.environ.get("ACC_MODE", "both")      # ablation: both | eta_only | nu_only

ROOT=Path(__file__).resolve().parents[1]
FIG=ROOT/"figures";VID=ROOT/"videos";RES=ROOT/"results"/"test1b"   # FIG, VID: legacy diagnostics only
RES.mkdir(parents=True,exist_ok=True)

DT=.01;T0=1.81;TM=4.;TF=12.;TSH=T0;EPS=.02;ETA_MIN=.35;KMIN=.30;KCAP=8.;C1=4.;C2=5.;R=.58
ARRIVAL_TOL=.12
STD_K1=2.3;STD_K2=2.
LCS_DEPTH=3
CENTERS=np.array([[0.,-1.7],[0.,1.7],[4.,1.7],[.90,.30]])
COLORS=["#4c78a8","#e45756","#59a14f","#f28e2b"]
OBSTACLE_LABELS=["C1: lower rack","C2: upper-left rack","C3: upper-right rack","C4: central pallet"]
CONTROLLER_LABELS={"standard":"Conventional HOCBF-QP","lcs":"LCS HOCBF-QP","proposed":"Proposed adaptive HOCBF-QP"}
NOBS=len(CENTERS)
SOURCE_TIME_V3E=.31
X0=np.array([
    -0.19430812336413172, 0.2761703758910268,
     0.9424633499916794, -1.8775789743224622,
    -0.5658242954692699, 3.4880128080005077,
])
PA=np.array([0.,0.]);VA=np.array([1.,-1.0852741950359408]);AA=np.array([0.,3.9835538370337904])
PB=np.array([2.,0.]);VB=np.array([1.38362536,-1.03078988]);AB=np.array([-3.9852242,3.97578914])

@dataclass
class Row:
    h:float;hd:float;hdd:float;psi1:float;psi2:float;au:np.ndarray;anu:np.ndarray;drift:float

def polynomial():
    T=TM-2.;M=np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,2,0,0,0],
      [1,T,T**2,T**3,T**4,T**5],[0,1,2*T,3*T**2,4*T**3,5*T**4],[0,0,2,6*T,12*T**2,20*T**3]],float)
    return np.linalg.solve(M,np.vstack([PA,VA,AA,PB,VB,AB]))
COEF=polynomial()

def _reference_profile(t):
    """Exact V2_b reference until its second incompatibility endpoint at (2,0)."""
    if t<=TM:
        s=t-2.;T=TM-2.
        if s<0:return np.r_[PA+VA*s+.5*AA*s*s,VA+AA*s,AA,np.zeros(2)]
        P=np.array([1,s,s*s,s**3,s**4,s**5]);V=np.array([0,1,2*s,3*s*s,4*s**3,5*s**4]);A=np.array([0,0,2,6*s,12*s*s,20*s**3]);J=np.array([0,0,0,6,24*s,60*s*s])
        return np.r_[P@COEF,V@COEF,A@COEF,J@COEF]
    return np.r_[PB,np.zeros(6)]

SAMPLE_TIMES=np.round(np.arange(T0,TF+.5*DT,DT),10)
REFERENCE=np.array([_reference_profile(float(t)) for t in SAMPLE_TIMES])
GOAL=np.r_[PB,np.zeros(6)]

def waypoint_index(t):
    """Return the unique sample index commanded at time t (no goal look-ahead)."""
    return int(np.clip(np.rint((float(t)-T0)/DT),0,len(SAMPLE_TIMES)-1))

def reference(t):
    """Return the timestamped V2_b reference, followed by the fixed goal."""
    return REFERENCE[waypoint_index(t)].copy()

def nominal_jerk(x,t,waypoint=None):
    r=reference(t) if waypoint is None else np.asarray(waypoint);p,v,a=x[:2],x[2:4],x[4:6]
    return r[6:8]+12.*(r[:2]-p)+18.*(r[2:4]-v)+8.*(r[4:6]-a)

def obstacle_row(x,c,g,adaptive):
    p,v,a=x[:2],x[2:4],x[4:6];d=p-c;h=float(d@d-R*R);hd=float(2*d@v);hdd=float(2*(v@v+d@a));k1,k1d,k2=g
    psi1=hd+k1*h;psi2=hdd+k1d*h+k1*hd+k2*psi1;au=2*d;anu=np.array([h,psi1]) if adaptive else np.empty(0)
    drift=float(6*v@a+2*k1d*hd+k1*hdd+k2*(hdd+k1d*h+k1*hd))
    return Row(h,hd,hdd,psi1,psi2,au,anu,drift)

def qp(diag,ref,A,rhs,lo,hi):
    G=np.diag(diag)+1e-10*np.eye(len(diag));a=np.diag(diag)@ref;blocks=[A];bounds=[rhs]
    for i,q in enumerate(lo):
        if np.isfinite(q):e=np.zeros(len(diag));e[i]=1.;blocks.append(e[None]);bounds.append(np.array([q]))
    for i,q in enumerate(hi):
        if np.isfinite(q):e=np.zeros(len(diag));e[i]=-1.;blocks.append(e[None]);bounds.append(np.array([-q]))
    try:return quadprog.solve_qp(G,a,np.vstack(blocks).T,np.concatenate(bounds))[0],True
    except ValueError:return ref.copy(),False

def standard_control(x,t,centers=CENTERS,waypoint=None):
    jnom=nominal_jerk(x,t,waypoint);g=np.array([STD_K1,0.,STD_K2]);rows=[obstacle_row(x,c,g,False) for c in centers]
    A=np.array([r.au for r in rows]);beta=np.array([r.drift+ETA_MIN*r.psi2-EPS for r in rows])
    j,ok=qp(np.ones(2),jnom,A,-beta,np.full(2,-np.inf),np.full(2,np.inf))
    return j,ok,jnom,rows,A,beta,A@j+beta

def nullspace_basis(M,tol=1e-10):
    """Return an orthonormal column basis for null(M)."""
    _,s,vh=np.linalg.svd(np.asarray(M,float),full_matrices=True)
    rank=int(np.sum(s>tol*(s[0] if len(s) else 1.)))
    return vh[rank:].T.copy()

def feasibility_check(Arows,beta,mask):
    """Paper feasibility LP for the selected fixed-gain HOCBF rows."""
    mask=np.asarray(mask,bool);Ap=-np.asarray(Arows,float).T;B=np.asarray(beta,float)
    NA=nullspace_basis(Ap)
    if NA.shape[1]==0:
        return True,np.zeros(len(B)),{"rank":int(np.linalg.matrix_rank(Ap)),"nullity":0,"objective":0.}
    BA=NA.T@B;cost=mask.astype(float)-.5
    bounds=[(0.,None) if keep else (None,None) for keep in mask]
    sol=linprog(cost,A_eq=NA.T,b_eq=BA,bounds=bounds,method="highs")
    if sol.success:
        return True,sol.x,{"rank":int(np.linalg.matrix_rank(Ap)),"nullity":int(NA.shape[1]),"objective":float(sol.fun)}
    phase=linprog(np.zeros(len(B)),A_eq=NA.T,b_eq=BA,bounds=bounds,method="highs")
    if phase.success:
        return True,phase.x,{"rank":int(np.linalg.matrix_rank(Ap)),"nullity":int(NA.shape[1]),"objective":None}
    return False,None,{"rank":int(np.linalg.matrix_rank(Ap)),"nullity":int(NA.shape[1]),"objective":None}

def selection_step(Arows,beta,mask_previous,nu_previous):
    """Stateful Local Configuration Search with depth D=3."""
    mask=np.asarray(mask_previous,bool).copy();nu_prev=np.asarray(nu_previous,float).copy()
    ok,nu,meta=feasibility_check(Arows,beta,mask);actions=[];was_infeasible=not ok
    if ok:
        score=nu if nu is not None else nu_prev
        for idx in sorted(np.flatnonzero(~mask),key=lambda i:score[i],reverse=True)[:LCS_DEPTH]:
            trial=mask.copy();trial[idx]=True;tok,tnu,tmeta=feasibility_check(Arows,beta,trial)
            if tok:mask,nu,meta=trial,tnu,tmeta;actions.append(f"restore C{idx+1}")
    else:
        for idx in sorted(np.flatnonzero(mask),key=lambda i:nu_prev[i])[:LCS_DEPTH]:
            trial=mask.copy();trial[idx]=False;tok,tnu,tmeta=feasibility_check(Arows,beta,trial)
            actions.append(f"test drop C{idx+1}")
            if tok:mask,nu,meta=trial,tnu,tmeta;ok=True;actions[-1]=f"drop C{idx+1}";break
        if not ok:
            candidates=[]
            for bits in range(1<<len(mask)):
                trial=np.array([(bits>>i)&1 for i in range(len(mask))],bool)
                tok,tnu,tmeta=feasibility_check(Arows,beta,trial)
                if tok:candidates.append((int(trial.sum()),tuple(trial.astype(int)),trial,tnu,tmeta))
            _,_,mask,nu,meta=max(candidates,key=lambda z:(z[0],z[1]));actions.append("exact fallback")
    if was_infeasible and ok:
        for idx in sorted(np.flatnonzero(~mask),key=lambda i:nu[i],reverse=True)[:LCS_DEPTH]:
            trial=mask.copy();trial[idx]=True;tok,tnu,tmeta=feasibility_check(Arows,beta,trial)
            if tok:mask,nu,meta=trial,tnu,tmeta;actions.append(f"restore C{idx+1}")
    ok,nu,meta=feasibility_check(Arows,beta,mask)
    return mask,nu,ok,"; ".join(actions) if actions else "retain",meta

def lcs_control(x,t,mask_previous,nu_previous,centers=CENTERS,waypoint=None):
    """Fixed-gain CBF-QP after Local Configuration Search row selection."""
    jnom=nominal_jerk(x,t,waypoint);g=np.array([STD_K1,0.,STD_K2])
    rows=[obstacle_row(x,c,g,False) for c in centers]
    A=np.array([r.au for r in rows]);beta=np.array([r.drift+ETA_MIN*r.psi2-EPS for r in rows])
    mask,nu,selection_ok,action,meta=selection_step(A,beta,mask_previous,nu_previous)
    if np.any(mask):
        j,qp_ok=qp(np.ones(2),jnom,A[mask],-beta[mask],np.full(2,-np.inf),np.full(2,np.inf))
    else:
        j=jnom.copy();qp_ok=True
    return j,bool(selection_ok and qp_ok),jnom,rows,A,beta,A@j+beta,mask,nu,action,meta

def proposed_control(x,t,G,centers=CENTERS,waypoint=None,lock_initial_eta=False):
    """The current QP without actuation bounds (common/current_qp.py); lock_initial_eta is ignored.
    If the QP is infeasible (possible only in the nu_only ablation) the fallback is (jnom, nu_nom, eta_lo)."""
    jnom=nominal_jerk(x,t,waypoint);rows=[obstacle_row(x,c,G[i],True) for i,c in enumerate(centers)]
    dr=np.array([r.drift for r in rows]);t0=time.perf_counter()
    z,ok,A,_=cq.solve(np.array([r.au for r in rows]),np.array([r.anu for r in rows]),np.array([r.psi2 for r in rows]),dr,G,jnom,MODE)
    proposed_control.last_time=time.perf_counter()-t0
    return z,ok,jnom,rows,A,dr,A@z+dr

def plant_step(x,j):
    p,v,a=x[:2],x[2:4],x[4:6]
    return np.r_[p+DT*v+.5*DT**2*a+(DT**3/6.)*j,v+DT*a+.5*DT**2*j,a+DT*j]

def gain_step(G,nu):
    return cq.gain_step(G,nu,DT)                                    # exact zero-order-hold update

def simulate(jitter=None):
    centers=CENTERS if jitter is None else CENTERS+np.asarray(jitter)
    times=SAMPLE_TIMES.copy();x0=X0.copy();xs=x0.copy();xl=x0.copy();xp=x0.copy();G=np.tile(np.array([STD_K1,0.,STD_K2]),(NOBS,1));out=[];lcs_mask=np.ones(NOBS,dtype=bool);lcs_nu=np.ones(NOBS)
    for k,t in enumerate(times):
        command=reference(t);mission_phase="REFERENCE TRACKING" if t<=TM else "GOAL REGULATION"
        sj,sok,snom,srows,sA,sb,sres=standard_control(xs,t,centers,command)
        lj,lok,lnom,lrows,lA,lb,lres,lcs_mask,lcs_nu,lcs_action,lcs_meta=lcs_control(xl,t,lcs_mask,lcs_nu,centers,command)
        z,pok,pnom,prows,pA,pdr,pres=proposed_control(xp,t,G,centers,command,lock_initial_eta=(k==0))
        pj=z[:2];nu=z[2:2+2*NOBS];eta=z[2+2*NOBS:]
        out.append({"t":float(t-TSH),"reference_index":int(k),"mission_phase":mission_phase,"ref":command.copy(),"standard_x":xs.copy(),"lcs_x":xl.copy(),"proposed_x":xp.copy(),
          "standard_u":sj.copy(),"lcs_u":lj.copy(),"proposed_u":pj.copy(),"standard_nominal":snom.copy(),"lcs_nominal":lnom.copy(),"proposed_nominal":pnom.copy(),
          "standard_ok":bool(sok),"lcs_ok":bool(lok),"proposed_ok":bool(pok),"standard_h":np.array([r.h for r in srows]),
          "standard_phi1":np.array([r.psi1 for r in srows]),"standard_phi2":np.array([r.psi2 for r in srows]),"standard_phi3":sres+EPS,
          "lcs_h":np.array([r.h for r in lrows]),"lcs_phi1":np.array([r.psi1 for r in lrows]),"lcs_phi2":np.array([r.psi2 for r in lrows]),"lcs_phi3":lres+EPS,
          "lcs_mask":lcs_mask.copy(),"lcs_nu":lcs_nu.copy(),"lcs_action":lcs_action,"lcs_lp_objective":lcs_meta["objective"],"lcs_nullity":lcs_meta["nullity"],"lcs_A":lA.copy(),"lcs_beta":lb.copy(),"lcs_residual":lres.copy(),
          "proposed_h":np.array([r.h for r in prows]),"proposed_phi1":np.array([r.psi1 for r in prows]),
          "proposed_phi2":np.array([r.psi2 for r in prows]),"proposed_phi3":pres.copy(),"nu":nu.copy(),"eta":eta.copy(),
          "kappa":G[:,[0,2]].copy(),"barriers_active":True,"std_A":sA.copy(),"std_beta":sb.copy(),"prop_Au":np.array([r.au for r in prows]),
          "prop_offset":np.array([prows[i].anu@nu[2*i:2*i+2]+prows[i].psi2*eta[i]+prows[i].drift-EPS for i in range(NOBS)])})
        if t<TF:
            xs=plant_step(xs,sj);xl=plant_step(xl,lj);xp=plant_step(xp,pj);G=gain_step(G,nu)
    return out

def summary(data):
    sh=np.array([q["standard_h"] for q in data]);lh=np.array([q["lcs_h"] for q in data]);ph=np.array([q["proposed_h"] for q in data])
    sp1=np.array([q["standard_phi1"] for q in data]);lp1=np.array([q["lcs_phi1"] for q in data]);pp1=np.array([q["proposed_phi1"] for q in data])
    sp2=np.array([q["standard_phi2"] for q in data]);lp2=np.array([q["lcs_phi2"] for q in data]);pp2=np.array([q["proposed_phi2"] for q in data])
    sp3=np.array([q["standard_phi3"] for q in data]);lp3=np.array([q["lcs_phi3"] for q in data]);pp3=np.array([q["proposed_phi3"] for q in data]);t=np.array([q["t"] for q in data])
    masks=np.array([q["lcs_mask"] for q in data]);lres=np.array([q["lcs_residual"] for q in data]);nr=mask_runs(np.array([not q["standard_ok"] for q in data]));cr=mask_runs(sh[:,3]<0);lcr=mask_runs(lh[:,3]<0)
    active=np.array([q["barriers_active"] for q in data]);kept_residual=np.where(masks & active[:,None],lres,np.nan);activation_id=0;c4_collision_ids=np.flatnonzero(lh[:,3]<0);c4_collision_id=int(c4_collision_ids[0]) if len(c4_collision_ids) else None
    Rf=np.array([q["ref"][:2] for q in data]);S=np.array([q["standard_x"][:2] for q in data]);L=np.array([q["lcs_x"][:2] for q in data]);P=np.array([q["proposed_x"][:2] for q in data])
    refh4=np.sum((Rf-CENTERS[3])**2,axis=1)-R*R;refbad=np.flatnonzero(refh4<0)
    goal_window=t>=TM-TSH;sgoal=np.flatnonzero(goal_window & (np.linalg.norm(S-GOAL[:2],axis=1)<=ARRIVAL_TOL));lgoal=np.flatnonzero(goal_window & (np.linalg.norm(L-GOAL[:2],axis=1)<=ARRIVAL_TOL));pgoal=np.flatnonzero(goal_window & (np.linalg.norm(P-GOAL[:2],axis=1)<=ARRIVAL_TOL))
    return {"dt":DT,"samples":len(data),"obstacles":NOBS,"command_samples":len(data),
      "controller_name":"Local Configuration Search (LCS) controller","source_method":"Local Configuration Search with feasibility LP","search_depth":LCS_DEPTH,"all_obstacle_constraints_soft":True,
      "experiment":"v3_e2: v3_e restarted from its first strictly admissible sample","source_experiment":"v3_e","source_reported_time":SOURCE_TIME_V3E,"source_absolute_reference_time":T0,"common_initial_state":X0.tolist(),"common_initial_gains":{"kappa_1":STD_K1,"kappa_1_dot":0.0,"kappa_2":STD_K2,"eta":ETA_MIN},"reference_policy":"continuation of the V2_b reference from absolute time 1.81 s, followed by regulation at its endpoint; time is rebased to t0=0","closed_loop_start_time":0.0,"all_controllers_active_from_first_sample":bool(np.all(active)),"reference_end_time":TM-TSH,"goal":GOAL[:2].tolist(),"arrival_tolerance":ARRIVAL_TOL,
      "standard_final_goal_reach_time":float(t[sgoal[0]]) if len(sgoal) else None,"lcs_final_goal_reach_time":float(t[lgoal[0]]) if len(lgoal) else None,"proposed_final_goal_reach_time":float(t[pgoal[0]]) if len(pgoal) else None,
      "standard_final_position_error":float(np.linalg.norm(S[-1]-GOAL[:2])),"lcs_final_position_error":float(np.linalg.norm(L[-1]-GOAL[:2])),"proposed_final_position_error":float(np.linalg.norm(P[-1]-GOAL[:2])),
      "reference_first_c4_unsafe_time":float(t[refbad[0]]) if len(refbad) else None,
      "standard_null_samples":sum(not q["standard_ok"] for q in data),"lcs_null_samples":sum(not q["lcs_ok"] for q in data),"proposed_null_samples":sum(not q["proposed_ok"] for q in data),
      "standard_null_intervals":[[float(t[r[0]]),float(t[r[-1]]),int(len(r))] for r in nr],
      "standard_c4_collision_intervals":[[float(t[r[0]]),float(t[r[-1]]),int(len(r))] for r in cr],
      "lcs_c4_collision_intervals":[[float(t[r[0]]),float(t[r[-1]]),int(len(r))] for r in lcr],
      "lcs_c4_always_selected":bool(masks[:,3].all()),"lcs_min_kept_residual_after_filter":float(np.nanmin(kept_residual)),
      "lcs_c4_at_filter_activation":{"time":float(t[activation_id]),"h":float(lh[activation_id,3]),"phi1":float(lp1[activation_id,3]),"phi2":float(lp2[activation_id,3]),"phi3":float(lp3[activation_id,3]),"selected":bool(masks[activation_id,3])},
      "lcs_c4_at_first_collision":None if c4_collision_id is None else {"time":float(t[c4_collision_id]),"h":float(lh[c4_collision_id,3]),"phi1":float(lp1[c4_collision_id,3]),"phi2":float(lp2[c4_collision_id,3]),"phi3":float(lp3[c4_collision_id,3]),"residual":float(lres[c4_collision_id,3]),"selected":bool(masks[c4_collision_id,3])},
      "lcs_drop_intervals_by_constraint":{f"C{i+1}":[[float(t[r[0]]),float(t[r[-1]]),int(len(r))] for r in mask_runs(~masks[:,i])] for i in range(NOBS)},
      "lcs_dropped_samples_by_constraint":np.sum(~masks,axis=0).astype(int).tolist(),"lcs_min_enforced_constraints":int(masks.sum(1).min()),"lcs_max_enforced_constraints":int(masks.sum(1).max()),
      "standard_min_h":float(sh.min()),"lcs_min_h":float(lh.min()),"proposed_min_h":float(ph.min()),
      "standard_min_phi1":float(sp1.min()),"lcs_min_phi1":float(lp1.min()),"proposed_min_phi1":float(pp1.min()),"standard_min_phi2":float(sp2.min()),"lcs_min_phi2":float(lp2.min()),"proposed_min_phi2":float(pp2.min()),
      "standard_min_phi3":float(sp3.min()),"lcs_min_phi3":float(lp3.min()),"proposed_min_phi3":float(pp3.min()),
      "standard_initial_chain_min":[float(sh[0].min()),float(sp1[0].min()),float(sp2[0].min())],
      "lcs_initial_chain_min":[float(lh[0].min()),float(lp1[0].min()),float(lp2[0].min())],
      "proposed_initial_chain_min":[float(ph[0].min()),float(pp1[0].min()),float(pp2[0].min())],
      "standard_collision_obstacles":[int(i+1) for i in range(NOBS) if np.min(sh[:,i])<0],
      "lcs_collision_obstacles":[int(i+1) for i in range(NOBS) if np.min(lh[:,i])<0],
      "proposed_collision_obstacles":[int(i+1) for i in range(NOBS) if np.min(ph[:,i])<0],
      "standard_min_h_by_obstacle":sh.min(axis=0).tolist(),"lcs_min_h_by_obstacle":lh.min(axis=0).tolist(),"proposed_min_h_by_obstacle":ph.min(axis=0).tolist(),
      "lcs_max_control":float(max(np.linalg.norm(q["lcs_u"]) for q in data)),
      "proposed_max_kappa":float(np.max([q["kappa"] for q in data])),"proposed_min_kappa":float(np.min([q["kappa"] for q in data])),
      "proposed_max_eta":float(np.max([q["eta"] for q in data])),"proposed_max_control":float(max(np.linalg.norm(q["proposed_u"]) for q in data))}

def mask_runs(mask):
    ids=np.flatnonzero(mask)
    return [] if len(ids)==0 else list(np.split(ids,np.flatnonzero(np.diff(ids)>1)+1))

def flatten(data):
    names=["t","reference_index","mission_phase","ref_x","ref_y","ref_vx","ref_vy","ref_ax","ref_ay","ref_jx","ref_jy","standard_x","standard_y","standard_vx","standard_vy","standard_ax","standard_ay",
      "lcs_x","lcs_y","lcs_vx","lcs_vy","lcs_ax","lcs_ay","proposed_x","proposed_y","proposed_vx","proposed_vy","proposed_ax","proposed_ay","barriers_active","standard_ok","lcs_ok","proposed_ok",
      "standard_jx","standard_jy","lcs_jx","lcs_jy","proposed_jx","proposed_jy","lcs_action","lcs_lp_objective"]
    for prefix,n in (("standard_h",NOBS),("standard_phi1",NOBS),("standard_phi2",NOBS),("standard_phi3",NOBS),("lcs_h",NOBS),("lcs_phi1",NOBS),("lcs_phi2",NOBS),("lcs_phi3",NOBS),
                     ("lcs_keep",NOBS),("lcs_nu",NOBS),("lcs_residual",NOBS),
                     ("proposed_h",NOBS),("proposed_phi1",NOBS),("proposed_phi2",NOBS),("proposed_phi3",NOBS),
                     ("nu",2*NOBS),("eta",NOBS),("kappa",2*NOBS)):
        names.extend([f"{prefix}_{i+1}" for i in range(n)])
    with open(RES/"closed_loop_signals.csv","w",newline="") as f:
        w=csv.writer(f);w.writerow(names)
        for q in data:w.writerow([q["t"],q["reference_index"],q["mission_phase"],*q["ref"],*q["standard_x"],*q["lcs_x"],*q["proposed_x"],q["barriers_active"],q["standard_ok"],q["lcs_ok"],q["proposed_ok"],*q["standard_u"],*q["lcs_u"],*q["proposed_u"],q["lcs_action"],q["lcs_lp_objective"],
          *q["standard_h"],*q["standard_phi1"],*q["standard_phi2"],*q["standard_phi3"],*q["lcs_h"],*q["lcs_phi1"],*q["lcs_phi2"],*q["lcs_phi3"],*q["lcs_mask"].astype(int),*q["lcs_nu"],*q["lcs_residual"],*q["proposed_h"],*q["proposed_phi1"],*q["proposed_phi2"],*q["proposed_phi3"],*q["nu"],*q["eta"],*q["kappa"].ravel()])

def shade_events(ax,data):
    t=np.array([q["t"] for q in data]);null=np.array([not q["standard_ok"] for q in data]);coll=np.array([q["standard_h"][3]<0 for q in data]);lcs_drop=np.array([np.any(~q["lcs_mask"]) for q in data]);lcs_coll=np.array([q["lcs_h"][3]<0 for q in data])
    ax.axvline(TM-TSH,color="#555",ls=":",lw=1.2,label="goal regulation begins")
    for n,r in enumerate(mask_runs(null)):ax.axvspan(t[r[0]],t[r[-1]],color="#d1495b",alpha=.13,label="conventional QP infeasible" if n==0 else None)
    for n,r in enumerate(mask_runs(coll)):ax.axvspan(t[r[0]],t[r[-1]],color="#7a0015",alpha=.10,label="conventional C4 violation" if n==0 else None)
    for n,r in enumerate(mask_runs(lcs_drop)):ax.axvspan(t[r[0]],t[r[-1]],color="#6f2da8",alpha=.07,label="LCS row selection" if n==0 else None)
    for n,r in enumerate(mask_runs(lcs_coll)):ax.axvspan(t[r[0]],t[r[-1]],color="#f28e2b",alpha=.055,label="LCS C4 violation" if n==0 else None)

def mission_figure(data):
    Rf=np.array([q["ref"][:2] for q in data]);S=np.array([q["standard_x"][:2] for q in data]);L=np.array([q["lcs_x"][:2] for q in data]);P=np.array([q["proposed_x"][:2] for q in data]);bad=np.array([q["standard_h"][3]<0 for q in data]);lbad=np.array([q["lcs_h"][3]<0 for q in data]);null=np.array([not q["standard_ok"] for q in data]);drop=np.array([np.any(~q["lcs_mask"]) for q in data])
    fig,ax=plt.subplots(figsize=(11.4,6.9))
    for i,c in enumerate(CENTERS):ax.add_patch(Circle(c,R,color=COLORS[i],alpha=.34));ax.text(c[0],c[1],OBSTACLE_LABELS[i].replace(": ","\n"),ha="center",va="center",weight="bold",fontsize=9)
    moving=np.array([q["t"]<=TM-TSH for q in data]);ax.plot(Rf[moving,0],Rf[moving,1],"--",color="#555",lw=2.2,label="Prescribed reference")
    ax.plot(S[:,0],S[:,1],color="#d1495b",lw=2.2,label=CONTROLLER_LABELS["standard"]);ax.plot(L[:,0],L[:,1],color="#6f2da8",lw=2.2,label=CONTROLLER_LABELS["lcs"]);ax.plot(P[:,0],P[:,1],color="#00798c",lw=2.7,label=CONTROLLER_LABELS["proposed"])
    ax.plot(S[null,0],S[null,1],color="#d1495b",lw=10,alpha=.20,label="Infeasible conventional-QP states")
    ax.plot(S[bad,0],S[bad,1],color="#7a0015",lw=6,alpha=.88,label="Conventional C4 violation")
    ax.plot(L[drop,0],L[drop,1],color="#6f2da8",lw=9,alpha=.18,label="LCS row-selection states")
    ax.plot(L[lbad,0],L[lbad,1],color="#f28e2b",lw=4.5,alpha=.92,label="LCS C4 violation")
    ax.scatter(*S[0],marker="o",s=70,color="#edae49",edgecolor="k",zorder=6,label="Common closed-loop start")
    ax.scatter(*GOAL[:2],marker="*",s=180,color="#b279a2",edgecolor="k",zorder=7,label=r"Goal $p_g=(2,0)$")
    ax.annotate(r"all filters active; $\psi_{i,k}(0)>0$",xy=S[0],xytext=(-1.02,.65),arrowprops=dict(arrowstyle="->"),fontsize=10)
    ax.annotate("second nominal incompatibility region",xy=GOAL[:2],xytext=(2.35,-.75),arrowprops=dict(arrowstyle="->"),fontsize=10)
    ax.set(xlim=(-1.15,4.85),ylim=(-2.5,2.55),aspect="equal",xlabel=r"Horizontal position, $x$ [m]",ylabel=r"Vertical position, $y$ [m]",title="Closed-loop warehouse navigation under four safety constraints")
    ax.grid(alpha=.18);ax.legend(ncol=2,fontsize=8.4,loc="upper center",bbox_to_anchor=(.5,-.13));fig.tight_layout(rect=(0,.16,1,1));fig.savefig(FIG/"warehouse_closed_loop_paths.pdf");fig.savefig(FIG/"warehouse_closed_loop_paths.png",dpi=300);plt.close(fig)

def feasibility_safety_figure(data):
    t=np.array([q["t"] for q in data]);sok=np.array([q["standard_ok"] for q in data]);lok=np.array([q["lcs_ok"] for q in data]);pok=np.array([q["proposed_ok"] for q in data]);sh=np.array([q["standard_h"] for q in data]);lh=np.array([q["lcs_h"] for q in data]);ph=np.array([q["proposed_h"] for q in data]);S=np.array([q["standard_x"][:2] for q in data]);L=np.array([q["lcs_x"][:2] for q in data]);P=np.array([q["proposed_x"][:2] for q in data]);Rf=np.array([q["ref"][:2] for q in data])
    fig,axs=plt.subplots(3,1,figsize=(12,8.4),sharex=True)
    axs[0].step(t,sok.astype(int),where="mid",color="#d1495b",label=CONTROLLER_LABELS["standard"]);axs[0].step(t,lok.astype(int),where="mid",color="#6f2da8",ls="--",label=CONTROLLER_LABELS["lcs"]);axs[0].step(t,pok.astype(int),where="mid",color="#00798c",ls=":",label=CONTROLLER_LABELS["proposed"]);axs[0].set(yticks=[0,1],yticklabels=["infeasible","feasible"],ylabel="Admissible jerk set");axs[0].legend(ncol=3,fontsize=8)
    axs[1].plot(t,sh.min(axis=1),color="#d1495b",lw=2,label=CONTROLLER_LABELS["standard"]);axs[1].plot(t,lh.min(axis=1),color="#6f2da8",lw=2,label=CONTROLLER_LABELS["lcs"]);axs[1].plot(t,ph.min(axis=1),color="#00798c",lw=2,label=CONTROLLER_LABELS["proposed"]);axs[1].axhline(0,color="k",ls="--",label="safety boundary");axs[1].set(ylabel=r"Worst barrier value, $\min_i h_i$");axs[1].legend(ncol=4,fontsize=8)
    axs[1].set_yscale("symlog",linthresh=.03,linscale=1.35)
    axs[2].semilogy(t,np.linalg.norm(S-Rf,axis=1)+1e-5,color="#d1495b",label=CONTROLLER_LABELS["standard"]);axs[2].semilogy(t,np.linalg.norm(L-Rf,axis=1)+1e-5,color="#6f2da8",label=CONTROLLER_LABELS["lcs"]);axs[2].semilogy(t,np.linalg.norm(P-Rf,axis=1)+1e-5,color="#00798c",label=CONTROLLER_LABELS["proposed"]);axs[2].set(xlabel="Time [s]",ylabel="Position tracking error [m]",title=r"Common reference ending at $p_g=(2,0)$");axs[2].legend(ncol=3,fontsize=8)
    for ax in axs:shade_events(ax,data);ax.grid(alpha=.18)
    fig.suptitle("Feasibility, physical safety, and tracking performance",fontsize=14);fig.tight_layout(rect=(0,0,1,.965));fig.savefig(FIG/"feasibility_safety_tracking.pdf");fig.savefig(FIG/"feasibility_safety_tracking.png",dpi=300);plt.close(fig)

def barrier_chain_figure(data,prefix,stem,title):
    t=np.array([q["t"] for q in data]);fields=["h","phi1","phi2","phi3"]
    fig,axs=plt.subplots(2,2,figsize=(13,8),sharex=True)
    for ax,f in zip(axs.ravel(),fields):
        Y=np.array([q[f"{prefix}_{f}"] for q in data])
        for i in range(NOBS):
            ax.plot(t,Y[:,i],color=COLORS[i],label=OBSTACLE_LABELS[i])
            cross=np.flatnonzero(Y[:-1,i]*Y[1:,i]<0)
            if len(cross):
                tc=t[cross]-Y[cross,i]*(t[cross+1]-t[cross])/(Y[cross+1,i]-Y[cross,i])
                ax.scatter(tc,np.zeros_like(tc),s=22,marker="o",color=COLORS[i],edgecolor="k",linewidth=.45,zorder=6)
        ax.axhline(0,color="k",ls="--",lw=1,label="zero")
        if f=="phi3":ax.axhline(EPS,color="#555",ls=":",lw=1.2,label="$\\epsilon$")
        ylabel={"h":"$h_i$","phi1":"$\\psi_{i,1}$","phi2":"$\\psi_{i,2}$","phi3":"$\\psi_{i,3}$"}[f]
        ax.set_ylabel(ylabel);ax.set_title({"h":"Physical safety margins","phi1":"First HOCBF predecessor","phi2":"Second HOCBF predecessor","phi3":"Terminal HOCBF inequalities"}[f]);ax.grid(alpha=.18);shade_events(ax,data)
        ax.set_yscale("symlog",linthresh={"h":.03,"phi1":.10,"phi2":.25,"phi3":.02}[f],linscale=1.35)
        ax.text(.02,.96,f"minimum = {Y.min():.4g}",transform=ax.transAxes,va="top",fontsize=9,bbox=dict(facecolor="white",alpha=.85,edgecolor="#ddd"))
        if f=="h":
            ins=ax.inset_axes([.47,.08,.50,.38]);ins.plot(t,Y[:,3],color=COLORS[3],lw=1.8);ins.axhline(0,color="k",ls="--",lw=.8);ins.set_xlim(.9,4.15)
            pad=.018;ins.set_ylim(min(-pad,float(Y[:,3].min())-pad),max(.24,float(Y[(t>=.9)&(t<=4.15),3].max())*.35));ins.set_title("C4 safety-boundary detail",fontsize=8);ins.tick_params(labelsize=7);ins.grid(alpha=.15)
    axs[0,0].legend(ncol=2,fontsize=7.5);axs[1,0].set_xlabel("Time [s]");axs[1,1].set_xlabel("Time [s]")
    fig.suptitle(title,fontsize=15);fig.tight_layout(rect=(0,0,1,.965));fig.savefig(FIG/f"{stem}.pdf");fig.savefig(FIG/f"{stem}.png",dpi=300);plt.close(fig)

def adaptation_figure(data):
    t=np.array([q["t"] for q in data]);su=np.array([q["standard_u"] for q in data]);lu=np.array([q["lcs_u"] for q in data]);pu=np.array([q["proposed_u"] for q in data]);nu=np.array([q["nu"] for q in data]);eta=np.array([q["eta"] for q in data]);kap=np.array([q["kappa"] for q in data])
    fig,axs=plt.subplots(2,2,figsize=(13,8),sharex=True)
    axs[0,0].plot(t,su[:,0],color="#d1495b",label=r"Conventional $j_x$");axs[0,0].plot(t,su[:,1],color="#d1495b",ls="--",label=r"Conventional $j_y$");axs[0,0].plot(t,lu[:,0],color="#6f2da8",label=r"LCS $j_x$");axs[0,0].plot(t,lu[:,1],color="#6f2da8",ls="--",label=r"LCS $j_y$");axs[0,0].plot(t,pu[:,0],color="#00798c",label=r"Proposed $j_x$");axs[0,0].plot(t,pu[:,1],color="#00798c",ls="--",label=r"Proposed $j_y$");axs[0,0].set_ylabel(r"Commanded jerk [m/s$^3$]");axs[0,0].set_yscale("symlog",linthresh=1.);axs[0,0].legend(ncol=3,fontsize=7)
    axs[0,1].plot(t,eta);axs[0,1].axhline(ETA_MIN,color="k",ls="--",label=r"lower bound $\underline{\eta}$");axs[0,1].set_ylabel(r"Barrier rates, $\eta_i$");axs[0,1].legend(fontsize=8)
    axs[1,0].plot(t,nu);axs[1,0].set(ylabel=r"Gain-channel inputs, $\nu_{i\ell}$",xlabel="Time [s]")
    for i in range(NOBS):axs[1,1].plot(t,kap[:,i,0],color=COLORS[i],label=rf"$\kappa_{{{i+1},1}}$");axs[1,1].plot(t,kap[:,i,1],color=COLORS[i],ls="--",label=rf"$\kappa_{{{i+1},2}}$")
    axs[1,1].axhline(KCAP,color="k",ls=":",label="upper bound");axs[1,1].axhline(KMIN,color="k",ls="--",label="lower bound");axs[1,1].set(ylabel=r"Adaptive gains, $\kappa_{ik}$",xlabel="Time [s]");axs[1,1].legend(ncol=3,fontsize=7)
    for ax in axs.ravel():shade_events(ax,data);ax.grid(alpha=.18)
    fig.suptitle("Control effort and adaptive HOCBF gain dynamics",fontsize=15);fig.tight_layout(rect=(0,0,1,.965));fig.savefig(FIG/"control_and_gain_adaptation.pdf");fig.savefig(FIG/"control_and_gain_adaptation.png",dpi=300);plt.close(fig)

def control_mask(A,beta,xlim,ylim,n=190,mask=None):
    x=np.linspace(*xlim,n);y=np.linspace(*ylim,n);X,Y=np.meshgrid(x,y);V=np.array([A[i,0]*X+A[i,1]*Y+beta[i] for i in range(NOBS)])
    mask=np.ones(NOBS,dtype=bool) if mask is None else np.asarray(mask,bool)
    feasible=np.all(V[mask]>=0,axis=0) if np.any(mask) else np.ones_like(X,dtype=bool)
    return X,Y,V,feasible

def control_axis_envelope(A,beta,u):
    xs=[0.,float(u[0])];ys=[0.,float(u[1])]
    for row,b in zip(A,beta):
        if abs(row[0])>.18:xs.append(float(-b/row[0]))
        if abs(row[1])>.18:ys.append(float(-b/row[1]))
    for i in range(NOBS):
        for j in range(i+1,NOBS):
            M=np.vstack((A[i],A[j]));scale=np.linalg.norm(A[i])*np.linalg.norm(A[j]);det=np.linalg.det(M)
            if scale>1e-8 and abs(det)/scale>.08:
                z=np.linalg.solve(M,-np.array([beta[i],beta[j]]));xs.append(float(z[0]));ys.append(float(z[1]))
    return np.asarray(xs),np.asarray(ys)

def shared_control_limits(data):
    raw=[]
    for q in data:
        sx,sy=control_axis_envelope(q["std_A"],q["std_beta"],q["standard_u"]);lx,ly=control_axis_envelope(q["lcs_A"],q["lcs_beta"],q["lcs_u"]);px,py=control_axis_envelope(q["prop_Au"],q["prop_offset"],q["proposed_u"])
        raw.append([np.clip(1.18*np.max(np.abs(np.r_[sx,lx,px])),18.,650.),np.clip(1.18*np.max(np.abs(np.r_[sy,ly,py])),18.,650.)])
    raw=np.asarray(raw);smooth=raw.copy()
    for k in range(1,len(smooth)):smooth[k]=np.maximum(raw[k],.94*smooth[k-1])
    return smooth

def halfspace_arrow(ax,row,b,xmax,ymax,color,label):
    cand=[]
    for x in (-.55*xmax,0.,.55*xmax):
        if abs(row[1])>1e-10:
            y=(-b-row[0]*x)/row[1]
            if abs(y)<=.68*ymax:cand.append(np.array([x,y]))
    for y in (-.55*ymax,0.,.55*ymax):
        if abs(row[0])>1e-10:
            x=(-b-row[1]*y)/row[0]
            if abs(x)<=.68*xmax:cand.append(np.array([x,y]))
    if not cand:return
    p=min(cand,key=lambda z:(z[0]/xmax)**2+(z[1]/ymax)**2);n=np.array([row[0]*xmax,row[1]*ymax]);n/=max(np.linalg.norm(n),1e-12);d=np.array([.14*xmax*n[0],.14*ymax*n[1]]);e=p+d
    ax.annotate("",xy=e,xytext=p,arrowprops=dict(arrowstyle="-|>",color=color,lw=2.2,mutation_scale=15,shrinkA=0,shrinkB=0),zorder=4)
    ax.text(e[0],e[1],f" {label}",color=color,fontsize=8,weight="bold",ha="left" if d[0]>=0 else "right",va="bottom" if d[1]>=0 else "top",bbox=dict(facecolor="white",alpha=.72,edgecolor="none",pad=.5),zorder=5)

def draw_control_space(ax,A,beta,u,ok,title,xmax,ymax,residual,mask=None):
    mask=np.ones(NOBS,dtype=bool) if mask is None else np.asarray(mask,bool)
    X,Y,V,M=control_mask(A,beta,(-xmax,xmax),(-ymax,ymax),mask=mask);ax.set_facecolor("#fff8f8" if not ok else "#fbfdfc")
    if np.any(M):ax.contourf(X,Y,M.astype(float),levels=[.5,1.5],colors=["#9bd8c7"],alpha=.60)
    for i in range(NOBS):
        kept=bool(mask[i]);active=kept and ok and abs(residual[i])<5e-3;ls="-" if kept else "--";lw=3.3 if active else (1.9 if kept else 2.4)
        if np.min(V[i])<=0<=np.max(V[i]):ax.contour(X,Y,V[i],levels=[0],colors=[COLORS[i]],linewidths=lw,linestyles=[ls],alpha=1 if kept else .82)
        ax.plot([],[],color=COLORS[i],lw=lw,ls=ls,label=OBSTACLE_LABELS[i]+(" (binding)" if active else " (disregarded)" if not kept else ""))
        if kept:halfspace_arrow(ax,A[i],beta[i],xmax,ymax,COLORS[i],rf"$g_{i+1}\geq0$")
    ax.plot(u[0],u[1],"*" if ok else "X",ms=14 if ok else 11,color="#111" if ok else "#b00020",mec="white",mew=1,label="QP solution" if ok else "nominal fallback",zorder=6)
    ax.text(.025,.03,"FEASIBLE" if ok else "INFEASIBLE",transform=ax.transAxes,color="#087f5b" if ok else "#b00020",weight="bold",fontsize=11,bbox=dict(facecolor="white",alpha=.92,edgecolor="#ccc",boxstyle="round,pad=.2"))
    if np.any(~mask):ax.text(.975,.03,"dashed boundary: disregarded\nselection mask "+"".join("1" if z else "0" for z in mask),transform=ax.transAxes,ha="right",fontsize=8,bbox=dict(facecolor="white",alpha=.92,edgecolor="#ccc",boxstyle="round,pad=.2"))
    ax.set(xlim=(-xmax,xmax),ylim=(-ymax,ymax),xlabel=r"Horizontal jerk, $j_x$ [m/s$^3$]",ylabel=r"Vertical jerk, $j_y$ [m/s$^3$]",title=title+"\n"+rf"common vertical range: $\pm{ymax:.1f}$ m/s$^3$");ax.grid(alpha=.18);ax.axhline(0,color="#999",lw=.7);ax.axvline(0,color="#999",lw=.7);ax.legend(ncol=2,fontsize=6.8,loc="upper right")

def control_event_figure(data):
    first=next((i for i,q in enumerate(data) if not q["standard_ok"]),int(np.argmin([min(q["standard_h"]) for q in data])));collision=int(np.argmin([q["standard_h"][3] for q in data]));limits=shared_control_limits(data);fig,axs=plt.subplots(2,3,figsize=(17,10))
    for row,k in enumerate((first,collision)):
        q=data[k];L=limits[k];sr=q["std_A"]@q["standard_u"]+q["std_beta"];pr=q["proposed_phi3"]-EPS
        draw_control_space(axs[row,0],q["std_A"],q["std_beta"],q["standard_u"],q["standard_ok"],f"Conventional HOCBF-QP, t={q['t']:.2f} s",*L,sr)
        draw_control_space(axs[row,1],q["lcs_A"],q["lcs_beta"],q["lcs_u"],q["lcs_ok"],f"Local Configuration Search, t={q['t']:.2f} s",*L,q["lcs_residual"],q["lcs_mask"])
        draw_control_space(axs[row,2],q["prop_Au"],q["prop_offset"],q["proposed_u"],q["proposed_ok"],f"Proposed adaptive HOCBF-QP, t={q['t']:.2f} s",*L,pr)
    fig.suptitle("Admissible jerk sets at incompatibility onset and maximum C4 penetration",fontsize=14);fig.tight_layout(rect=(0,0,1,.965));fig.savefig(FIG/"control_space_witnesses.pdf");fig.savefig(FIG/"control_space_witnesses.png",dpi=300);plt.close(fig)

def lcs_diagnostics_figure(data):
    t=np.array([q["t"] for q in data]);M=np.array([q["lcs_mask"] for q in data]);V=np.array([q["lcs_nu"] for q in data]);Rr=np.array([q["lcs_residual"] for q in data])
    sok=np.array([q["standard_ok"] for q in data]);lok=np.array([q["lcs_ok"] for q in data]);pok=np.array([q["proposed_ok"] for q in data])
    fig,axs=plt.subplots(4,1,figsize=(12.5,10.5),sharex=True)
    axs[0].step(t,sok,where="mid",color="#d1495b",label=CONTROLLER_LABELS["standard"]);axs[0].step(t,lok,where="mid",color="#6f2da8",ls="--",label=CONTROLLER_LABELS["lcs"]);axs[0].step(t,pok,where="mid",color="#00798c",ls=":",label=CONTROLLER_LABELS["proposed"])
    axs[0].set(yticks=[0,1],yticklabels=["infeasible","feasible"],ylabel="Admissible jerk set",title="Controller feasibility");axs[0].legend(ncol=3,fontsize=8)
    for i in range(NOBS):axs[1].step(t,M[:,i].astype(float)+.035*i,where="mid",color=COLORS[i],label=OBSTACLE_LABELS[i])
    axs[1].set(yticks=[0,.035,.07,.105,.14,1,1.035,1.07,1.105,1.14],yticklabels=["disregard","","","","","enforce","","","",""],ylabel="LCS configuration",title="Selected barrier rows");axs[1].legend(ncol=2,fontsize=7.5)
    for i in range(NOBS):
        axs[2].plot(t,Rr[:,i],color=COLORS[i],label=OBSTACLE_LABELS[i])
        for n,r in enumerate(mask_runs(~M[:,i])):axs[2].plot(t[r],Rr[r,i],color=COLORS[i],ls="--",lw=3,label=f"C{i+1} disregarded" if n==0 else None)
    axs[2].axhline(0,color="k",ls="--",lw=.9);axs[2].set(ylabel=r"Terminal-row residual, $g_i$",yscale="symlog",title=r"Retained rows satisfy $g_i\geq0$; dashed segments are disregarded");axs[2].legend(ncol=3,fontsize=7)
    for i in range(NOBS):axs[3].plot(t,V[:,i],color=COLORS[i],label=rf"$\nu_{i+1}$")
    axs[3].axhline(0,color="k",ls="--",lw=.9);axs[3].set(xlabel="Time [s]",ylabel=r"LP certificate component, $\nu_i$",yscale="symlog",title="Feasibility-LP ranking variables");axs[3].legend(ncol=4,fontsize=8)
    for ax in axs:shade_events(ax,data);ax.grid(alpha=.18)
    fig.suptitle("Local Configuration Search: feasibility restoration and row re-entry",fontsize=15);fig.tight_layout(rect=(0,0,1,.97));fig.savefig(FIG/"lcs_selection_diagnostics.pdf");fig.savefig(FIG/"lcs_selection_diagnostics.png",dpi=300);plt.close(fig)

def c4_transition_figure(data):
    """Deterministic certificate separating terminal-row feasibility from safety."""
    t=np.array([q["t"] for q in data]);window=(t<=3.35)
    sh=np.array([q["standard_h"] for q in data]);lh=np.array([q["lcs_h"] for q in data]);ph=np.array([q["proposed_h"] for q in data])
    sp1=np.array([q["standard_phi1"] for q in data]);sp2=np.array([q["standard_phi2"] for q in data]);sp3=np.array([q["standard_phi3"] for q in data]);M=np.array([q["lcs_mask"] for q in data]);sok=np.array([q["standard_ok"] for q in data])
    fig,axs=plt.subplots(2,2,figsize=(13,7.7),sharex=True)
    axs[0,0].plot(t[window],sh[window,3],color="#d1495b",lw=2,label=CONTROLLER_LABELS["standard"]);axs[0,0].plot(t[window],lh[window,3],color="#6f2da8",lw=2,label=CONTROLLER_LABELS["lcs"]);axs[0,0].plot(t[window],ph[window,3],color="#00798c",lw=2.3,label=CONTROLLER_LABELS["proposed"]);axs[0,0].axhline(0,color="k",ls="--",label="C4 safety boundary");axs[0,0].set(ylabel=r"C4 barrier value, $h_4$",title="Physical safety relative to the central pallet");axs[0,0].legend(fontsize=7.5)
    axs[0,1].plot(t[window],sp1[window,3],color="#6f2da8",lw=2,label=r"$\psi_{4,1}$");axs[0,1].plot(t[window],sp2[window,3],color="#f28e2b",lw=2,label=r"$\psi_{4,2}$");axs[0,1].axhline(0,color="k",ls="--",label="chain boundary");axs[0,1].set(ylabel="Predecessor value",title="Conventional C4 predecessor conditions");axs[0,1].set_yscale("symlog",linthresh=.08);axs[0,1].legend(fontsize=8)
    axs[1,0].plot(t[window],sp3[window,3],color="#f28e2b",lw=2,label=r"$\psi_{4,3}$");axs[1,0].axhline(EPS,color="k",ls="--",label=r"required $\varepsilon=0.02$");axs[1,0].set(ylabel=r"Terminal value, $\psi_{4,3}$",xlabel="Time [s]",title="Conventional C4 terminal condition");axs[1,0].set_yscale("symlog",linthresh=.02);axs[1,0].legend(fontsize=8)
    axs[1,1].step(t[window],sok[window].astype(float),where="mid",color="#d1495b",lw=2,label="Conventional QP feasible");axs[1,1].step(t[window],M[window,3].astype(float),where="mid",color="#f28e2b",ls="--",lw=2,label="C4 selected by LCS");axs[1,1].set(yticks=[0,1],yticklabels=["no","yes"],ylabel="Logical state",xlabel="Time [s]",title="Feasibility and C4 selection");axs[1,1].legend(fontsize=8)
    for ax in axs.ravel():ax.grid(alpha=.18)
    fig.suptitle("C4 chain evolution after a common strictly admissible initialization",fontsize=14);fig.tight_layout(rect=(0,0,1,.965));fig.savefig(FIG/"c4_chain_transition.pdf");fig.savefig(FIG/"c4_chain_transition.png",dpi=300);plt.close(fig)

def locus_figure(data):
    t=np.array([q["t"] for q in data]);S=np.array([q["standard_x"][:2] for q in data]);L=np.array([q["lcs_x"][:2] for q in data]);P=np.array([q["proposed_x"][:2] for q in data]);null=np.array([not q["standard_ok"] for q in data]);drop=np.array([np.any(~q["lcs_mask"]) for q in data]);collision=np.array([q["standard_h"][3]<0 for q in data]);lcollision=np.array([q["lcs_h"][3]<0 for q in data]);fig,axs=plt.subplots(1,3,figsize=(17,5.1))
    for ax in axs:
        for i,c in enumerate(CENTERS):ax.add_patch(Circle(c,R,color=COLORS[i],alpha=.33));ax.text(c[0],c[1],f"C{i+1}",ha="center",va="center",weight="bold")
        ax.scatter(*GOAL[:2],marker="*",s=90,color="#b279a2",edgecolor="k",zorder=7);ax.set(xlim=(-1.1,4.8),ylim=(-2.45,2.5),aspect="equal",xlabel=r"Horizontal position, $x$ [m]",ylabel=r"Vertical position, $y$ [m]");ax.grid(alpha=.18)
    axs[0].plot(S[:,0],S[:,1],"--",color="#777",label=CONTROLLER_LABELS["standard"]);axs[0].plot(S[null,0],S[null,1],color="#d1495b",lw=10,alpha=.25);axs[0].scatter(S[null,0],S[null,1],s=15,color="#b00020",label="infeasible states");axs[0].set_title(f"Conventional QP: {null.sum()} infeasible samples");axs[0].legend(fontsize=8)
    axs[1].plot(L[:,0],L[:,1],color="#6f2da8",label=CONTROLLER_LABELS["lcs"]);axs[1].plot(L[drop,0],L[drop,1],color="#6f2da8",lw=10,alpha=.22,label="row selection active");axs[1].plot(L[lcollision,0],L[lcollision,1],color="#f28e2b",lw=6,label="C4 safety violation");axs[1].set_title(f"LCS QP: feasible throughout, {lcollision.sum()} unsafe samples");axs[1].legend(fontsize=8)
    axs[2].plot(S[:,0],S[:,1],color="#d1495b",label=CONTROLLER_LABELS["standard"]);axs[2].plot(L[:,0],L[:,1],color="#6f2da8",label=CONTROLLER_LABELS["lcs"]);axs[2].plot(P[:,0],P[:,1],color="#00798c",label=CONTROLLER_LABELS["proposed"]);axs[2].plot(S[collision,0],S[collision,1],color="#7a0015",lw=6,label="conventional C4 violation");axs[2].plot(L[lcollision,0],L[lcollision,1],color="#f28e2b",lw=4,label="LCS C4 violation");axs[2].set_title("Proposed controller preserves all safety margins");axs[2].legend(fontsize=7.5)
    fig.suptitle("Workspace loci of infeasibility and safety violation",fontsize=14);fig.tight_layout(rect=(0,0,1,.95));fig.savefig(FIG/"workspace_failure_loci.pdf");fig.savefig(FIG/"workspace_failure_loci.png",dpi=300);plt.close(fig)

def unified_control_video(data):
    S=np.array([q["standard_x"][:2] for q in data]);L=np.array([q["lcs_x"][:2] for q in data]);P=np.array([q["proposed_x"][:2] for q in data]);Rf=np.array([q["ref"][:2] for q in data]);limits=shared_control_limits(data);fig=plt.figure(figsize=(20,10));gs=fig.add_gridspec(2,4,width_ratios=(.86,1.33,1.33,1.33),height_ratios=(1.08,1.),hspace=.24,wspace=.27);scene=fig.add_subplot(gs[0,0]);bars=fig.add_subplot(gs[1,0]);sa=fig.add_subplot(gs[:,1]);la=fig.add_subplot(gs[:,2]);pa=fig.add_subplot(gs[:,3])
    def update(k):
        for ax in (scene,bars,sa,la,pa):ax.clear()
        q=data[k];xmax,ymax=limits[k]
        for i,c in enumerate(CENTERS):scene.add_patch(Circle(c,R,color=COLORS[i],alpha=.38));scene.text(c[0],c[1],f"C{i+1}",ha="center",va="center",weight="bold")
        scene.plot(Rf[:,0],Rf[:,1],"--",color="#777",lw=1.4,label="prescribed reference");scene.plot(Rf[k,0],Rf[k,1],"D",ms=6,color="#edae49",mec="k",label="current reference");scene.plot(S[:k+1,0],S[:k+1,1],color="#d1495b",lw=1.8,label="conventional HOCBF-QP");scene.plot(L[:k+1,0],L[:k+1,1],color="#6f2da8",lw=4.2,alpha=.95,solid_capstyle="round",label="LCS HOCBF-QP");scene.plot(P[:k+1,0],P[:k+1,1],color="#00798c",lw=1.9,ls=(0,(4.5,2.2)),label="proposed adaptive HOCBF-QP");scene.plot(S[k,0],S[k,1],"o",color="#d1495b",mec="k");scene.plot(L[k,0],L[k,1],"o",ms=10,color="#6f2da8",mec="k");scene.plot(P[k,0],P[k,1],"o",ms=5.5,color="#00798c",mec="k");scene.set(xlim=(-1.1,4.8),ylim=(-2.45,2.5),aspect="equal",xlabel="x [m]",ylabel="y [m]",title="Closed-loop warehouse navigation");scene.grid(alpha=.18);scene.legend(fontsize=6.2,ncol=1,loc="lower right")
        yy=np.arange(NOBS);w=.24;bars.barh(yy+w,q["standard_h"],height=w,color="#d1495b",label="Conventional");bars.barh(yy,q["lcs_h"],height=w,color="#6f2da8",label="LCS");bars.barh(yy-w,q["proposed_h"],height=w,color="#00798c",label="Proposed");bars.axvline(0,color="k");bars.set_xscale("symlog",linthresh=.05);bars.set_xticks([-10.,-.1,0.,.1,10.],labels=["-10","-0.1","0","0.1","10"]);bars.tick_params(axis="x",labelsize=8);bars.set_yticks(yy,[f"C{i+1}" for i in range(NOBS)]);bars.set(xlabel=r"Barrier value, $h_i$",title="Safety margins | LCS mask "+"".join("1" if z else "0" for z in q["lcs_mask"]));bars.grid(axis="x",alpha=.18);bars.legend(fontsize=6.7,ncol=3)
        sr=q["std_A"]@q["standard_u"]+q["std_beta"];pr=q["proposed_phi3"]-EPS;draw_control_space(sa,q["std_A"],q["std_beta"],q["standard_u"],q["standard_ok"],"Conventional HOCBF-QP",xmax,ymax,sr);draw_control_space(la,q["lcs_A"],q["lcs_beta"],q["lcs_u"],q["lcs_ok"],"Local Configuration Search",xmax,ymax,q["lcs_residual"],q["lcs_mask"]);draw_control_space(pa,q["prop_Au"],q["prop_offset"],q["proposed_u"],q["proposed_ok"],"Proposed adaptive HOCBF-QP",xmax,ymax,pr)
        ss="SAFE" if np.min(q["standard_h"])>=0 else "UNSAFE";ls="SAFE" if np.min(q["lcs_h"])>=0 else "UNSAFE";ps="SAFE" if np.min(q["proposed_h"])>=0 else "UNSAFE";fig.suptitle(f"Closed-loop comparison | t={q['t']:.2f} s | {q['mission_phase']} | Conventional: {'FEASIBLE' if q['standard_ok'] else 'INFEASIBLE/FALLBACK'}, {ss} | LCS: FEASIBLE, {ls}, mask {''.join('1' if z else '0' for z in q['lcs_mask'])} | Proposed: FEASIBLE, {ps}",fontsize=11.4,weight="bold",y=.987);return []
    ani=animation.FuncAnimation(fig,update,frames=range(0,len(data),3),interval=60,blit=False);fig.subplots_adjust(top=.91,bottom=.09,left=.045,right=.99);ani.save(VID/"three_controller_control_space_comparison.mp4",writer="ffmpeg",fps=18,dpi=100);plt.close(fig)

def diagnostics_video(data):
    t=np.array([q["t"] for q in data]);Rf=np.array([q["ref"][:2] for q in data]);S=np.array([q["standard_x"][:2] for q in data]);L=np.array([q["lcs_x"][:2] for q in data]);P=np.array([q["proposed_x"][:2] for q in data])
    sok=np.array([q["standard_ok"] for q in data]);lok=np.array([q["lcs_ok"] for q in data]);pok=np.array([q["proposed_ok"] for q in data]);su=np.array([q["standard_u"] for q in data]);lu=np.array([q["lcs_u"] for q in data]);pu=np.array([q["proposed_u"] for q in data]);masks=np.array([q["lcs_mask"] for q in data]);sh=np.array([q["standard_h"] for q in data]);lh=np.array([q["lcs_h"] for q in data]);ph=np.array([q["proposed_h"] for q in data])
    fig,axs=plt.subplots(4,4,figsize=(19.2,11.5));scene=axs[0,0];rr,=scene.plot([],[],"--",color="#777",label="prescribed reference");wd,=scene.plot([],[],"D",ms=6,color="#edae49",mec="k",label="current reference")
    for i,c in enumerate(CENTERS):scene.add_patch(Circle(c,R,color=COLORS[i],alpha=.35))
    sr,=scene.plot([],[],color="#d1495b",lw=1.8,label="Conventional");lr,=scene.plot([],[],color="#6f2da8",lw=4.2,alpha=.95,solid_capstyle="round",label="LCS");pr,=scene.plot([],[],color="#00798c",lw=1.9,ls=(0,(4.5,2.2)),label="Proposed");sd,=scene.plot([],[],"o",color="#d1495b");ld,=scene.plot([],[],"o",ms=10,color="#6f2da8");pd,=scene.plot([],[],"o",ms=5.5,color="#00798c");scene.set(xlim=(-1.1,4.8),ylim=(-2.45,2.5),aspect="equal",title="Closed-loop warehouse navigation",xlabel="x [m]",ylabel="y [m]");scene.grid(alpha=.18);scene.legend(fontsize=6,ncol=2)
    c_names=[f"C{i+1}" for i in range(NOBS)]
    series=[(np.c_[sok,lok,pok],["Conventional","LCS","Proposed"],"QP feasibility"),(np.c_[su,lu,pu],["Conv. x","Conv. y","LCS x","LCS y","Prop. x","Prop. y"],"Commanded jerk"),(masks.astype(float),c_names,"LCS constraint mask"),
      (sh,c_names,r"Conventional $h_i$"),(lh,c_names,r"LCS $h_i$"),(ph,c_names,r"Proposed $h_i$"),
      (np.array([q["standard_phi1"] for q in data]),c_names,r"Conventional $\psi_{i,1}$"),(np.array([q["lcs_phi1"] for q in data]),c_names,r"LCS $\psi_{i,1}$"),(np.array([q["proposed_phi1"] for q in data]),c_names,r"Proposed $\psi_{i,1}$"),
      (np.array([q["standard_phi2"] for q in data]),c_names,r"Conventional $\psi_{i,2}$"),(np.array([q["lcs_phi2"] for q in data]),c_names,r"LCS $\psi_{i,2}$"),(np.array([q["proposed_phi2"] for q in data]),c_names,r"Proposed $\psi_{i,2}$"),
      (np.array([q["standard_phi3"] for q in data]),c_names,r"Conventional $\psi_{i,3}$"),(np.array([q["lcs_phi3"] for q in data]),c_names,r"LCS $\psi_{i,3}$"),(np.array([q["proposed_phi3"] for q in data]),c_names,r"Proposed $\psi_{i,3}$")]
    curs=[]
    for ax,(Y,labels,title) in zip(axs.ravel()[1:],series):
        ax.plot(t,Y);ax.axhline(0,color="k",ls="--",lw=.8)
        if "i,3" in title:ax.axhline(EPS,color="#555",ls=":",lw=.8)
        ax.set_title(title);ax.grid(alpha=.16);ax.legend(labels,ncol=2,fontsize=6)
        if "$h_i$" in title:ax.set_yscale("symlog",linthresh=.03,linscale=1.35)
        elif "i,1" in title:ax.set_yscale("symlog",linthresh=.10,linscale=1.35)
        elif "i,2" in title:ax.set_yscale("symlog",linthresh=.25,linscale=1.35)
        elif "i,3" in title:ax.set_yscale("symlog",linthresh=.02,linscale=1.35)
        cur=ax.axvline(t[0],color="k",lw=1.2);curs.append(cur)
    status=fig.suptitle("",fontsize=11.2,weight="bold",y=.992)
    def update(frame):
        k=min(frame*5,len(t)-1);rr.set_data(Rf[:k+1,0],Rf[:k+1,1]);wd.set_data([Rf[k,0]],[Rf[k,1]]);sr.set_data(S[:k+1,0],S[:k+1,1]);lr.set_data(L[:k+1,0],L[:k+1,1]);pr.set_data(P[:k+1,0],P[:k+1,1]);sd.set_data([S[k,0]],[S[k,1]]);ld.set_data([L[k,0]],[L[k,1]]);pd.set_data([P[k,0]],[P[k,1]])
        for c in curs:c.set_xdata([t[k],t[k]])
        status.set_text(f"Barrier diagnostics | t={t[k]:.2f} s, {data[k]['mission_phase'].lower()} | Conventional: {'feasible' if sok[k] else 'infeasible/fallback'}, h_min={sh[k].min():.3f} | LCS: feasible, h_min={lh[k].min():.3f}, mask {''.join('1' if z else '0' for z in masks[k])} | Proposed: feasible, h_min={ph[k].min():.3f}");return [rr,wd,sr,lr,pr,sd,ld,pd,*curs,status]
    fig.tight_layout(rect=(0,0,1,.965));ani=animation.FuncAnimation(fig,update,frames=(len(t)+4)//5,interval=60,blit=False);ani.save(VID/"closed_loop_barrier_diagnostics.mp4",writer="ffmpeg",fps=18,dpi=95);plt.close(fig)

def verification(data):
    """Independent, machine-readable checks for the strict restart and run."""
    q0=data[0]
    controller_prefixes=("standard","lcs","proposed")
    initial={}
    for prefix in controller_prefixes:
        chain=[q0[f"{prefix}_h"],q0[f"{prefix}_phi1"],q0[f"{prefix}_phi2"],q0[f"{prefix}_phi3"]]
        initial[prefix]={
            "state":q0[f"{prefix}_x"].tolist(),
            "psi_0_h":chain[0].tolist(),"psi_1":chain[1].tolist(),
            "psi_2":chain[2].tolist(),"psi_3":chain[3].tolist(),
            "minimum_over_all_i_k":float(min(np.min(z) for z in chain)),
            "all_psi_i_k_nonnegative":bool(all(np.all(z>=-1e-10) for z in chain)),
            "qp_feasible":bool(q0[f"{prefix}_ok"]),
        }
    dynamics_error={}
    for prefix in controller_prefixes:
        err=[]
        for k in range(len(data)-1):
            expected=plant_step(data[k][f"{prefix}_x"],data[k][f"{prefix}_u"])
            err.append(np.max(np.abs(expected-data[k+1][f"{prefix}_x"])))
        dynamics_error[prefix]=float(max(err))
    infeasible=[q for q in data if not q["standard_ok"]]
    fallback_error=max((float(np.max(np.abs(q["standard_u"]-q["standard_nominal"]))) for q in infeasible),default=0.)
    first_bad=next((q for q in data if not q["standard_ok"]),None)
    farkas=None
    if first_bad is not None:
        A=first_bad["std_A"];beta=first_bad["std_beta"]
        cert=linprog(beta,A_eq=np.vstack((A.T,np.ones((1,NOBS)))),b_eq=np.r_[np.zeros(2),1.],bounds=[(0.,None)]*NOBS,method="highs")
        farkas={"time":first_bad["t"],"solver_success":bool(cert.success)}
        if cert.success:
            farkas.update({"lambda":cert.x.tolist(),"A_transpose_lambda_inf_norm":float(np.linalg.norm(A.T@cert.x,np.inf)),"beta_transpose_lambda":float(beta@cert.x),"strict_infeasibility_certificate":bool(beta@cert.x<-1e-9)})
    proposed_min=min(float(np.min(q["proposed_h"])) for q in data)
    report={
        "experiment":"v3_e2","dt":DT,"source_sample_in_v3_e":SOURCE_TIME_V3E,
        "common_initial_state_exact":bool(np.array_equal(q0["standard_x"],q0["lcs_x"]) and np.array_equal(q0["standard_x"],q0["proposed_x"])),
        "common_initial_state":q0["standard_x"].tolist(),
        "initial_fixed_and_proposed_gains":{"kappa_1":STD_K1,"kappa_1_dot":0.,"kappa_2":STD_K2,"eta":ETA_MIN},
        "proposed_logged_kappa_at_t0":q0["kappa"].tolist(),"proposed_eta_at_t0":q0["eta"].tolist(),
        "proposed_eta_matches_requested_initial_value":bool(np.allclose(q0["eta"],ETA_MIN,rtol=0.,atol=1e-9)),
        "all_controllers_active_at_t0":bool(all(q0[f"{p}_ok"] for p in controller_prefixes)),
        "all_controllers_have_nonnegative_complete_chain_at_t0":bool(all(initial[p]["all_psi_i_k_nonnegative"] for p in controller_prefixes)),
        "initial_chain_by_controller":initial,
        "maximum_exact_step_dynamics_error":dynamics_error,
        "standard_fallback_equals_nominal_max_error":fallback_error,
        "first_standard_infeasibility_farkas_certificate":farkas,
        "sampled_run":{"standard_qp_infeasible_samples":len(infeasible),"lcs_qp_infeasible_samples":sum(not q["lcs_ok"] for q in data),"proposed_qp_infeasible_samples":sum(not q["proposed_ok"] for q in data),"proposed_minimum_physical_barrier":proposed_min,"proposed_safe_at_every_sample":bool(proposed_min>=-1e-10)},
    }
    (RES/"verification.json").write_text(json.dumps(report,indent=2)+"\n")
    return report

def main():
    data=simulate();flatten(data);s=summary(data);(RES/"closed_loop_summary.json").write_text(json.dumps(s,indent=2)+"\n");verification(data)
    mission_figure(data);feasibility_safety_figure(data)
    barrier_chain_figure(data,"standard","conventional_hocbf_chain","Conventional fixed-gain HOCBF chain: predecessor and safety loss")
    barrier_chain_figure(data,"lcs","lcs_hocbf_chain","Local Configuration Search: feasibility recovery without predecessor recovery")
    barrier_chain_figure(data,"proposed","adaptive_hocbf_chain","Proposed adaptive HOCBF chain: feasibility and safety preserved")
    adaptation_figure(data);control_event_figure(data);lcs_diagnostics_figure(data);c4_transition_figure(data);locus_figure(data);unified_control_video(data);diagnostics_video(data);print(json.dumps(s,indent=2))

if __name__=="__main__":main()
