from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
import json, math, joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .evaluation import metrics

REQ = {"datetime","player_A","player_B","A_sets","B_sets","winner"}
WINDOWS=(5,10,20,50)

def validate_matches(df):
    missing=REQ-set(df.columns)
    if missing: raise ValueError(f"Missing required columns: {sorted(missing)}")
    x=df.copy(); x["datetime"]=pd.to_datetime(x["datetime"],utc=True,errors="coerce")
    if x["datetime"].isna().any(): raise ValueError("Invalid datetime values found")
    for c in ("A_sets","B_sets"): x[c]=pd.to_numeric(x[c],errors="coerce")
    if x[["A_sets","B_sets"]].isna().any().any(): raise ValueError("Set scores must be numeric")
    if (x.player_A==x.player_B).any(): raise ValueError("A player cannot play themself")
    bad=~(((x.A_sets==3)&x.B_sets.between(0,2))|((x.B_sets==3)&x.A_sets.between(0,2)))
    if bad.any(): raise ValueError(f"Found {int(bad.sum())} impossible/non-completed match scores")
    inferred=np.where(x.A_sets>x.B_sets,"A","B")
    supplied=np.where(x.winner.astype(str).isin(["A","B"]),x.winner.astype(str),
                      np.where(x.winner.astype(str)==x.player_A.astype(str),"A",
                      np.where(x.winner.astype(str)==x.player_B.astype(str),"B","?")))
    if np.any(supplied!=inferred): raise ValueError("Winner does not agree with match score")
    x["winner"]=inferred; x["y"]=(x.winner=="A").astype(int)
    x["a_key"] = x.get("player_A_id",x.player_A).astype(str)
    x["b_key"] = x.get("player_B_id",x.player_B).astype(str)
    keycols=["datetime","a_key","b_key"]
    if x.duplicated(keycols).any(): raise ValueError("Duplicate matches found")
    return x.sort_values(["datetime"]+(["match_id"] if "match_id" in x else [])).reset_index(drop=True)

@dataclass
class PState:
    elo: float=1500.; set_elo: float=1500.; matches:int=0; wins:int=0; sets_w:int=0; sets_l:int=0
    points_w:int=0; points_l:int=0; last_time:object=None
    history:deque=field(default_factory=lambda:deque(maxlen=50))
    times:deque=field(default_factory=lambda:deque(maxlen=100))

def _rate(num,den,prior=.5,strength=8): return (num+prior*strength)/(den+strength)
def _recent(s,w,key):
    h=list(s.history)[-w:]; n=len(h)
    if not n:return .5
    if key=="win": return _rate(sum(z[0] for z in h),n)
    if key=="set": return _rate(sum(z[1] for z in h),sum(z[1]+z[2] for z in h))
    return _rate(sum(z[3] for z in h),sum(z[3]+z[4] for z in h))

FEATURES=["elo_diff","set_elo_diff","experience_log_diff","career_win_diff","career_set_diff","career_point_diff",
          *[f"recent_win_{w}_diff" for w in WINDOWS],*[f"recent_set_{w}_diff" for w in WINDOWS],
          *[f"recent_point_{w}_diff" for w in WINDOWS],"h2h_win_diff","h2h_n_log","rest_log_diff",
          "matches_6h_diff","matches_24h_diff","sets_24h_diff","trend_diff","volatility_diff"]

def build_features(df, return_state=False):
    df=validate_matches(df); st=defaultdict(PState); h2h=defaultdict(lambda:[0,0]); rows=[]
    def snapshot(s,t):
        rest=30*24*60 if s.last_time is None else min((t-s.last_time).total_seconds()/60,30*24*60)
        t6=sum((t-z[0]).total_seconds()<=21600 for z in s.times); t24=sum((t-z[0]).total_seconds()<=86400 for z in s.times)
        sets24=sum(z[1] for z in s.times if (t-z[0]).total_seconds()<=86400)
        wins=[z[0] for z in s.history]
        return {"elo":s.elo,"set_elo":s.set_elo,"exp":math.log1p(s.matches),
          "career_win":_rate(s.wins,s.matches),"career_set":_rate(s.sets_w,s.sets_w+s.sets_l),
          "career_point":_rate(s.points_w,s.points_w+s.points_l),"rest":math.log1p(rest),"m6":t6,"m24":t24,"s24":sets24,
          "trend":_recent(s,10,"point")-_recent(s,50,"point"),"vol":float(np.std(wins[-20:])) if wins else .5,
          **{f"rw{w}":_recent(s,w,"win") for w in WINDOWS},**{f"rs{w}":_recent(s,w,"set") for w in WINDOWS},
          **{f"rp{w}":_recent(s,w,"point") for w in WINDOWS}}
    for _,r in df.iterrows():
        a,b,t=st[r.a_key],st[r.b_key],r.datetime; A,B=snapshot(a,t),snapshot(b,t); hh=h2h[(r.a_key,r.b_key)]
        f={"elo_diff":A["elo"]-B["elo"],"set_elo_diff":A["set_elo"]-B["set_elo"],"experience_log_diff":A["exp"]-B["exp"],
           "career_win_diff":A["career_win"]-B["career_win"],"career_set_diff":A["career_set"]-B["career_set"],
           "career_point_diff":A["career_point"]-B["career_point"],"h2h_win_diff":_rate(hh[0],sum(hh))-.5,
           "h2h_n_log":math.log1p(sum(hh)),"rest_log_diff":A["rest"]-B["rest"],"matches_6h_diff":A["m6"]-B["m6"],
           "matches_24h_diff":A["m24"]-B["m24"],"sets_24h_diff":A["s24"]-B["s24"],"trend_diff":A["trend"]-B["trend"],"volatility_diff":A["vol"]-B["vol"]}
        for w in WINDOWS:
            f[f"recent_win_{w}_diff"]=A[f"rw{w}"]-B[f"rw{w}"]; f[f"recent_set_{w}_diff"]=A[f"rs{w}"]-B[f"rs{w}"]; f[f"recent_point_{w}_diff"]=A[f"rp{w}"]-B[f"rp{w}"]
        rows.append(f)
        pa=pb=0
        for i in range(1,6):
            ca,cb=f"set{i}_A",f"set{i}_B"
            if ca in df and pd.notna(r.get(ca)) and pd.notna(r.get(cb)): pa+=int(r[ca]);pb+=int(r[cb])
        y=int(r.y); expected=1/(1+10**((b.elo-a.elo)/400)); margin=abs(r.A_sets-r.B_sets)/3
        delta=24*(.75+.25*margin)*(y-expected); a.elo+=delta;b.elo-=delta
        set_share=r.A_sets/(r.A_sets+r.B_sets); ed=1/(1+10**((b.set_elo-a.set_elo)/400)); ds=16*(set_share-ed);a.set_elo+=ds;b.set_elo-=ds
        for s,win,sw,sl,pw,pl in [(a,y,int(r.A_sets),int(r.B_sets),pa,pb),(b,1-y,int(r.B_sets),int(r.A_sets),pb,pa)]:
            s.matches+=1;s.wins+=win;s.sets_w+=sw;s.sets_l+=sl;s.points_w+=pw;s.points_l+=pl;s.last_time=t
            s.history.append((win,sw,sl,pw,pl));s.times.append((t,sw+sl))
        h2h[(r.a_key,r.b_key)][0]+=y;h2h[(r.a_key,r.b_key)][1]+=1-y
        h2h[(r.b_key,r.a_key)][0]+=1-y;h2h[(r.b_key,r.a_key)][1]+=y
    X=pd.DataFrame(rows)[FEATURES]
    return (X,df,st,h2h) if return_state else (X,df)

class SetkaPredictor:
    def __init__(self,seed=5212): self.seed=seed
    def fit(self,df):
        X,d,st,h2h=build_features(df,True); n=len(d)
        if n<300: raise ValueError("At least 300 completed matches are required; 2,000+ is recommended")
        i,j=int(n*.65),int(n*.82); tr,cal,te=slice(0,i),slice(i,j),slice(j,n)
        self.models=[make_pipeline(StandardScaler(),LogisticRegression(C=.5,max_iter=2000,random_state=self.seed)),
                     HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=15,l2_regularization=3,learning_rate=.055,random_state=self.seed)]
        for m in self.models:m.fit(X.iloc[tr],d.y.iloc[tr])
        pc=[m.predict_proba(X.iloc[cal])[:,1] for m in self.models]
        candidates=np.linspace(0,1,11); losses=[log_loss(d.y.iloc[cal],w*pc[0]+(1-w)*pc[1]) for w in candidates]
        self.weight=float(candidates[int(np.argmin(losses))])
        raw=self.weight*pc[0]+(1-self.weight)*pc[1]
        base=LogisticRegression(C=1e6,max_iter=1000).fit(np.log(np.clip(raw,1e-6,1-1e-6)/(1-np.clip(raw,1e-6,1-1e-6))).reshape(-1,1),d.y.iloc[cal])
        self.calibrator=base; self.states=dict(st);self.h2h=dict(h2h);self.names={**dict(zip(d.a_key,d.player_A)),**dict(zip(d.b_key,d.player_B))}
        pt=[m.predict_proba(X.iloc[te])[:,1] for m in self.models]; rawt=self.weight*pt[0]+(1-self.weight)*pt[1]
        pred=self._cal(rawt); self.report={"split":{"train":i,"calibration":j-i,"test":n-j},"test":metrics(d.y.iloc[te],pred),"blend_logistic_weight":self.weight,
          "date_ranges":{"train_end":str(d.datetime.iloc[i-1]),"calibration_end":str(d.datetime.iloc[j-1]),"test_end":str(d.datetime.iloc[-1])}}
        self._bootstrap_residual=np.asarray(d.y.iloc[te])-pred
        return self
    def _cal(self,p):
        p=np.clip(np.asarray(p),1e-6,1-1e-6); z=np.log(p/(1-p)).reshape(-1,1)
        return self.calibrator.predict_proba(z)[:,1]
    def _row(self,a_key,b_key,when=None):
        when=pd.Timestamp.now(tz="UTC") if when is None else pd.to_datetime(when,utc=True)
        # Build one dummy future match after history to reuse the single, audited sequential feature engine.
        raise RuntimeError("internal")
    def predict(self,player_a,player_b,when=None,n_boot=1000):
        keys={v:k for k,v in self.names.items()}; ak=player_a if player_a in self.states else keys.get(player_a);bk=player_b if player_b in self.states else keys.get(player_b)
        if ak is None or bk is None: return {"status":"ABSTAIN","reason":"Unknown player; collect history before predicting"}
        # Snapshot calculation via a compact mirror of training definitions.
        t=pd.Timestamp.now(tz="UTC") if when is None else pd.to_datetime(when,utc=True)
        def snap(s):
            rest=30*24*60 if s.last_time is None else max(0,min((t-s.last_time).total_seconds()/60,30*24*60));wins=[z[0] for z in s.history]
            recent=lambda w,k:_recent(s,w,k)
            return dict(elo=s.elo,set_elo=s.set_elo,exp=math.log1p(s.matches),career_win=_rate(s.wins,s.matches),career_set=_rate(s.sets_w,s.sets_w+s.sets_l),career_point=_rate(s.points_w,s.points_w+s.points_l),rest=math.log1p(rest),m6=sum((t-z[0]).total_seconds()<=21600 for z in s.times),m24=sum((t-z[0]).total_seconds()<=86400 for z in s.times),s24=sum(z[1] for z in s.times if (t-z[0]).total_seconds()<=86400),trend=recent(10,"point")-recent(50,"point"),vol=float(np.std(wins[-20:])) if wins else .5,**{f"rw{w}":recent(w,"win") for w in WINDOWS},**{f"rs{w}":recent(w,"set") for w in WINDOWS},**{f"rp{w}":recent(w,"point") for w in WINDOWS})
        A,B=snap(self.states[ak]),snap(self.states[bk]);hh=self.h2h.get((ak,bk),[0,0])
        f={"elo_diff":A['elo']-B['elo'],"set_elo_diff":A['set_elo']-B['set_elo'],"experience_log_diff":A['exp']-B['exp'],"career_win_diff":A['career_win']-B['career_win'],"career_set_diff":A['career_set']-B['career_set'],"career_point_diff":A['career_point']-B['career_point'],"h2h_win_diff":_rate(hh[0],sum(hh))-.5,"h2h_n_log":math.log1p(sum(hh)),"rest_log_diff":A['rest']-B['rest'],"matches_6h_diff":A['m6']-B['m6'],"matches_24h_diff":A['m24']-B['m24'],"sets_24h_diff":A['s24']-B['s24'],"trend_diff":A['trend']-B['trend'],"volatility_diff":A['vol']-B['vol']}
        for w in WINDOWS:f[f"recent_win_{w}_diff"]=A[f"rw{w}"]-B[f"rw{w}"];f[f"recent_set_{w}_diff"]=A[f"rs{w}"]-B[f"rs{w}"];f[f"recent_point_{w}_diff"]=A[f"rp{w}"]-B[f"rp{w}"]
        X=pd.DataFrame([f])[FEATURES]; ps=[m.predict_proba(X)[:,1][0] for m in self.models];p=float(self._cal([self.weight*ps[0]+(1-self.weight)*ps[1]])[0])
        rng=np.random.default_rng(self.seed); sims=np.clip(p+rng.choice(self._bootstrap_residual,n_boot,replace=True)/math.sqrt(max(10,min(self.states[ak].matches,self.states[bk].matches))),0,1);lo,hi=np.quantile(sims,[.025,.975])
        confidence=max(p,1-p); minhist=min(self.states[ak].matches,self.states[bk].matches)
        abstain=bool(minhist<20 or confidence<.57 or (hi-lo)>.20)
        factors=sorted({"Elo":abs(f['elo_diff']/400),"recent form":abs(f['recent_win_20_diff']),"head-to-head":abs(f['h2h_win_diff']),"fatigue/rest":abs(f['matches_24h_diff']/10)}.items(),key=lambda z:z[1],reverse=True)
        return {"status":"ABSTAIN" if abstain else "PREDICTION","player_a":self.names[ak],"player_b":self.names[bk],"p_a":p,"p_b":1-p,"winner":self.names[ak] if p>=.5 else self.names[bk],"interval_95":[float(lo),float(hi)],"confidence":"very high" if confidence>=.8 else "high" if confidence>=.72 else "moderate" if confidence>=.65 else "low","history":{"a":self.states[ak].matches,"b":self.states[bk].matches,"h2h":sum(hh)},"checks":{"known_players":True,"minimum_history":bool(minhist>=20),"clear_edge":bool(confidence>=.57),"uncertainty_acceptable":bool((hi-lo)<=.20)},"top_factors":[x[0] for x in factors[:3]],"reason":"Insufficient evidence or edge" if abstain else None}
    def save(self,path): joblib.dump(self,path)
    @classmethod
    def load(cls,path): return joblib.load(path)
