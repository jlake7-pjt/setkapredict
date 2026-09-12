from __future__ import annotations
import numpy as np
import pandas as pd

def generate(path, n_matches=5000, n_players=48, seed=5212):
    rng = np.random.default_rng(seed)
    names = [f"Demo Player {i:02d}" for i in range(n_players)]
    skill = rng.normal(0, .9, n_players)
    rows, now = [], pd.Timestamp("2023-01-01 08:00", tz="UTC")
    for i in range(n_matches):
        a, b = rng.choice(n_players, 2, replace=False)
        q = 1/(1+np.exp(-(skill[a]-skill[b] + rng.normal(0,.16))))
        sa=sb=0; sets=[]
        while sa < 3 and sb < 3:
            aw = rng.random() < q
            loser = int(rng.integers(4, 10)); winner = 11 if loser < 10 else int(rng.integers(12, 16))
            x,y=(winner,loser) if aw else (loser,winner)
            sets.append((x,y)); sa += aw; sb += not aw
        row={"match_id":f"demo_{i:06d}","datetime":now,"player_A_id":str(a),"player_A":names[a],
             "player_B_id":str(b),"player_B":names[b],"A_sets":sa,"B_sets":sb,"winner":"A" if sa>sb else "B",
             "tournament":f"Demo-{i//12:04d}","hall":f"Hall-{i%3+1}","session":["morning","day","evening"][i%3]}
        for j,(x,y) in enumerate(sets,1): row[f"set{j}_A"],row[f"set{j}_B"]=x,y
        rows.append(row)
        now += pd.Timedelta(minutes=int(rng.integers(12, 38)))
        if i and i % 900 == 0: skill += rng.normal(0,.08,n_players)
    df=pd.DataFrame(rows); path.parent.mkdir(parents=True,exist_ok=True); df.to_csv(path,index=False)
    return df

