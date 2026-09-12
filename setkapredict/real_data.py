from __future__ import annotations
import re
import pandas as pd

PUBLIC_DATA_URL = "https://data.scorenetwork.org/data/table_tennis-sept2022.csv"

def _player_key(name: str) -> str:
    """Stable fallback key when the public archive has names but no Setka ID."""
    return re.sub(r"[^a-z0-9]+", "-", str(name).strip().lower()).strip("-")

def clean_public_archive(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required = {"X","Date","Time","Player1","Player2","Sets_P1","Sets_P2","HomeWinner"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Archive schema changed; missing {sorted(missing)}")
    d = raw.copy()
    input_rows = len(d)
    d["datetime"] = pd.to_datetime(
        d["Date"].astype(str) + " " + d["Time"].astype(str),
        format="%m/%d/%Y %H:%M:%S", utc=True, errors="coerce"
    )
    for c in ["Sets_P1","Sets_P2","HomeWinner"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["player_A"] = d["Player1"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    d["player_B"] = d["Player2"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    d["player_A_id"] = d["player_A"].map(_player_key)
    d["player_B_id"] = d["player_B"].map(_player_key)
    d["A_sets"], d["B_sets"] = d["Sets_P1"], d["Sets_P2"]
    d["winner"] = d["HomeWinner"].map({1:"A", 0:"B"})
    d["source_row_id"] = d["X"].astype(str)
    for i in range(1,6):
        d[f"set{i}_A"] = pd.to_numeric(d.get(f"P1_G{i}"), errors="coerce")
        d[f"set{i}_B"] = pd.to_numeric(d.get(f"P2_G{i}"), errors="coerce")
    valid_score = (((d.A_sets == 3) & d.B_sets.between(0,2)) |
                   ((d.B_sets == 3) & d.A_sets.between(0,2)))
    complete = d.datetime.notna() & d.winner.notna() & valid_score & (d.player_A != d.player_B)
    rejected = int((~complete).sum())
    d = d.loc[complete].copy()
    identity = ["datetime","player_A_id","player_B_id","A_sets","B_sets"] + [f"set{i}_{side}" for i in range(1,6) for side in ("A","B")]
    duplicate_mask = d.duplicated(identity, keep="first")
    duplicates = int(duplicate_mask.sum())
    d = d.loc[~duplicate_mask].copy()
    # Verify set totals against the point-level winners.
    a_wins = sum((d[f"set{i}_A"] > d[f"set{i}_B"]).fillna(False).astype(int) for i in range(1,6))
    b_wins = sum((d[f"set{i}_B"] > d[f"set{i}_A"]).fillna(False).astype(int) for i in range(1,6))
    consistent = (a_wins == d.A_sets) & (b_wins == d.B_sets)
    point_mismatches = int((~consistent).sum())
    d = d.loc[consistent].copy()
    d = d.sort_values(["datetime","source_row_id"]).reset_index(drop=True)
    d["match_id"] = "score-2022-" + d.source_row_id
    cols = ["match_id","datetime","player_A_id","player_A","player_B_id","player_B","A_sets","B_sets","winner"] + [f"set{i}_{side}" for i in range(1,6) for side in ("A","B")]
    audit = {"source":PUBLIC_DATA_URL,"input_rows":input_rows,"rejected_incomplete_or_invalid":rejected,
             "duplicates_removed":duplicates,"point_score_mismatches_removed":point_mismatches,
             "clean_rows":int(len(d)),"players":int(pd.unique(pd.concat([d.player_A,d.player_B])).size),
             "start":str(d.datetime.min()),"end":str(d.datetime.max()),
             "identity_limitation":"Archive has abbreviated names but no official Setka player IDs; normalized names are fallback identifiers."}
    return d[cols], audit

def load_public_archive(url: str = PUBLIC_DATA_URL):
    return clean_public_archive(pd.read_csv(url))
