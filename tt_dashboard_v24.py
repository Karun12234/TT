# tt_dashboard.py  (FULL PLUG-AND-PLAY with Date Selector pulling ALL games on the selected date)
# ==============================================================================================
# Streamlit Dashboard for TT Elite (ID-based + per-player metrics)
# + BetsAPI Odds Summary (Totals, Match ML, 1st Set ML)
# ==============================================================================================

import os
import io
import time
import json
import re
import requests
import numpy as np
import pandas as pd
from pandas import json_normalize
from datetime import datetime, date
import streamlit as st
from typing import Optional


# ---------- Global % formatter (use everywhere) ----------
def fmt_pct(value, *, hi2=80, lo2=20, hi1=65, lo1=35, dash="—"):
    if value is None:
        return dash
    try:
        v = float(value)
    except Exception:
        return str(value)
    if v >= hi2 or v <= lo2:
        color = "#2ecc71"
    elif v >= hi1 or v <= lo1:
        color = "#9b59b6"
    else:
        color = None
    txt = f"{v:.1f}%"
    return f"<span style='color:{color};font-weight:bold'>{txt}</span>" if color else f"**{txt}**"

# =========================
# DEFAULT CONFIG (editable in UI)
# =========================
DEFAULT_TOKEN = ""  # <- your token default
DEFAULT_SPORT_ID = 92
DEFAULT_TZ = "America/Toronto"

LEAGUES = [
    {"label": "ALL", "league_id": None, "name": "ALL"},
    {"label": "TT Elite Series", "league_id": 29128, "name": "TT Elite Series"},
    {"label": "CZECH", "league_id": 22742, "name": "CZECH"},
    {"label": "TT CUP", "league_id": 29097, "name": "TT CUP"},
    #{"label": "SETKA CUP", "league_id": 22307, "name": "SETKA CUP"},
]

DEFAULT_HISTORY_QTY = 20
DEFAULT_OVER_TOTAL_THRESHOLD = 75.5
DEFAULT_SET1_OVER_THRESHOLD = 18.5
DEFAULT_LIMIT_EVENTS = None
DEFAULT_RATE_LIMIT_SECONDS = 0.25

# =========================
# Helpers (HTTP / small utils)
# =========================
def fetch_json(url: str) -> dict:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def normalize_score_str(s):
    if pd.isna(s) or "-" not in s:
        return s
    a, b = s.split("-", 1)
    a, b = int(a), int(b)
    return f"{max(a,b)}-{min(a,b)}"

def pct(numer, denom):
    return float(np.round((numer / denom) * 100, 1)) if denom else None

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def pid_to_name_map(df_h2h):
    rows = []
    for side in ("home", "away"):
        if f"{side}_id" in df_h2h.columns and f"{side}_name" in df_h2h.columns:
            part = df_h2h[[f"{side}_id", f"{side}_name"]].copy()
            part.columns = ["pid", "pname"]
            rows.append(part)
    if not rows:
        return {}
    z = pd.concat(rows, ignore_index=True).dropna()
    if z.empty:
        return {}
    return (
        z.groupby("pid")["pname"]
         .agg(lambda s: s.value_counts().idxmax())
         .to_dict()
    )

# --- Fixed thresholds for Selected Summary sheet (independent of Hot Sheets) ---
SELECTED_SHEET_HI = 75.0
SELECTED_SHEET_LO = 25.0

def _build_selected_sheet_rows_for_event(
    eid: int,
    home_nm: str,
    away_nm: str,
    event_time,                 # naive local dt (what you pass to compute_event_package)
    timezone_local: str,
    sport_id: int,
    token: str,
    history_qty: int,
    rate_limit_seconds: float,
    up_or_end: str,             # "upcoming" | "inplay" | "ended"
):
    """
    Returns exploded rows for ONE event (independent of Hot Sheets).
    Fixed rule:
      - If metric >= 75% ⇒ recommend Over / S1 winner etc.
      - If metric <= 25% ⇒ recommend Under (for Over/Under-type metrics)
      - Else → "no recommendation"
    American odds only.
    """
    rows = []

    # Overview once (reuse for outcomes + PIDs)
    ov = get_event_overview(int(eid), sport_id, token)
    p1_pid = ov.get("home_id")
    p2_pid = ov.get("away_id")

    (df_h2h, agg_summary, _pa, _piv, df_sets, _totals, met) = compute_event_package(
        int(eid),
        timezone_local, sport_id, token,
        history_qty,
        float(st.session_state.get("over_total_threshold", DEFAULT_OVER_TOTAL_THRESHOLD)),
        float(st.session_state.get("set1_over_threshold", DEFAULT_SET1_OVER_THRESHOLD)),
        rate_limit_seconds,
        event_cutoff_dt=event_time,
    )

    # H2H scorelines %
    def _score_pct_and_total(agg_df, key):
        if agg_df is None or agg_df.empty:
            return None, 0, 0
        tot = int(agg_df.loc[agg_df["Score"]=="Total","Count"].iloc[0]) \
            if "Total" in agg_df["Score"].values else int(agg_df["Count"].sum())
        rowk = agg_df[agg_df["Score"]==key]
        if rowk.empty:
            return 0.0, 0, tot
        cnt = int(rowk["Count"].iloc[0])
        pct = float(rowk["Percent"].iloc[0]) if "Percent" in rowk.columns else (100.0*cnt/max(1,tot))
        return pct, cnt, tot

    p30,_,totGames = _score_pct_and_total(agg_summary, "3-0")
    p31,_,_        = _score_pct_and_total(agg_summary, "3-1")
    p32,_,_        = _score_pct_and_total(agg_summary, "3-2")

    # Odds (American only)
    try:
        odds_summary = get_odds_summary_cached(int(eid), sport_id, token)
        stage = "end" if up_or_end == "inplay" else "start"
        totals_odds, match_ml, set1_ml = parse_event_odds_summary(
            odds_summary, home_nm, away_nm, preferred_stage=stage
        )
    except Exception:
        totals_odds, match_ml, set1_ml = {}, {}, {}

    base = {
        "event_id": str(eid),
        "match": f"{home_nm or '?'} vs {away_nm or '?'}",
        "time": event_time,
        "H2H_n": totGames,
        "P1_Name": home_nm or "",
        "P2_Name": away_nm or "",
        "P1_PID": p1_pid or "",
        "P2_PID": p2_pid or "",
        # American odds + book
        "OU_Line": totals_odds.get("line"),
        "Over_Amer": totals_odds.get("over_amer"),
        "Under_Amer": totals_odds.get("under_amer"),
        "P1_ML_Amer": match_ml.get("home_amer"),
        "P2_ML_Amer": match_ml.get("away_amer"),
        "S1_P1_ML_Amer": set1_ml.get("home_amer"),
        "S1_P2_ML_Amer": set1_ml.get("away_amer"),
        "Odds_Book": totals_odds.get("book") or match_ml.get("book") or set1_ml.get("book"),
    }

    # Metrics (same full set you use in Hot Sheets)
    base.update({
        "s1_winner_wins_match_pct":  met.get("s1_winner_wins_match_pct"),
        "s1_winner_opponent_s2_pct": met.get("s1_winner_opponent_s2_pct"),
        "s1_winner_s2_win_pct":      met.get("s1_winner_s2_win_pct"),
        "pct_3_0": p30, "pct_3_1": p31, "pct_3_2": p32,
        "set1_over_18_5_pct": met.get("set1_over_18_5_pct"),
        "set1_under_18_5_pct": met.get("set1_under_18_5_pct"),
        "full_over_pct":       met.get("over_75_5_prob_pct"),
    })

    # Per-set Over% (1–5)
    per_set = met.get("per_set_over_df", pd.DataFrame())
    if isinstance(per_set, pd.DataFrame) and not per_set.empty:
        for s in range(1,6):
            slot = per_set[per_set["set_no"]==s]
            base[f"set{s}_over_pct"] = (float(slot["over_pct"].iloc[0])
                                        if not slot.empty and pd.notna(slot["over_pct"].iloc[0]) else None)

    # P1/P2 overall win% (PID-based) for the selected sheet
    def _overall_win_pct_by_pid(df_local, pid):
        if df_local.empty or not pid:
            return None
        pool = df_local[(df_local["home_id"] == pid) | (df_local["away_id"] == pid)]
        if pool.empty:
            return None
        wins = int((pool["match_winner_pid"] == pid).sum())
        den  = int(len(pool))
        return (wins/den*100.0) if den else None

    base["p1_overall_win_pct"] = _overall_win_pct_by_pid(df_h2h, p1_pid)
    base["p2_overall_win_pct"] = _overall_win_pct_by_pid(df_h2h, p2_pid)

    # Fixed recommend logic (≥75 or ≤25 only)
    METRIC_ORDER = (
        [("Full Over %", "full_over_pct")] +
        [(f"Set {i} Over %", f"set{i}_over_pct") for i in range(1,6)] +
        [("Set 1 Over %", "set1_over_18_5_pct"), ("Set 1 Under %", "set1_under_18_5_pct")] +
        [("S1 → Match Win %", "s1_winner_wins_match_pct"),
         ("S1 Opp Wins S2 %", "s1_winner_opponent_s2_pct"),
         ("S1 Wins S2 %", "s1_winner_s2_win_pct")] +
        [("P1 Overall Win %", "p1_overall_win_pct"),
         ("P2 Overall Win %", "p2_overall_win_pct")] +
        [("Exact 3-0 %", "pct_3_0"), ("Exact 3-1 %", "pct_3_1"), ("Exact 3-2 %", "pct_3_2")]
    )

    def decide_rec(metric_key: str, pct_val: float):
        if pct_val is None or pd.isna(pct_val):
            return None, None
        x = float(pct_val)

        # Over/Under family
        if metric_key in ("full_over_pct", "set1_over_18_5_pct") or metric_key.endswith("_over_pct"):
            th = float(st.session_state.get("set1_over_threshold", DEFAULT_SET1_OVER_THRESHOLD)) \
                 if metric_key != "full_over_pct" else float(st.session_state.get("over_total_threshold", DEFAULT_OVER_TOTAL_THRESHOLD))
            if x >= SELECTED_SHEET_HI: return (f"Play **Over {th:.1f}**", "SET1_OVER" if metric_key!="full_over_pct" else "FULL_OVER")
            if x <= SELECTED_SHEET_LO: return (f"Play **Under {th:.1f}**", "SET1_UNDER" if metric_key!="full_over_pct" else "FULL_UNDER")
            return None, None

        if metric_key == "set1_under_18_5_pct":
            th = float(st.session_state.get("set1_over_threshold", DEFAULT_SET1_OVER_THRESHOLD))
            if x >= SELECTED_SHEET_HI: return (f"Play **Under {th:.1f}**", "SET1_UNDER")
            return None, None

        # S1 dependent signals (no under/over)
        if metric_key == "s1_winner_wins_match_pct" and x >= SELECTED_SHEET_HI:
            return ("Back **S1 winner to win the match**", "S1_WINS_MATCH")
        if metric_key == "s1_winner_opponent_s2_pct" and x >= SELECTED_SHEET_HI:
            return ("Back **S1 opponent to win Set 2**", "S1_OPP_WINS_S2")
        if metric_key == "s1_winner_s2_win_pct" and x >= SELECTED_SHEET_HI:
            return ("Back **S1 winner to win Set 2**", "S1_WINS_S2")

        # P1/P2 overall win (no gating)
        if metric_key == "p1_overall_win_pct" and x >= SELECTED_SHEET_HI:
            return ("Back **Player 1 to win**", "P1_WIN")
        if metric_key == "p2_overall_win_pct" and x >= SELECTED_SHEET_HI:
            return ("Back **Player 2 to win**", "P2_WIN")

        # Exact score props: only if very strong
        if metric_key in ("pct_3_0","pct_3_1","pct_3_2") and x >= SELECTED_SHEET_HI:
            final = metric_key.replace("pct_","").replace("_","-")
            return (f"Play **Exact {final}**", f"EXACT_{final}")

        return None, None

    for lbl, key in METRIC_ORDER:
        v = base.get(key)
        rec, desired = decide_rec(key, v)

        # Units (ladder you already have)
        u = units_for_prob(v)

        # Outcome / Pending
        if up_or_end == "ended" and desired:
            outcome, fval = evaluate_outcome_and_value(desired, ov)
            rec_result = outcome if outcome in ("Win","Loss") else "N/A"
            final_value = fval
        else:
            rec_result = "Pending" if up_or_end != "ended" else ("N/A" if not desired else "N/A")
            final_value = None

        rows.append({
            **base,
            "Rec_Metric":  lbl,
            "Rec_Text":    rec or "_no recommendation_",
            "Rec_Key":     desired,
            "Rec_ProbPct": v,
            "Rec_Units":   u,
            "Final_Value": final_value,
            "Rec_Result":  rec_result,
        })

    return rows



def _build_hot_like_rows_for_event(
    eid: int,
    home_nm: str,
    away_nm: str,
    event_time,                 # naive local dt you already pass to compute_event_package
    timezone_local: str,
    sport_id: int,
    token: str,
    history_qty: int,
    rate_limit_seconds: float,
    up_or_end: str,             # "upcoming" | "inplay" | "ended"
    selected_metric_labels: list[str],
    label2key: dict[str,str],
    only_actionable: bool,
    show_book: bool,
    hi2_val: float,
    lo2_val: float,
    hi1_val: float,
    lo1_val: float,
):
    """
    Returns: list[dict] exploded rows like Hot Sheets for ONE event.
    Uses *current* sidebar thresholds & selected metrics.
    American odds only.
    """
    rows = []

    # pull PIDs + overview ONCE (reuse for outcome calc)
    ov_for_pids = get_event_overview(int(eid), sport_id, token)
    p1_pid = ov_for_pids.get("home_id")  # HOME
    p2_pid = ov_for_pids.get("away_id")  # AWAY

    (df_h2h_ev, agg_summary_ev, _pa, _piv, df_sets_ev, _totals_ev, met) = compute_event_package(
        int(eid),
        timezone_local, sport_id, token,
        history_qty,
        float(st.session_state.get("over_total_threshold", DEFAULT_OVER_TOTAL_THRESHOLD)),
        float(st.session_state.get("set1_over_threshold", DEFAULT_SET1_OVER_THRESHOLD)),
        rate_limit_seconds,
        event_cutoff_dt=event_time,
    )

    # H2H scorelines %
    def _score_pct_and_total(agg_df, key):
        if agg_df is None or agg_df.empty:
            return None, 0, 0
        tot = int(agg_df.loc[agg_df["Score"]=="Total","Count"].iloc[0]) \
            if "Total" in agg_df["Score"].values else int(agg_df["Count"].sum())
        rowk = agg_df[agg_df["Score"]==key]
        if rowk.empty:
            return 0.0, 0, tot
        cnt = int(rowk["Count"].iloc[0])
        pct = float(rowk["Percent"].iloc[0]) if "Percent" in rowk.columns else (100.0*cnt/max(1,tot))
        return pct, cnt, tot

    p30,_,totGames = _score_pct_and_total(agg_summary_ev, "3-0")
    p31,_,_        = _score_pct_and_total(agg_summary_ev, "3-1")
    p32,_,_        = _score_pct_and_total(agg_summary_ev, "3-2")

    # overall win % by PID
    def _overall_win_pct_by_pid(df_h2h_local: pd.DataFrame, pid: Optional[str]):
        if df_h2h_local.empty or not pid:
            return None
        pool = df_h2h_local[(df_h2h_local["home_id"] == pid) | (df_h2h_local["away_id"] == pid)]
        if pool.empty:
            return None
        wins = int((pool["match_winner_pid"] == pid).sum())
        den  = int(len(pool))
        return (wins/den*100.0) if den else None

    p1_overall = _overall_win_pct_by_pid(df_h2h_ev, p1_pid)
    p2_overall = _overall_win_pct_by_pid(df_h2h_ev, p2_pid)

    # leader for P1/P2 gating (like Hot Sheets)
    leader_side = None
    try:
        if p1_overall is not None and p2_overall is not None:
            leader_side = "p1" if float(p1_overall) >= float(p2_overall) else "p2"
        elif p1_overall is not None:
            leader_side = "p1"
        elif p2_overall is not None:
            leader_side = "p2"
    except Exception:
        leader_side = None

    # odds (American only)
    try:
        odds_summary = get_odds_summary_cached(int(eid), sport_id, token)
        stage = "end" if up_or_end == "inplay" else "start"
        totals_odds, match_ml, set1_ml = parse_event_odds_summary(
            odds_summary, home_nm, away_nm, preferred_stage=stage
        )
    except Exception:
        totals_odds, match_ml, set1_ml = {}, {}, {}

    base = {
        "event_id": str(eid),
        "match": f"{home_nm or '?'} vs {away_nm or '?'}",
        "time": event_time,
        "H2H_n": totGames,
        "P1_Name": home_nm or "",
        "P2_Name": away_nm or "",
        "P1_PID": p1_pid or "",
        "P2_PID": p2_pid or "",
        # American only + book
        "OU_Line": totals_odds.get("line"),
        "Over_Amer": totals_odds.get("over_amer"),
        "Under_Amer": totals_odds.get("under_amer"),
        "P1_ML_Amer": match_ml.get("home_amer"),
        "P2_ML_Amer": match_ml.get("away_amer"),
        "S1_P1_ML_Amer": set1_ml.get("home_amer"),
        "S1_P2_ML_Amer": set1_ml.get("away_amer"),
    }
    if show_book:
        base["Odds_Book"] = totals_odds.get("book") or match_ml.get("book") or set1_ml.get("book")

    # metric values (like Hot Sheets)
    base.update({
        "s1_winner_wins_match_pct":  met.get("s1_winner_wins_match_pct"),
        "s1_winner_opponent_s2_pct": met.get("s1_winner_opponent_s2_pct"),
        "s1_winner_s2_win_pct":      met.get("s1_winner_s2_win_pct"),
        "pct_3_0": p30, "pct_3_1": p31, "pct_3_2": p32,
        "set1_over_18_5_pct": met.get("set1_over_18_5_pct"),
        "set1_under_18_5_pct": met.get("set1_under_18_5_pct"),
        "full_over_pct":       met.get("over_75_5_prob_pct"),
        "p1_overall_win_pct":  p1_overall,
        "p2_overall_win_pct":  p2_overall,
    })

    # per-set Over% table (1–5)
    per_set = met.get("per_set_over_df", pd.DataFrame())
    if isinstance(per_set, pd.DataFrame) and not per_set.empty:
        for s in range(1,6):
            slot = per_set[per_set["set_no"]==s]
            base[f"set{s}_over_pct"] = (float(slot["over_pct"].iloc[0])
                                        if not slot.empty and pd.notna(slot["over_pct"].iloc[0]) else None)

    # same recommend + outcome logic as Hot Sheets
    def band_of(x):
        if x is None or pd.isna(x): return None
        x = float(x)
        if x >= hi2_val or x <= lo2_val: return "HIGH"
        if (hi1_val <= x < hi2_val) or (lo2_val < x <= lo1_val): return "MHIGH"
        return None

    def recommend_direction(metric_key: str, pct_val: float):
        b = band_of(pct_val)
        if not b: return None, None
        if metric_key == "full_over_pct":
            th = float(st.session_state.get("over_total_threshold", DEFAULT_OVER_TOTAL_THRESHOLD))
            return (f"Play Full **Over {th:.1f}**", "FULL_OVER") if pct_val >= hi1_val else (f"Play Full **Under {th:.1f}**", "FULL_UNDER")
        if metric_key == "set1_over_18_5_pct":
            th = float(st.session_state.get("set1_over_threshold", DEFAULT_SET1_OVER_THRESHOLD))
            return (f"Play **Set 1 Over {th:.1f}**", "SET1_OVER") if pct_val >= hi1_val else (f"Play **Set 1 Under {th:.1f}**", "SET1_UNDER")
        if metric_key == "set1_under_18_5_pct":
            th = float(st.session_state.get("set1_over_threshold", DEFAULT_SET1_OVER_THRESHOLD))
            return (f"Play **Set 1 Under {th:.1f}**", "SET1_UNDER")
        if metric_key.startswith("set") and metric_key.endswith("_over_pct"):
            n = int(metric_key[3])
            th = float(st.session_state.get("set1_over_threshold", DEFAULT_SET1_OVER_THRESHOLD))
            return (f"Play **Set {n} Over {th:.1f}**", f"SET{n}_OVER") if pct_val >= hi1_val else (f"Play **Set {n} Under {th:.1f}**", f"SET{n}_UNDER")
        if metric_key in ("pct_3_0","pct_3_1","pct_3_2"):
            if pct_val >= hi1_val:
                final = metric_key.replace("pct_","").replace("_","-")
                return (f"Play **Exact {final}**", f"EXACT_{final}")
            return None, None
        if metric_key == "s1_winner_wins_match_pct":
            return ("Back **S1 winner to win the match**", "S1_WINS_MATCH") if pct_val >= hi1_val else (None, None)
        if metric_key == "s1_winner_opponent_s2_pct":
            return ("Back **S1 opponent to win Set 2**", "S1_OPP_WINS_S2") if pct_val >= hi1_val else (None, None)
        if metric_key == "s1_winner_s2_win_pct":
            return ("Back **S1 winner to win Set 2**", "S1_WINS_S2") if pct_val >= hi1_val else (None, None)
        if metric_key == "p1_overall_win_pct":
            return ("Back **Player 1 to win**", "P1_WIN")
        if metric_key == "p2_overall_win_pct":
            return ("Back **Player 2 to win**", "P2_WIN")
        return None, None

    # explode recommendations (respect only_actionable & P1/P2 leader gating)
    for lbl in selected_metric_labels:
        k = label2key[lbl]
        v = base.get(k)

        if k == "p1_overall_win_pct" and leader_side == "p2":
            rec, desired = (None, None)
        elif k == "p2_overall_win_pct" and leader_side == "p1":
            rec, desired = (None, None)
        else:
            rec, desired = recommend_direction(k, v)

        if not rec and only_actionable:
            continue

        # Units from prob
        u = units_for_prob(v)
        # Result / Pending
        if up_or_end == "ended":
            outcome, fval = evaluate_outcome_and_value(desired, ov_for_pids) if desired else ("N/A", None)
            rec_result = outcome if outcome in ("Win","Loss") else "N/A"
            final_value = fval
        else:
            rec_result = "Pending"
            final_value = None

        row = dict(base)
        row.update({
            "Rec_Metric":  lbl,
            "Rec_Text":    rec or "_no rec_",
            "Rec_Key":     desired,
            "Rec_ProbPct": v,
            "Rec_Units":   u,
            "Final_Value": final_value,
            "Rec_Result":  rec_result,
        })
        if show_book and "Odds_Book" not in row:
            row["Odds_Book"] = None
        rows.append(row)

    # If nothing and not only_actionable, add a single no-rec row
    if not rows and not only_actionable:
        rows.append(dict(base, Rec_Metric="—", Rec_Text="_no rec_", Rec_Key=None, Rec_ProbPct=None, Rec_Units=None, Final_Value=None, Rec_Result="Pending" if up_or_end!="ended" else "N/A"))

    return rows


# -----------------------------
# 🔧 Units ladder + styling utils
# -----------------------------
def units_for_prob(pct: Optional[float]) -> Optional[float]:
    """
    Applies the two ladders you specified.
    - High side: 75,80,85,90,95,100 => 1,1.5,2,2.5,3,5
    - Low side:  25,20,15,10,5,0    => 1,1.5,2,2.5,3,5
    Returns None if pct is None or in the middle zone (no ladder).
    """
    if pct is None or pd.isna(pct):
        return None
    x = float(pct)

    # high side
    if x >= 100: return 5.0
    if x >= 95:  return 3.0
    if x >= 90:  return 2.5
    if x >= 85:  return 2.0
    if x >= 80:  return 1.5
    if x >= 75:  return 1.0

    # low side (mirror)
    if x <= 0:   return 5.0
    if x <= 5:   return 3.0
    if x <= 10:  return 2.5
    if x <= 15:  return 2.0
    if x <= 20:  return 1.5
    if x <= 25:  return 1.0

    return None


def color_style_for_result(val: str) -> str:
    """Green for Win, Red for Loss; empty otherwise."""
    if isinstance(val, str):
        v = val.strip().lower()
        if v == "win":
            return "background-color:#e6ffe6; color:#077307; font-weight:bold;"
        if v == "loss":
            return "background-color:#ffe6e6; color:#8a0000; font-weight:bold;"
    return ""



# =========================
# Event overview (sets/scoreboard)
# =========================
def get_event_overview(event_id: int, sport_id: int, token: str):
    
    url = (
        f"https://api.b365api.com/v3/event/view?"
        f"sport_id={sport_id}&token={token}&event_id={event_id}"
    )
    j = fetch_json(url)
    res = j.get("results", [])
    if not isinstance(res, list) or not res:
        return {}
    ev = res[0]

    home_name = ev.get("home", {}).get("name")
    away_name = ev.get("away", {}).get("name")
    # ✨ NEW: grab PIDs from the event payload
    home_id   = str(ev.get("home", {}).get("id")) if ev.get("home", {}).get("id") is not None else None
    away_id   = str(ev.get("away", {}).get("id")) if ev.get("away", {}).get("id") is not None else None

    time_status = str(ev.get("time_status", ""))
    status_map = {"0": "Not Started", "1": "Live", "3": "Finished"}
    status_text = status_map.get(time_status, ev.get("time_status", "Unknown"))
    ss_sets = ev.get("ss")

    sets = []
    scores = ev.get("scores", {})
    if isinstance(scores, dict):
        for k, v in sorted(scores.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 999):
            try:
                set_no = int(k)
            except Exception:
                continue
            sh = v.get("home") if isinstance(v, dict) else None
            sa = v.get("away") if isinstance(v, dict) else None
            if sh is None and isinstance(v, str) and "-" in v:
                try:
                    a, b = v.split("-", 1)
                    sh, sa = int(a), int(b)
                except Exception:
                    pass
            sets.append({"set_no": set_no, "home": sh, "away": sa})
    elif isinstance(scores, list):
        for v in sorted(scores, key=lambda x: int(x.get("number", 999))):
            try:
                set_no = int(v.get("number"))
            except Exception:
                continue
            sh = v.get("home")
            sa = v.get("away")
            sets.append({"set_no": set_no, "home": sh, "away": sa})

    total_points = 0
    for s in sets:
        if s["home"] is not None and s["away"] is not None:
            s["total"] = int(s["home"]) + int(s["away"])
            total_points += s["total"]
        else:
            s["total"] = None

    return {
        "home_id": home_id,          # ✨ NEW
        "away_id": away_id,          # ✨ NEW
        "home_name": home_name,
        "away_name": away_name,
        "status": status_text,
        "time_status": time_status,
        "sets_score": ss_sets,
        "sets_table": sets,
        "total_points": total_points,
    }


# ---------- SHEET-NAME HELPER (≤31 chars + unique) ----------
def make_sheet_name(base: str, suffix: str, used: set[str]) -> str:
    base = (base or "sheet").replace("/", "_").replace("\\", "_")
    # leave room for '_' + suffix
    max_base = max(1, 31 - (len(suffix) + 1))
    name = f"{base[:max_base]}_{suffix}"
    if name not in used:
        used.add(name)
        return name
    # uniquify with a counter if collision
    i = 2
    while True:
        extra = len(f" ({i})")
        max_base2 = max(1, 31 - (len(suffix) + 1 + extra))
        name_try = f"{base[:max_base2]}_{suffix} ({i})"
        if name_try not in used:
            used.add(name_try)
            return name_try
        i += 1


# ---------- OUTCOME EVALUATOR (uses sidebar thresholds) ----------
def evaluate_outcome_and_value(desired_key: str, ov: dict):
    """
    Returns (OutcomeStr, FinalValue)

    OutcomeStr: "Win" | "Loss" | "N/A"
    FinalValue:
    - FULL_*          -> total match points (int)
    - SETn_*          -> set n total points (int)
    - EXACT_*         -> final sets score string like "3-1"
    - S1_* and P*_WIN -> short descriptor "S1W:HOME, S2W:AWAY" or final ss for ML
    """
    if not ov:
        return "N/A", None

    ss = ov.get("sets_score")
    sets = ov.get("sets_table", []) or []
    total_points = 0
    has_total = False
    for s in sets:
        if s.get("home") is not None and s.get("away") is not None:
            has_total = True
            total_points += int(s["home"]) + int(s["away"])

    def set_total(n):
        for s in sets:
            if s.get("set_no") == n and s.get("home") is not None and s.get("away") is not None:
                return int(s["home"]) + int(s["away"])
        return None

    def s1_winner_id():
        s1 = next((s for s in sets if s.get("set_no") == 1), None)
        if not s1 or s1.get("home") is None or s1.get("away") is None:
            return None
        if s1["home"] > s1["away"]:
            return "HOME"
        if s1["away"] > s1["home"]:
            return "AWAY"
        return None

    def s2_winner_id():
        s2 = next((s for s in sets if s.get("set_no") == 2), None)
        if not s2 or s2.get("home") is None or s2.get("away") is None:
            return None
        if s2["home"] > s2["away"]:
            return "HOME"
        if s2["away"] > s2["home"]:
            return "AWAY"
        return None

    # FULL totals
    if desired_key in ("FULL_OVER", "FULL_UNDER"):
        if not has_total:
            return "N/A", None
        over = total_points > float(st.session_state.get("over_total_threshold", DEFAULT_OVER_TOTAL_THRESHOLD))
        outcome = "Win" if (over and desired_key == "FULL_OVER") or ((not over) and desired_key == "FULL_UNDER") else "Loss"
        return outcome, total_points

    # SETn totals
    if desired_key.startswith("SET") and desired_key.endswith(("OVER", "UNDER")):
        try:
            num = int(desired_key[3])
        except Exception:
            return "N/A", None
        tot = set_total(num)
        if tot is None:
            return "N/A", None
        over = tot > float(st.session_state.get("set1_over_threshold", DEFAULT_SET1_OVER_THRESHOLD))
        want_over = desired_key.endswith("OVER")
        return ("Win" if over == want_over else "Loss"), tot

    # EXACT scoreline
    if desired_key.startswith("EXACT_"):
        if not ss:
            return "N/A", None
        want = desired_key.replace("EXACT_", "")
        return ("Win" if ss == want else "Loss"), ss

    # S1 dependent markets
    if desired_key in ("S1_WINS_MATCH", "S1_OPP_WINS_S2", "S1_WINS_S2"):
        s1w = s1_winner_id()
        if s1w is None:
            return "N/A", None
        if desired_key == "S1_WINS_MATCH":
            if not ss:
                return "N/A", None
            home_won = int(ss.split("-")[0]) > int(ss.split("-")[1])
            match_winner = "HOME" if home_won else "AWAY"
            return ("Win" if match_winner == s1w else "Loss"), f"S1W:{s1w}, MatchW:{match_winner}"
        if desired_key == "S1_OPP_WINS_S2":
            s2w = s2_winner_id()
            if s2w is None:
                return "N/A", None
            opp = "AWAY" if s1w == "HOME" else "HOME"
            return ("Win" if s2w == opp else "Loss"), f"S1W:{s1w}, S2W:{s2w}"
        if desired_key == "S1_WINS_S2":
            s2w = s2_winner_id()
            if s2w is None:
                return "N/A", None
            return ("Win" if s2w == s1w else "Loss"), f"S1W:{s1w}, S2W:{s2w}"

    # Straight ML (P1/P2) — compare to HOME/AWAY
    if desired_key in ("P1_WIN", "P2_WIN"):
        if not ss:
            return "N/A", None
        home_won = int(ss.split("-")[0]) > int(ss.split("-")[1])
        want_home = (desired_key == "P1_WIN")  # P1 = HOME
        return ("Win" if home_won == want_home else "Loss"), ss

    return "N/A", None


def mainview_recommend_over_only(metric_key: str, pct_val):
    """
    Over/Under decision (one line per metric), independent of Hot Sheets.

      - >=75.0   → Play Over   (ladder units)
      - 50.0–74.9→ Lean Over   (0.5u)
      - 25.1–49.9→ Lean Under  (0.5u)
      - <=25.0   → Play Under  (ladder units)
      - ==50.0   → No recommendation

    Returns (rec_text, desired_key, units, is_lean)
      desired_key ∈ {"FULL_OVER","FULL_UNDER","SET1_OVER","SET1_UNDER"}
      so evaluate_outcome_and_value can grade it.
    """
    # guard
    try:
        x = float(pct_val)
    except Exception:
        return None, None, None, False
    if pd.isna(x):
        return None, None, None, False

    def _mk(over: bool):
        if metric_key == "full_over_pct":
            line = float(st.session_state.get("over_total_threshold", DEFAULT_OVER_TOTAL_THRESHOLD))
            return (f"Play Full **{'Over' if over else 'Under'} {line:.1f}**",
                    "FULL_OVER" if over else "FULL_UNDER")
        if metric_key == "set1_over_18_5_pct":
            line = float(st.session_state.get("set1_over_threshold", DEFAULT_SET1_OVER_THRESHOLD))
            return (f"Play **Set 1 {'Over' if over else 'Under'} {line:.1f}**",
                    "SET1_OVER" if over else "SET1_UNDER")
        # future: map per-set keys like "set2_over_pct" -> "SET2_OVER/UNDER"
        return (None, None)

    def _ladder_units_sym(v):
        if v >= 100: return 5.0
        if v >= 95:  return 3.0
        if v >= 90:  return 2.5
        if v >= 85:  return 2.0
        if v >= 80:  return 1.5
        if v >= 75:  return 1.0
        if v <= 0:   return 5.0
        if v <= 5:   return 3.0
        if v <= 10:  return 2.5
        if v <= 15:  return 2.0
        if v <= 20:  return 1.5
        if v <= 25:  return 1.0
        return None

    # Strong zones (ladder)
    if x >= 75.0:
        txt, key = _mk(True)
        return txt, key, (_ladder_units_sym(x) or 1.0), False
    if x <= 25.0:
        txt, key = _mk(False)
        return txt, key, (_ladder_units_sym(x) or 1.0), False

    # Lean zones (0.5u)
    if 50.0 < x < 75.0:
        base, key = _mk(True)
        return (base.replace("Play", "Lean") if base else None), key, 0.5, True
    if 25.0 < x < 50.0:
        base, key = _mk(False)
        return (base.replace("Play", "Lean") if base else None), key, 0.5, True

    # exactly 50 → no rec
    return None, None, None, False



def _safe_float(x):
    try:
        if isinstance(x, str):
            x = x.strip()
        return float(x)
    except Exception:
        return None

def _dec_to_amer(dec):
    d = _safe_float(dec)
    if not d or d <= 1.0:
        return None
    if d >= 2.0:
        return int(round((d - 1.0) * 100))
    return int(round(-100 / (d - 1.0)))

def _slug(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

def _maybe_name_side(sel_name, home_name, away_name):
    s = _slug(sel_name)
    hn = _slug(home_name or "")
    an = _slug(away_name or "")
    if s in {"1", "home", "player 1", "p1", "team 1"}:
        return "HOME"
    if s in {"2", "away", "player 2", "p2", "team 2"}:
        return "AWAY"
    if hn and (hn in s or s in hn):
        return "HOME"
    if an and (an in s or s in an):
        return "AWAY"
    return None

def _extract_price(d):
    candidates = [
        d.get("odds"), d.get("odd"), d.get("price"), d.get("koef"), d.get("coef"),
        d.get("decimal"), d.get("o"), d.get("d"), d.get("value")
    ]
    dec = None
    for c in candidates:
        dec = _safe_float(c)
        if dec: break
    amer = _safe_float(d.get("american"))
    if amer is None and dec:
        amer = _dec_to_amer(dec)
    return dec, amer

def _extract_line(d):
    for k in ("handicap", "line", "hdp", "hcap", "total", "points"):
        v = d.get(k)
        if v is not None:
            f = _safe_float(v)
            return f if f is not None else v
    return None

def fetch_event_odds_summary(event_id: int, sport_id: int, token: str) -> dict:
    base = f"sport_id={sport_id}&token={token}&event_id={event_id}"
    for ver in ("v2", "v1"):
        url = f"https://api.b365api.com/{ver}/event/odds/summary?{base}"
        try:
            j = fetch_json(url)
            if j and j.get("results"):
                return {"version": ver, "payload": j}
        except Exception:
            pass
    return {"version": None, "payload": {}}

def parse_event_odds_summary(summary_json: dict, home_name: str, away_name: str, preferred_stage: str | None = None):
    """
    Robust odds parser:
      - Scans ALL bookmakers (not just Bet365)
      - Prefers stage order: preferred_stage (if given) -> 'start' -> 'kickoff' -> 'end'
      - Totals: explicitly looks for market code '92_3' first; then any market that has over_od/under_od(+handicap)
      - Match ML: prefers '92_1' first; then any market with home_od/away_od and no handicap (or handicap == 0)
      - 1st Set ML: same ML shape but keys that look like set1 (set1/1st/first_set/set_1/s1) OR any stage with those keys

    Returns (totals, match_ml, set1_ml) where each is {} if not found.
    """
    def _safe_float(x):
        try:
            return float(str(x).strip())
        except Exception:
            return None

    def _dec_to_amer(dec):
        d = _safe_float(dec)
        if not d or d <= 1.0:
            return None
        if d >= 2.0:
            return int(round((d - 1.0) * 100))
        return int(round(-100 / (d - 1.0)))

    def _looks_like_set1_key(k: str):
        k = str(k).lower()
        return any(x in k for x in ["set1", "1st", "first_set", "set_1", "s1"])

    def _mk_ml(d, book_name):
        hd = _safe_float(d.get("home_od"))
        ad = _safe_float(d.get("away_od"))
        return {
            "home_dec": hd,
            "home_amer": _dec_to_amer(hd) if hd else None,
            "away_dec": ad,
            "away_amer": _dec_to_amer(ad) if ad else None,
            "book": book_name,
            "updated_ts": d.get("add_time"),
        }

    def _mk_totals(d, book_name):
        over = _safe_float(d.get("over_od"))
        under = _safe_float(d.get("under_od"))
        line = d.get("handicap")
        linef = _safe_float(line)
        return {
            "line": (linef if linef is not None else line),
            "over_dec": over,
            "over_amer": _dec_to_amer(over) if over else None,
            "under_dec": under,
            "under_amer": _dec_to_amer(under) if under else None,
            "book": book_name,
            "updated_ts": d.get("add_time"),
        }

    # Stage preference: caller’s preferred first, then start -> kickoff -> end
    base_order = ["start", "kickoff", "end"]
    if preferred_stage in ("start", "end", "kickoff"):
        stage_order = [preferred_stage] + [s for s in base_order if s != preferred_stage]
    else:
        stage_order = base_order

    results = summary_json.get("payload", {}).get("results")
    if not isinstance(results, dict) or not results:
        return {}, {}, {}

    best_totals = {}
    best_match_ml = {}
    best_set1_ml = {}

    # iterate all bookmakers
    for book_name, book_block in results.items():
        if not isinstance(book_block, dict):
            continue
        odds = book_block.get("odds")
        if not isinstance(odds, dict):
            continue

        # Try stages in preferred order
        for stage in stage_order:
            markets = odds.get(stage)
            if not isinstance(markets, dict) or not markets:
                continue

            # --------- 1) Totals: prefer explicit '92_3' ----------
            # exact key
            d_92_3 = markets.get("92_3")
            if isinstance(d_92_3, dict) and "over_od" in d_92_3 and "under_od" in d_92_3:
                if not best_totals:
                    best_totals = _mk_totals(d_92_3, book_name)

            # fallback: any market that looks like totals (over/under + handicap)
            if not best_totals:
                for mkey, md in markets.items():
                    if not isinstance(md, dict):
                        continue
                    if ("over_od" in md and "under_od" in md) and ("handicap" in md or "line" in md or "total" in md):
                        best_totals = _mk_totals(md, book_name)
                        break

            # --------- 2) Match ML: prefer explicit '92_1' ----------
            d_92_1 = markets.get("92_1")
            if isinstance(d_92_1, dict) and "home_od" in d_92_1 and "away_od" in d_92_1 and "handicap" not in d_92_1:
                if not best_match_ml:
                    best_match_ml = _mk_ml(d_92_1, book_name)

            # fallback: any ML-like (home_od/away_od) with no handicap or 0 handicap
            if not best_match_ml:
                for mkey, md in markets.items():
                    if not isinstance(md, dict):
                        continue
                    if "home_od" in md and "away_od" in md:
                        hcap = str(md.get("handicap", "")).strip()
                        if (hcap == "" or hcap in ("0", "0.0", "+0", "+0.0")):
                            best_match_ml = _mk_ml(md, book_name)
                            break

            # --------- 3) 1st Set ML: ML-like in a set1 key ----------
            if not best_set1_ml:
                # common explicit key shapes vary; scan by key name
                for mkey, md in markets.items():
                    if not isinstance(md, dict):
                        continue
                    if "home_od" in md and "away_od" in md and "handicap" not in md and _looks_like_set1_key(mkey):
                        best_set1_ml = _mk_ml(md, book_name)
                        break

            # If all three found for this stage, no need to try lower-priority stages for this book
            if best_totals and best_match_ml and best_set1_ml:
                break

        # If we already found everything from some book/stage, we can keep going to let later books overwrite,
        # but typically the first match in preferred stage order is good enough. We’ll keep earliest finds.

    return best_totals, best_match_ml, best_set1_ml



# --- Auto-refresh helper ---
def do_live_autorefresh(enabled: bool, interval_ms: int = 15000, key: str = "live_refresh"):
    if not enabled:
        return
    if hasattr(st, "autorefresh"):
        st.autorefresh(interval=interval_ms, key=key); return
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=interval_ms, key=key); return
    except Exception:
        pass
    st.caption(f"Auto-refreshing every {int(interval_ms/1000)}s…")
    time.sleep(interval_ms / 1000.0)
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

# =========================
# Core builder (metrics/H2H)
# =========================
# (UNCHANGED from your version)
def build_h2h_report_for_event(
    event_id: int,
    timezone_local: str,
    sport_id: int,
    token: str,
    history_qty: int,
    over_total_threshold: float,
    set1_over_threshold: float,
    rate_limit_seconds: float,
    event_cutoff_dt: Optional[datetime] = None,   # <-- NEW
):
    # ... [IDENTICAL CONTENT AS YOURS] ...
    # (Keeping your full function body exactly as you posted)
    # (For brevity here, use your original definition without edits)
    # -----------------------
    # BEGIN: original content pasted 1:1
    # -----------------------
    url_hist = (
        f"https://api.b365api.com/v3/event/history?"
        f"sport_id={sport_id}&token={token}&event_id={event_id}&qty={history_qty}"
    )
    data = fetch_json(url_hist)
    h2h = data.get("results", {}).get("h2h", [])
    if not h2h:
        metrics = {
            "s1_winner_wins_match_pct": None,
            "s1_winner_opponent_s2_pct": None,
            "s1_winner_s2_win_pct": None,
            "over_75_5_prob_pct": None,
            "over_75_5_n": 0,
            "over_75_5_den": 0,
            "avg_total_points": None,      # <-- add
            "avg_total_vs_line": None,     # <-- add
            "set1_over_18_5_pct": None,
            "set1_under_18_5_pct": None,
            "players": [],
            "per_set_over_df": pd.DataFrame(),
        }
        return (
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
            pd.DataFrame(), pd.DataFrame(), metrics
        )


    df = json_normalize(h2h, sep=".")
    df["time"] = pd.to_datetime(df["time"].astype(int), unit="s", utc=True)
    df["time"] = df["time"].dt.tz_convert(timezone_local).dt.tz_localize(None)

    
    # >>> IMPORTANT: exclude all matches from the same local calendar day as the event <<<
    # df["time"] is already in local timezone (naive), so compare by date
    if event_cutoff_dt is not None:
        try:
            event_day = pd.Timestamp(event_cutoff_dt).date()
            df["__time_date"] = df["time"].dt.date
            df = df[df["__time_date"] < event_day].copy()
            df.drop(columns=["__time_date"], inplace=True, errors="ignore")
        except Exception:
            pass


    df["league"] = df["league.name"]
    df["home_id"] = df["home.id"].astype(str)
    df["away_id"] = df["away.id"].astype(str)
    df["home_name"] = df["home.name"]
    df["away_name"] = df["away.name"]
    df["score"] = df["ss"]

    # Ensure "ss" exists; if not, create it as all-None
    if "ss" not in df.columns:
        df["ss"] = None

    # Our working "score" column
    df["score"] = df["ss"]

    # Robust extract: always yields two columns even if no match
    # (when no match, both columns are NaN; but the columns still exist -> no KeyError)
    split_scores = (
        df["score"]
        .astype(str)
        .str.extract(r'^\s*(\d+)\s*-\s*(\d+)\s*$')
    )

    df["home_sets"] = pd.to_numeric(split_scores[0], errors="coerce")
    df["away_sets"] = pd.to_numeric(split_scores[1], errors="coerce")

    df["match_winner_pid"] = np.where(
        df["home_sets"] > df["away_sets"], df["home_id"],
        np.where(df["away_sets"] > df["home_sets"], df["away_id"], None)
    )
    df["winner_name"] = np.where(df["match_winner_pid"] == df["home_id"], df["home_name"],
                           np.where(df["match_winner_pid"] == df["away_id"], df["away_name"], None))
    df["loser_name"]  = np.where(df["match_winner_pid"] == df["home_id"], df["away_name"],
                           np.where(df["match_winner_pid"] == df["away_id"], df["home_name"], None))
    df.rename(columns={"id": "child_event_id"}, inplace=True)

    cols = [
        "child_event_id", "league",
        "home_id", "home_name",
        "away_id", "away_name",
        "score", "home_sets", "away_sets",
        "match_winner_pid", "winner_name", "loser_name", "time"
    ]
    df_h2h = df[cols].sort_values(by="time", ascending=False).reset_index(drop=True)
    df_h2h["child_event_id"] = df_h2h["child_event_id"].astype(str)

    df_h2h["score_norm"] = df_h2h["score"].apply(normalize_score_str)
    agg_counts = df_h2h["score_norm"].value_counts().sort_index()
    agg_summary = agg_counts.rename_axis("Score").reset_index(name="Count")
    total_games = int(agg_summary["Count"].sum())
    agg_summary["Percent"] = (agg_summary["Count"] / total_games * 100).round(1)
    agg_summary = pd.concat(
        [agg_summary, pd.DataFrame([{"Score": "Total", "Count": total_games, "Percent": 100.0}])],
        ignore_index=True
    )

    player_agg = (
        df_h2h.groupby(["winner_name", "score_norm"])
        .size()
        .reset_index(name="Count")
        .sort_values(["winner_name", "score_norm"])
    )
    player_totals = player_agg.groupby("winner_name")["Count"].transform("sum")
    player_agg["Percent"] = (player_agg["Count"] / player_totals * 100).round(1)
    player_agg_pivot = (
        player_agg.pivot(index="winner_name", columns="score_norm", values="Count")
        .fillna(0).astype(int)
    )
    player_agg_pivot["Total"] = player_agg_pivot.sum(axis=1)
    player_agg_pivot = player_agg_pivot.reindex(
        columns=[c for c in ["3-0","3-1","3-2"] if c in player_agg_pivot.columns] + ["Total"]
    )

    child_ids = df_h2h["child_event_id"].dropna().astype(str).tolist()
    set_rows = []
    for batch in chunks(child_ids, 10):
        ids_param = ",".join(batch)
        url_view = (
            f"https://api.b365api.com/v3/event/view?"
            f"sport_id={sport_id}&token={token}&event_id={ids_param}"
        )
        view_json = fetch_json(url_view)
        results = view_json.get("results", [])
        if not isinstance(results, list):
            continue
        for item in results:
            ev_id_raw = item.get("id")
            if ev_id_raw is None:
                continue
            ev_id = str(ev_id_raw)
            scores = item.get("scores", {})
            if isinstance(scores, dict):
                for k, v in scores.items():
                    try:
                        set_no = int(k)
                    except Exception:
                        continue
                    sh, sa = None, None
                    if isinstance(v, dict):
                        sh = v.get("home")
                        sa = v.get("away")
                    elif isinstance(v, str) and "-" in v:
                        try:
                            a, b = v.split("-", 1)
                            sh, sa = int(a), int(b)
                        except Exception:
                            pass
                    if sh is not None or sa is not None:
                        set_rows.append({
                            "child_event_id": ev_id, "set_no": set_no,
                            "set_home_points": None if sh is None else int(sh),
                            "set_away_points": None if sa is None else int(sa),
                        })
            elif isinstance(scores, list):
                for v in scores:
                    if not isinstance(v, dict):
                        continue
                    try:
                        set_no = int(v.get("number"))
                    except Exception:
                        continue
                    sh = v.get("home")
                    sa = v.get("away")
                    if sh is not None or sa is not None:
                        set_rows.append({
                            "child_event_id": ev_id, "set_no": set_no,
                            "set_home_points": None if sh is None else int(sh),
                            "set_away_points": None if sa is None else int(sa),
                        })

    df_sets = pd.DataFrame(set_rows)
    if not df_sets.empty:
        df_sets["child_event_id"] = df_sets["child_event_id"].astype(str)
        df_sets = df_sets.merge(
            df_h2h[[
                "child_event_id", "home_id", "home_name", "away_id", "away_name",
                "score", "time", "match_winner_pid"
            ]],
            on="child_event_id", how="left"
        ).sort_values(["child_event_id", "set_no"]).reset_index(drop=True)

    s1_winner_wins_match_pct = None
    s1_winner_opponent_s2_pct = None
    s1_winner_s2_win_pct = None
    over_75_5_prob_pct = None
    over_75_5_n = 0
    over_75_5_den = 0
    set1_over_18_5_pct = None
    set1_under_18_5_pct = None
    per_player_rows = []
    # Defaults so we don't hit UnboundLocalError when df_sets is empty
    avg_total_points = None
    avg_total_vs_line = None


    if not df_sets.empty:
        s1 = df_sets[df_sets["set_no"] == 1][[
            "child_event_id", "set_home_points", "set_away_points", "home_id", "away_id"
        ]].copy()
        s1["s1_winner_pid"] = np.where(
            s1["set_home_points"] > s1["set_away_points"], s1["home_id"],
            np.where(s1["set_away_points"] > s1["set_home_points"], s1["away_id"], None)
        )
        s1["s1_opponent_pid"] = np.where(
            s1["s1_winner_pid"] == s1["home_id"], s1["away_id"],
            np.where(s1["s1_winner_pid"] == s1["away_id"], s1["home_id"], None)
        )

        s2 = df_sets[df_sets["set_no"] == 2][[
            "child_event_id", "set_home_points", "set_away_points", "home_id", "away_id"
        ]].copy()
        s2["s2_winner_pid"] = np.where(
            s2["set_home_points"] > s2["set_away_points"], s2["home_id"],
            np.where(s2["set_away_points"] > s2["set_home_points"], s2["away_id"], None)
        )

        winners = df_h2h[["child_event_id", "match_winner_pid"]].copy()
        tmp = (
            s1.merge(winners, on="child_event_id", how="left")
              .merge(s2[["child_event_id", "s2_winner_pid"]], on="child_event_id", how="left")
        )

        mask_valid_match = tmp["s1_winner_pid"].notna() & tmp["match_winner_pid"].notna()
        s1_winner_wins_match_pct = pct(
            (mask_valid_match & (tmp["s1_winner_pid"] == tmp["match_winner_pid"])).sum(),
            mask_valid_match.sum()
        )
        mask_valid_s2 = tmp["s1_winner_pid"].notna() & tmp["s2_winner_pid"].notna()
        s1_winner_opponent_s2_pct = pct(
            (mask_valid_s2 & (tmp["s2_winner_pid"] == tmp["s1_opponent_pid"])).sum(),
            mask_valid_s2.sum()
        )
        s1_winner_s2_win_pct = pct(
            (mask_valid_s2 & (tmp["s2_winner_pid"] == tmp["s1_winner_pid"])).sum(),
            mask_valid_s2.sum()
        )

        pids = pd.unique(pd.concat([df_h2h["home_id"], df_h2h["away_id"]], ignore_index=True)).tolist()
        name_map = pid_to_name_map(df_h2h)
        for pid in pids:
            m_pid = tmp[tmp["s1_winner_pid"] == pid]
            den_match = len(m_pid[m_pid["match_winner_pid"].notna()])
            den_s2 = len(m_pid[m_pid["s2_winner_pid"].notna()])
            per_player_rows.append({
                "pid": pid,
                "name": name_map.get(pid, pid),
                "s1_win_match_pct": pct((m_pid["match_winner_pid"] == pid).sum(), den_match),
                "s1_opponent_s2_pct": pct((m_pid["s2_winner_pid"] == m_pid["s1_opponent_pid"]).sum(), den_s2),
                "s1_s2_win_pct": pct((m_pid["s2_winner_pid"] == pid).sum(), den_s2),
            })

        
        # --- FULL MATCH TOTALS (use param thresholds!) ---
        totals = (
            df_sets.groupby("child_event_id", as_index=False)
            .agg(total_home_points=("set_home_points", "sum"),
                total_away_points=("set_away_points", "sum"),
                n_sets=("set_no", "nunique"))
        )
        totals["total_points"] = totals["total_home_points"] + totals["total_away_points"]

        # (A) Match-by-match Over% (your existing interpretation)
        over_mask = totals["total_points"] > over_total_threshold  # <-- use parameter
        over_75_5_n = int(over_mask.sum())
        over_75_5_den = int(totals["total_points"].notna().sum())
        over_75_5_prob_pct = pct(over_75_5_n, over_75_5_den)

        # (B) NEW: Average total vs line (side-by-side with the %)
        avg_total_points = float(totals["total_points"].mean()) if over_75_5_den else None
        avg_total_vs_line = (avg_total_points - over_total_threshold) if avg_total_points is not None else None  # positive ⇒ above line

        # --- SET 1 TOTALS (use param thresholds!) ---
        s1 = df_sets[df_sets["set_no"] == 1][[
            "child_event_id", "set_home_points", "set_away_points"
        ]].copy()
        s1["set1_total_points"] = s1["set_home_points"] + s1["set_away_points"]
        valid = s1["set1_total_points"].notna().sum()
        over = (s1["set1_total_points"] > set1_over_threshold).sum()  # <-- use parameter
        set1_over_18_5_pct = pct(over, valid)
        set1_under_18_5_pct = pct(valid - over, valid)

        # Merge for UI tables (unchanged except we keep param lines)
        df_totals = totals.merge(
            df_h2h[["child_event_id", "home_name", "away_name", "score", "time"]],
            on="child_event_id", how="left"
        ).merge(
            s1[["child_event_id", "set1_total_points"]],
            on="child_event_id", how="left"
        ).sort_values("time").reset_index(drop=True)

        df_totals["over_param_line"] = over_total_threshold
        df_totals["over_match_over_line"] = df_totals["total_points"] > over_total_threshold
        df_totals["set1_param_line"] = set1_over_threshold
        df_totals["set1_over_line"] = df_totals["set1_total_points"] > set1_over_threshold



    else:
        df_totals = pd.DataFrame(columns=[
            "child_event_id","total_home_points","total_away_points","n_sets",
            "total_points","home_name","away_name","score","time",
            "set1_total_points",
            "over_param_line","over_match_over_line",
            "set1_param_line","set1_over_line",
        ])


    per_set_over_rows = []
    if not df_sets.empty:
        for s in range(1, 5+1):
            s_df = df_sets[df_sets["set_no"] == s].copy()
            if s_df.empty:
                per_set_over_rows.append({"set_no": s, "over_pct": None, "n_over": 0, "den": 0})
                continue
            s_df["set_total"] = s_df["set_home_points"] + s_df["set_away_points"]
            den = int(s_df["set_total"].notna().sum())
            n_over = int((s_df["set_total"] > set1_over_threshold).sum())
            per_set_over_rows.append({
                "set_no": s,
                "over_pct": pct(n_over, den),
                "n_over": n_over,
                "den": den
            })
    per_set_over_df = pd.DataFrame(per_set_over_rows)

   

    metrics = {
        "s1_winner_wins_match_pct": s1_winner_wins_match_pct,
        "s1_winner_opponent_s2_pct": s1_winner_opponent_s2_pct,
        "s1_winner_s2_win_pct": s1_winner_s2_win_pct,
        "over_75_5_prob_pct": over_75_5_prob_pct,
        "over_75_5_n": over_75_5_n,
        "over_75_5_den": over_75_5_den,
        "avg_total_points": avg_total_points,          # <-- NEW
        "avg_total_vs_line": avg_total_vs_line,        # <-- NEW
        "set1_over_18_5_pct": set1_over_18_5_pct,
        "set1_under_18_5_pct": set1_under_18_5_pct,
        "players": per_player_rows,
        "per_set_over_df": per_set_over_df
    }


    time.sleep(rate_limit_seconds)

    return df_h2h, agg_summary, player_agg, player_agg_pivot, df_sets, df_totals, metrics
    # -----------------------
    # END: original content pasted 1:1
    # -----------------------


def recent_over_counts(df_sets: pd.DataFrame, set_no: int, threshold: float):
    out = {"L1": (0, 0), "L3": (0, 0), "L5": (0, 0), "L10": (0, 0), "ALL": (0, 0)}
    if df_sets.empty:
        return out
    s_df = df_sets[df_sets["set_no"] == set_no].copy()
    if s_df.empty:
        return out
    if "time" in s_df.columns:
        s_df = s_df.sort_values("time", ascending=False)
    s_df["set_total"] = s_df["set_home_points"] + s_df["set_away_points"]
    s_df = s_df[s_df["set_total"].notna()]
    if s_df.empty:
        return out
    totals = s_df["set_total"].tolist()
    def calc(n):
        k = min(n, len(totals))
        if k == 0:
            return (0, 0)
        arr = totals[:k]
        n_over = sum(1 for t in arr if t > threshold)
        return (n_over, k)
    out["L1"]  = calc(1)
    out["L3"]  = calc(3)
    out["L5"]  = calc(5)
    out["L10"] = calc(10)
    out["ALL"] = (sum(1 for t in totals if t > threshold), len(totals))
    return out


# =========================
# DATE-AWARE EVENTS FETCHER (NEW)
# =========================
def _base_events(endpoint: str, sport_id: int, token: str, league_id_or_none, tz: str, day_str: str | None):
    url = f"https://api.b365api.com/v3/events/{endpoint}?sport_id={sport_id}&token={token}"
    if league_id_or_none is not None:
        url += f"&league_id={league_id_or_none}"
    # BetsAPI supports &day=YYYYMMDD for upcoming/ended; inplay ignores day
    if day_str and endpoint in ("upcoming", "ended"):
        url += f"&day={day_str}"

    data = fetch_json(url)
    events = data.get("results", []) or []
    df = json_normalize(events, sep=".")
    if df.empty:
        return df
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"].astype(int), unit="s", utc=True, errors="coerce")
        df["time"] = df["time"].dt.tz_convert(tz).dt.tz_localize(None)
    df.rename(columns={"id": "event_id", "home.name": "home_name", "away.name": "away_name"}, inplace=True)
    keep = [c for c in ["event_id", "home_name", "away_name", "league.name", "time", "cc"] if c in df.columns]
    df = df[keep].copy() if keep else df.copy()
    if "event_id" in df.columns:
        df["event_id"] = df["event_id"].astype(str)
    return df

# Cache events list by settings (with optional local calendar date filter)
# Cache events list by settings (date-aware, handles UTC/local crossover)

@st.cache_data(show_spinner=False)
def load_events(
    _mode: str,
    _sport_id: int,
    _token: str,
    _league_id_or_none,
    _tz: str,
    date_filter: Optional[date] = None,
):
    """
    Date-aware + pagination:
      - For upcoming/ended with date_filter: fetch D-1, D, D+1 (UTC) and paginate all pages.
      - For upcoming/ended without date_filter: paginate the current day (UTC) only.
      - For inplay: single fetch (no &day), optional local-date filter if provided.
    Then convert to local tz and filter by LOCAL calendar date when date_filter is set.
    """
    def _one_page(endpoint: str, day_str: Optional[str], page: int) -> pd.DataFrame:
        url = f"https://api.b365api.com/v3/events/{endpoint}?sport_id={_sport_id}&token={_token}"
        if _league_id_or_none is not None:
            url += f"&league_id={_league_id_or_none}"
        if day_str is not None and endpoint in ("upcoming", "ended"):
            url += f"&day={day_str}"
        if endpoint in ("upcoming", "ended"):
            url += f"&page={page}"
        j = fetch_json(url)
        events = j.get("results", []) or []
        return json_normalize(events, sep=".")

    
    def _all_pages(endpoint: str, day_str: Optional[str]) -> pd.DataFrame:
        # pull every page; stop when empty or when a page has no new IDs
        pages = []
        seen = set()
        p = 1
        while True:
            dfp = _one_page(endpoint, day_str, p)
            if dfp.empty:
                break
            if "id" in dfp.columns:
                ids = dfp["id"].astype(str)
                new = dfp[~ids.isin(seen)].copy()
                if new.empty:
                    break
                seen.update(new["id"].astype(str).tolist())
                pages.append(new)
            else:
                pages.append(dfp)
            p += 1
            time.sleep(0.05)  # gentle rate-limit pacing
        return pd.concat(pages, ignore_index=True) if pages else pd.DataFrame()


    endpoint = {"upcoming": "upcoming", "inplay": "inplay", "ended": "ended"}[_mode]

    # ---- INPLAY: single fetch, optional local-date filter
    if endpoint == "inplay":
        df = _one_page(endpoint, None, 1)  # page ignored by API for inplay
        if df.empty:
            return df
        # Convert to local and (optionally) filter by local calendar date
        df["time_utc"] = pd.to_datetime(df["time"].astype(int), unit="s", utc=True, errors="coerce")
        df["time_local"] = df["time_utc"].dt.tz_convert(_tz)
        if date_filter is not None:
            target_date = pd.to_datetime(date_filter).date()
            df = df[df["time_local"].dt.date == target_date].copy()
            
        # --- MODIFICATION 1 (Inplay) ---
        df.rename(columns={"id": "event_id", "home.name": "home_name", "away.name": "away_name", "home.id": "home_id", "away.id": "away_id"}, inplace=True) # <-- MODIFIED
        df["time"] = df["time_local"].dt.tz_localize(None)
        keep = [c for c in ["event_id","home_name","away_name","league.name","time","cc", "home_id", "away_id"] if c in df.columns] # <-- MODIFIED
        if keep:
            df = df[keep].copy()
        df["event_id"] = df["event_id"].astype(str)
        if "home_id" in df.columns: df["home_id"] = df["home_id"].astype(str) # <-- NEW
        if "away_id" in df.columns: df["away_id"] = df["away_id"].astype(str) # <-- NEW
        return df.sort_values("time", ascending=True, na_position="last").reset_index(drop=True)

    # ---- UPCOMING / ENDED: paginate (with or without a date filter)
    if date_filter is None:
        # No date filter -> just paginate today's UTC day
        today_utc = pd.Timestamp.utcnow().strftime("%Y%m%d")
        df = _all_pages(endpoint, today_utc)
    else:
        # Date filter -> fetch D-1, D, D+1 UTC, then combine
        target_date = pd.to_datetime(date_filter).date()
        days = [
            (target_date - pd.Timedelta(days=1)).strftime("%Y%m%d"),
            target_date.strftime("%Y%m%d"),
            (target_date + pd.Timedelta(days=1)).strftime("%Y%m%d"),
        ]
        parts = []
        for ds in days:
            part = _all_pages(endpoint, ds)
            if not part.empty:
                parts.append(part)
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    if df.empty:
        return df

    # Convert to local tz and (if date_filter) keep only that local day
    df["time_utc"] = pd.to_datetime(df["time"].astype(int), unit="s", utc=True, errors="coerce")
    df["time_local"] = df["time_utc"].dt.tz_convert(_tz)

    if date_filter is not None:
        target_date = pd.to_datetime(date_filter).date()
        df = df[df["time_local"].dt.date == target_date].copy()

    # Shape for UI
    df = df.drop_duplicates(subset=["id"], keep="last")
    
    # --- MODIFICATION 2 (Upcoming/Ended) ---
    df.rename(columns={"id": "event_id", "home.name": "home_name", "away.name": "away_name", "home.id": "home_id", "away.id": "away_id"}, inplace=True) # <-- MODIFIED
    df["time"] = df["time_local"].dt.tz_localize(None)
    keep = [c for c in ["event_id","home_name","away_name","league.name","time","cc", "home_id", "away_id"] if c in df.columns] # <-- MODIFIED
    if keep:
        df = df[keep].copy()
    df["event_id"] = df["event_id"].astype(str)
    if "home_id" in df.columns: df["home_id"] = df["home_id"].astype(str) # <-- NEW
    if "away_id" in df.columns: df["away_id"] = df["away_id"].astype(str) # <-- NEW
    return df.sort_values("time", ascending=True, na_position="last").reset_index(drop=True)




# =========================
# Cache per-event computations and odds (UNCHANGED)
# =========================
@st.cache_data(show_spinner=False)
def compute_event_package(
    eid: int,
    timezone_local: str,
    sport_id: int,
    token: str,
    history_qty: int,
    over_total_threshold: float,
    set1_over_threshold: float,
    rate_limit_seconds: float,
    event_cutoff_dt: Optional[datetime] = None,   # <-- NEW
):
    return build_h2h_report_for_event(
        event_id=eid,
        timezone_local=timezone_local,
        sport_id=sport_id,
        token=token,
        history_qty=history_qty,
        over_total_threshold=over_total_threshold,
        set1_over_threshold=set1_over_threshold,
        rate_limit_seconds=rate_limit_seconds,
        event_cutoff_dt=event_cutoff_dt,
    )

@st.cache_data(show_spinner=False)
def get_odds_summary_cached(eid: int, sport_id: int, token: str):
    return fetch_event_odds_summary(eid, sport_id, token)

# ----- persistent selection log across reruns -----
if "download_log" not in st.session_state:
    # maps event_id -> basic info we’ll use when exporting
    st.session_state["download_log"] = {}


# =========================
# Streamlit App
# =========================
st.set_page_config(page_title="TT Elite Series H2H + Odds", layout="wide")
st.title("🏓 TT Elite Series — H2H, Per-Set Metrics & Odds (ID-based)")

with st.sidebar:
    st.header("Configuration")
    token = st.text_input("API Token", value=DEFAULT_TOKEN, type="password", key="token")
    sport_id = st.number_input("Sport ID", value=DEFAULT_SPORT_ID, step=1, key="sport_id")

    view_mode = st.selectbox(
        "Main View",
        options=["Main View", "Hot Sheets"],
        index=0,
        key="view_mode"
    )

    league_choice = st.selectbox(
        "League",
        options=[item["label"] for item in LEAGUES],
        index=1,
        key="league_choice"
    )
    league_obj = next(x for x in LEAGUES if x["label"] == league_choice)
    league_id = league_obj["league_id"]
    league_name = league_obj["name"]

    timezone_local = st.text_input("Timezone", value=DEFAULT_TZ, key="timezone_local")

    up_or_end = st.radio(
        "Event Type",
        options=["upcoming", "inplay", "ended"],
        index=1,
        key="mode_radio"
    )

    # 🔹 Date Filter (NEW)
    filter_by_date = st.checkbox("Filter by specific date", value=False)
    selected_date  = st.date_input("Pick date", value=pd.Timestamp.now(tz=timezone_local).date(), disabled=not filter_by_date)

    limit_events = st.number_input("Limit events (optional)", min_value=0, value=DEFAULT_LIMIT_EVENTS or 0, step=1, key="limit_events_raw")
    limit_events = None if st.session_state.get("limit_events_raw", 0) == 0 else int(st.session_state["limit_events_raw"])

    history_qty = st.slider("History qty (H2H matches to fetch)", 5, 50, DEFAULT_HISTORY_QTY, 1, key="history_qty")
    over_total_threshold = st.number_input("Full-match Over threshold (points)", value=DEFAULT_OVER_TOTAL_THRESHOLD, step=0.5, format="%.1f", key="over_total_threshold")
    set1_over_threshold = st.number_input("Set 1 Over threshold (points)", value=DEFAULT_SET1_OVER_THRESHOLD, step=0.5, format="%.1f", key="set1_over_threshold")
    rate_limit_seconds = st.number_input("Rate limit sleep (seconds)", value=DEFAULT_RATE_LIMIT_SECONDS, step=0.05, format="%.2f", key="rate_limit_seconds")

    refresh = st.button("🔄 Refresh Events")

# Min H2H sample (applies to Main View & Hot Sheets)
MIN_H2H_OPTIONS = [0, 3, 5, 10, 15, 20, 30]
min_h2h_required = st.selectbox(
    "Min H2H sample (matches)",
    options=MIN_H2H_OPTIONS,
    index=MIN_H2H_OPTIONS.index(10),   # default 10
    help="Filter out matchups with fewer historical H2H matches than this."
)

# =========================
# VIEW SWITCH
# =========================
if st.session_state.get("view_mode", "Main View") == "Main View":
    colL, colR = st.columns([2, 3], gap="large")
    
    # ---------- LEFT SIDE (Main View) ----------
    with colL:
        # manual refresh clears caches
        if refresh:
            st.cache_data.clear()

        # fetch events with the SAME sidebar controls (+ date filter)
        with st.spinner("Fetching events..."):
            date_arg = selected_date if filter_by_date else None
            events_df = load_events(
                up_or_end,          # "upcoming" | "inplay" | "ended" (from sidebar)
                sport_id,           # from sidebar
                token,              # from sidebar
                league_id,          # from sidebar (ALL or a specific league)
                timezone_local,     # from sidebar
                date_arg            # None or a date
            )

        # header
        hdr_mode = {"upcoming": "Upcoming", "inplay": "Live", "ended": "Ended"}[up_or_end]
        if filter_by_date:
            st.caption(f"Date filter active: **{selected_date.strftime('%Y-%m-%d')}** (local TZ)")

        # --- Player search (fast, accent/case-insensitive, multi-term) ---
        import unicodedata, re
        def _norm_txt(s: str) -> str:
            try:
                s = "" if s is None else str(s)
                s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
                return s.lower().strip()
            except Exception:
                return ""

        player_query = st.text_input(
            "🔎 Search player (Main View)",
            value=st.session_state.get("player_query", ""),
            key="player_query",
            placeholder="Type one or more names, e.g. ivan, petrov"
        )

        events_display = events_df.copy()
        qraw = (player_query or "").strip()
        if qraw and not events_display.empty:
            terms = [t for t in re.split(r"[,\s;]+", qraw) if t]
            hn = events_display["home_name"].map(_norm_txt) if "home_name" in events_display.columns else ""
            an = events_display["away_name"].map(_norm_txt) if "away_name" in events_display.columns else ""
            mask_any = None
            for t in terms:
                tnorm = _norm_txt(t)
                m = hn.str.contains(tnorm, na=False) | an.str.contains(tnorm, na=False)
                mask_any = m if mask_any is None else (mask_any | m)
            events_display = events_display[mask_any].copy() if mask_any is not None else events_display

        # optional limit (treat 0 or None as "no limit")
        if limit_events and limit_events > 0 and not events_display.empty and len(events_display) > limit_events:
            events_display = events_display.head(limit_events).copy()

        st.subheader(f"Events ({hdr_mode}) — {'ALL Leagues' if league_name == 'ALL' else league_name}")

        selected_eid = None
        if events_display.empty:
            st.warning("No events match your filters/search.")
        else:
            # build pretty labels for the picker
            def _mk_label(row):
                t = row.get("time")
                t_str = t.strftime("%Y-%m-%d %H:%M") if (t is not None and not pd.isna(t)) else "N/A"
                return f"{row.get('home_name','?')} vs {row.get('away_name','?')} — {t_str}"

            id_list = events_display["event_id"].tolist()
            label_by_id = {row["event_id"]: _mk_label(row) for _, row in events_display.iterrows()}

            prev_eid = st.session_state.get("selected_eid")
            default_index = id_list.index(prev_eid) if prev_eid in id_list else 0

            # ======= Select multiple matchups to a temp editor + add to log =======
            if not events_display.empty:
                # Build a small table for checkboxes
                _table = events_display.copy()
                _table["Label"] = _table.apply(
                    lambda r: f"{r.get('home_name','?')} vs {r.get('away_name','?')} — {r.get('time').strftime('%Y-%m-%d %H:%M') if pd.notna(r.get('time')) else 'N/A'}",
                    axis=1
                )

                # Default "Include" based on what's already in the persistent log
                _table["Include"] = _table["event_id"].astype(str).map(lambda eid: eid in st.session_state["download_log"]).astype(bool)

                # Keep only columns we want to show
                ed_cols = ["Include", "event_id", "Label", "league.name", "time"]
                ed_cols = [c for c in ed_cols if c in _table.columns]
                _view = _table[ed_cols].rename(columns={"league.name": "League", "event_id": "Event ID"})

                st.markdown("#### Add matchups to your download log")
                edited = st.data_editor(
                    _view,
                    hide_index=True,
                    use_container_width=True,
                    key="multi_pick_editor",
                    column_config={"Include": st.column_config.CheckboxColumn("Include")}
                )

                c_add, c_clear = st.columns([1,1])
                if c_add.button("➕ Add checked to log"):
                    # Add all rows with Include==True to the persistent log
                    add_rows = edited[edited["Include"] == True] if "Include" in edited.columns else edited.iloc[0:0]
                    for _, rr in add_rows.iterrows():
                        eid = str(rr["Event ID"])
                        st.session_state["download_log"][eid] = {
                            "event_id": eid,
                            "label": rr.get("Label"),
                            "league": rr.get("League"),
                            "time": rr.get("time"),
                        }
                    st.success(f"Added {len(add_rows)} matchup(s) to the log.")

                if c_clear.button("🧹 Clear log"):
                    st.session_state["download_log"].clear()
                    st.info("Selection log cleared.")


            selected_eid = st.selectbox(
                "Select a matchup",
                options=id_list,
                index=default_index,
                key="selected_eid",
                format_func=lambda eid: label_by_id.get(eid, str(eid))
            )

            # force recompute often for live view
            if up_or_end == "inplay":
                st.cache_data.clear()


            # we already have events_display with a 'time' column
            event_time = None
            try:
                event_time = events_display.loc[events_display["event_id"] == selected_eid, "time"].iloc[0]
            except Exception:
                pass

            


            # compute the selected matchup package (right column reads these vars)
            with st.spinner("Computing H2H & set-by-set metrics..."):
                (df_h2h, agg_summary, player_agg, player_agg_pivot,
                df_sets, df_totals, metrics) = compute_event_package(
                    int(selected_eid),
                    timezone_local, sport_id, token,
                    history_qty, over_total_threshold, set1_over_threshold, rate_limit_seconds,
                    event_cutoff_dt=event_time,     # <-- NEW
                )

            # live/ended overview + odds table (kept here so you can preview on the left too)
            ov = get_event_overview(int(selected_eid), sport_id, token)
            if ov:
                p1_pid = ov.get("home_id")
                p2_pid = ov.get("away_id")
                title_home = ov.get("home_name", "Home")
                title_away = ov.get("away_name", "Away")
                status = ov.get("status", "Unknown")
                sets_score = ov.get("sets_score")
                total_points = ov.get("total_points", 0)
                sets_table = ov.get("sets_table", [])


                st.markdown("### Matchup Display")
                st.markdown(
                    f"**{title_home} vs {title_away}**  ·  "
                    f"**Status:** {status}  ·  "
                    f"**Sets:** {sets_score or '—'}  ·  "
                    f"**Total Points:** {total_points}"
                )

                # small odds snapshot
                totals, match_ml, set1_ml = {}, {}, {}   # <— add this line 
                try:
                    odds_summary = get_odds_summary_cached(int(selected_eid), sport_id, token)
                    stage = "end" if up_or_end == "inplay" else "start"
                    totals, match_ml, set1_ml = parse_event_odds_summary(
                        odds_summary, title_home, title_away, preferred_stage=stage
                    )
                    if totals:
                        st.caption(f"Totals line: {totals.get('line', '—')} | Over {totals.get('over_dec')} / Under {totals.get('under_dec')}")
                    else:
                        st.caption("Totals: not available.")
                except Exception as _e:
                    st.caption(f"Odds read error: {_e}")

                # === Pretty matchup + full set table + odds (DROP-IN) ===


                # Two columns: left = sets table, right = odds tables
         
                if sets_table:
                    disp = pd.DataFrame(sets_table)
                    disp = disp.sort_values("set_no").reset_index(drop=True)
                    disp.rename(
                        columns={
                            "set_no": "Set",
                            "home": title_home,
                            "away": title_away,
                            "total": "Total"
                        },
                        inplace=True
                    )
                    st.dataframe(disp, use_container_width=True, hide_index=True)
                else:
                    st.info("No set-by-set data available for this event yet.")

            # ----- RIGHT: Odds tables (Totals, Match ML, 1st Set ML) -----
            
                # ----- RIGHT: Odds tables (Totals, Match ML, 1st Set ML) -----
                st.markdown("**Totals (O/U)**")
                if totals:
                    to_df = pd.DataFrame([{
                        "Line": totals.get("line"),
                        "Over (Dec)": totals.get("over_dec"),
                        "Over (Amer)": totals.get("over_amer"),
                        "Under (Dec)": totals.get("under_dec"),
                        "Under (Amer)": totals.get("under_amer"),
                        "Book": totals.get("book") or "—",
                    }])
                    st.dataframe(to_df, use_container_width=True, hide_index=True)
                else:
                    st.caption("Totals: not available.")

                st.markdown("**Match Moneyline**")
                if match_ml:
                    mm_df = pd.DataFrame([
                        {"Side": title_home, "Dec": match_ml.get("home_dec"), "Amer": match_ml.get("home_amer"), "Book": match_ml.get("book") or "—"},
                        {"Side": title_away, "Dec": match_ml.get("away_dec"), "Amer": match_ml.get("away_amer"), "Book": match_ml.get("book") or "—"},
                    ])
                    st.dataframe(mm_df, use_container_width=True, hide_index=True)
                else:
                    st.caption("Match ML: not available.")

                st.markdown("**1st Set Moneyline**")
                if set1_ml:
                    s1_df = pd.DataFrame([
                        {"Side": title_home, "Dec": set1_ml.get("home_dec"), "Amer": set1_ml.get("home_amer"), "Book": set1_ml.get("book") or "—"},
                        {"Side": title_away, "Dec": set1_ml.get("away_dec"), "Amer": set1_ml.get("away_amer"), "Book": set1_ml.get("book") or "—"},
                    ])
                    st.dataframe(s1_df, use_container_width=True, hide_index=True)
                else:
                    st.caption("1st Set ML: not available.")


                # Optional raw debug
                with st.expander("Raw odds payload (debug)", expanded=False):
                    st.json(odds_summary.get("payload", {}))
            

            else:
                st.info("No live/ended overview returned for this event.")


            st.markdown("### 📥 Download selected matchups")

            if not st.session_state["download_log"]:
                st.caption("No matchups in your log yet. Use the checkboxes on the left and click **Add checked to log**.")
            else:
                with st.expander(f"{len(st.session_state['download_log'])} matchup(s) queued", expanded=False):
                    for eid, info in st.session_state["download_log"].items():
                        st.markdown(f"- {info.get('label') or eid}")


                # =========================
                # ⬇️ Download selected matchups (Excel) — FULL REPLACEMENT BLOCK (PER-PLAYER)
                # =========================
                if st.button("⬇️ Download selected matchups (Excel)"):
                    out_buf = io.BytesIO()
                    used_names: set[str] = set()

                    HI_PCT = 75.0
                    LO_PCT = 25.0

                    

                    # ---- helpers for per-player stats ----
                    def _overall_win_pct_by_pid(df_local: pd.DataFrame, pid: Optional[str]):
                        if df_local.empty or not pid:
                            return None
                        pool = df_local[(df_local["home_id"] == pid) | (df_local["away_id"] == pid)]
                        if pool.empty:
                            return None
                        wins = int((pool["match_winner_pid"] == pid).sum())
                        den  = int(len(pool))
                        return (wins / den * 100.0) if den else None

                    def _player_row_for_pid(players_list, pid, fallback_name):
                        """
                        Find the dict for this PID in metrics['players']; otherwise return empty shell.
                        """
                        row = next((x for x in (players_list or []) if str(x.get("pid")) == str(pid)), None)
                        if not row:
                            row = {"pid": pid, "name": fallback_name,
                                "s1_win_match_pct": None,
                                "s1_opponent_s2_pct": None,
                                "s1_s2_win_pct": None}
                        return row

                    # ---- build summary rows (one row per metric per selected game) ----
                    summary_rows = []

                    def _add_summary_row(base_row, metric_label, metric_key, pct_val, ov, odds, ended: bool):
                        rec_txt, desired, units, _is_lean = mainview_recommend_over_only(metric_key, pct_val)
                        

                        def _final_value_for_metric(metric_key: str, ov: dict):
                            if not ov:
                                return None
                            sets = ov.get("sets_table") or []
                            if metric_key == "full_over_pct":
                                have_any = any(s.get("home") is not None and s.get("away") is not None for s in sets)
                                if not have_any:
                                    return None
                                return sum(
                                    int(s["home"]) + int(s["away"])
                                    for s in sets
                                    if s.get("home") is not None and s.get("away") is not None
                                )
                            if metric_key in ("set1_over_18_5_pct",):
                                for s in sets:
                                    if s.get("set_no") == 1 and s.get("home") is not None and s.get("away") is not None:
                                        return int(s["home"]) + int(s["away"])
                                return None
                            return None

                        if ended:
                            if rec_txt and desired:
                                outcome, fval = evaluate_outcome_and_value(desired, ov)
                                rec_result = outcome if outcome in ("Win", "Loss") else "N/A"
                                final_value = fval
                            else:
                                rec_result = "N/A"
                                final_value = _final_value_for_metric(metric_key, ov)
                        else:
                            rec_result = "Pending"
                            final_value = None

                        summary_rows.append({
                            **base_row,
                            "Rec_Metric": metric_label,          # e.g., "Full Over %", "Set 1 Over %"
                            "Rec_Text": rec_txt or "No recommendation",
                            "Rec_ProbPct": pct_val,
                            "Rec_Units": units,                  # 0.5 for leans; ladder for strong; None for no rec
                            "Final_Value": final_value,
                            "Rec_Result": rec_result,
                            # Odds (American only)
                            "OU_Line": odds.get("ou_line"),
                            "Over_Amer": odds.get("over_amer"),
                            "Under_Amer": odds.get("under_amer"),
                            "P1_ML_Amer": odds.get("p1_ml_amer"),
                            "P2_ML_Amer": odds.get("p2_ml_amer"),
                            "S1_P1_ML_Amer": odds.get("s1_p1_ml_amer"),
                            "S1_P2_ML_Amer": odds.get("s1_p2_ml_amer"),
                            "Odds_Book": odds.get("book"),
                        })

                    # ---------- per selected event ----------
                    for eid, info in st.session_state["download_log"].items():
                        try:
                            eid_int = int(str(info["event_id"]))
                        except Exception:
                            eid_int = info["event_id"]

                        ev_time = info.get("time", None)

                        # Overview (names, PIDs, status)
                        try:
                            ov = get_event_overview(int(eid_int), sport_id, token)
                        except Exception:
                            ov = {}

                        home_nm = ov.get("home_name", "")
                        away_nm = ov.get("away_name", "")
                        p1_pid  = ov.get("home_id")
                        p2_pid  = ov.get("away_id")
                        match_txt = f"{home_nm or '?'} vs {away_nm or '?'}"

                        # Event package (date-aware cutoff)
                        (df_h2h_x, agg_summary_x, _pa_x, _piv_x,
                        df_sets_x, df_totals_x, metrics_x) = compute_event_package(
                            int(eid_int),
                            timezone_local, sport_id, token,
                            history_qty,
                            float(st.session_state.get("over_total_threshold", DEFAULT_OVER_TOTAL_THRESHOLD)),
                            float(st.session_state.get("set1_over_threshold", DEFAULT_SET1_OVER_THRESHOLD)),
                            rate_limit_seconds,
                            event_cutoff_dt=ev_time,
                        )

                        # Odds snapshot (American only)
                        try:
                            odds_summary = get_odds_summary_cached(int(eid_int), sport_id, token)
                            preferred = "end" if (ov.get("time_status") in ("3",) or (ov.get("status") == "Finished")) else "start"
                            totals_odds, match_ml, set1_ml = parse_event_odds_summary(
                                odds_summary, home_nm, away_nm, preferred_stage=preferred
                            )
                        except Exception:
                            totals_odds, match_ml, set1_ml = {}, {}, {}

                        odds_pack = {
                            "ou_line": totals_odds.get("line"),
                            "over_amer": totals_odds.get("over_amer"),
                            "under_amer": totals_odds.get("under_amer"),
                            "p1_ml_amer": match_ml.get("home_amer"),
                            "p2_ml_amer": match_ml.get("away_amer"),
                            "s1_p1_ml_amer": set1_ml.get("home_amer"),
                            "s1_p2_ml_amer": set1_ml.get("away_amer"),
                            "book": totals_odds.get("book") or match_ml.get("book") or set1_ml.get("book"),
                        }

                        # ---- PER-PLAYER values (by PID) ----
                        players_list = metrics_x.get("players", []) or []
                        p1_row = _player_row_for_pid(players_list, p1_pid, home_nm)
                        p2_row = _player_row_for_pid(players_list, p2_pid, away_nm)

                        p1_overall = _overall_win_pct_by_pid(df_h2h_x, p1_pid)
                        p2_overall = _overall_win_pct_by_pid(df_h2h_x, p2_pid)

                        # Ended?
                        ended = str(ov.get("time_status")) == "3" or ov.get("status") == "Finished"

                        base_row = {
                            "event_id": str(eid_int),
                            "match": match_txt,
                            "time": ev_time,
                            "H2H_n": int(agg_summary_x.loc[agg_summary_x["Score"] == "Total", "Count"].iloc[0]) if (
                                isinstance(agg_summary_x, pd.DataFrame)
                                and not agg_summary_x.empty
                                and "Score" in agg_summary_x.columns
                                and "Count" in agg_summary_x.columns
                                and "Total" in agg_summary_x["Score"].values
                            ) else None,

                            # 👇 P1 (HOME) — INDIVIDUAL
                            "P1 Name": home_nm,
                            "P1 Overall Win %": p1_overall,
                            "P1 S1 → Match Win %": p1_row.get("s1_win_match_pct"),
                            "P1 Opponent Wins S2 %": p1_row.get("s1_opponent_s2_pct"),
                            "P1 Wins S2 %": p1_row.get("s1_s2_win_pct"),

                            # 👇 P2 (AWAY) — INDIVIDUAL
                            "P2 Name": away_nm,
                            "P2 Overall Win %": p2_overall,
                            "P2 S1 → Match Win %": p2_row.get("s1_win_match_pct"),
                            "P2 Opponent Wins S2 %": p2_row.get("s1_opponent_s2_pct"),
                            "P2 Wins S2 %": p2_row.get("s1_s2_win_pct"),
                        }

                        # Metrics we export (Over-only)
                        metric_list = [
                            ("Full Over %", "over_75_5_prob_pct", "full_over_pct"),
                            ("Set 1 Over %", "set1_over_18_5_pct", "set1_over_18_5_pct"),
                        ]

                        for label_txt, metric_key_in_metrics, direction_key in metric_list:
                            pct_val = metrics_x.get(metric_key_in_metrics)
                            _add_summary_row(base_row, label_txt, direction_key, pct_val, ov, odds_pack, ended)

                    # -------------- Write Excel once --------------
                    with pd.ExcelWriter(out_buf, engine="xlsxwriter") as writer:
                        summary_df = pd.DataFrame(summary_rows)

                        display_df = summary_df.rename(columns={
                            "Rec_ProbPct": "Rec Prob %",
                            "Rec_Units": "Units",
                            "Final_Value": "Final Total/Score",
                            "Rec_Result": "Rec Result",
                            "OU_Line": "OU Line",
                            "Over_Amer": "Over (Amer)",
                            "Under_Amer": "Under (Amer)",
                            "P1_ML_Amer": "P1 ML (Amer)",
                            "P2_ML_Amer": "P2 ML (Amer)",
                            "S1_P1_ML_Amer": "S1 P1 ML (Amer)",
                            "S1_P2_ML_Amer": "S1 P2 ML (Amer)",
                            "Odds_Book": "Odds Book",
                        })

                        def pct_fmt(x):
                            try: return "" if pd.isna(x) else f"{float(x):.1f}%"
                            except: return ""

                        def units_fmt(u):
                            if u is None: return ""
                            try:
                                fu = float(u)
                                return f"{int(fu)}" if float(fu).is_integer() else f"{fu:.1f}"
                            except:
                                return str(u)

                        def final_fmt(v):
                            if v is None: return ""
                            try:
                                if isinstance(v, (int, np.integer)): return str(int(v))
                                fv = float(v)
                                return f"{int(fv)}" if float(fv).is_integer() else f"{fv:.0f}"
                            except:
                                return str(v)

                        sheet_summary = "Hot Sheets (Selected)"
                        display_df.to_excel(writer, sheet_name=sheet_summary, index=False)
                        wb = writer.book
                        ws = writer.sheets[sheet_summary]

                        # Column widths
                        for i, col in enumerate(display_df.columns):
                            width = max(12, min(44, int(display_df[col].astype(str).str.len().clip(upper=44).mean() + 4)))
                            ws.set_column(i, i, width)

                        # Rewrite cells with formatting
                        for r in range(1, len(display_df) + 1):
                            for c, col in enumerate(display_df.columns):
                                val = display_df.iloc[r-1, c]

                                if isinstance(col, str) and col.strip().endswith("%"):
                                    ws.write(r, c, pct_fmt(val))
                                elif col == "Units":
                                    ws.write(r, c, units_fmt(val))
                                elif col == "Final Total/Score":
                                    ws.write(r, c, final_fmt(val))
                                elif col == "OU Line":
                                    if val is None or (isinstance(val, float) and pd.isna(val)):
                                        ws.write(r, c, "")
                                    else:
                                        ws.write(r, c, f"{float(val):.1f}")
                                else:
                                    ws.write(r, c, "" if val is None or (isinstance(val, float) and pd.isna(val)) else val)

                        # Conditional formatting
                        fmt_green_text  = wb.add_format({"font_color": "#2ecc71", "bold": True})
                        fmt_purple_text = wb.add_format({"font_color": "#9b59b6", "bold": True})
                        fmt_win_fill    = wb.add_format({"bg_color": "#e6ffe6", "font_color": "#077307", "bold": True})
                        fmt_loss_fill   = wb.add_format({"bg_color": "#ffe6e6", "font_color": "#8a0000", "bold": True})

                        headers = list(display_df.columns)
                        nrows = len(display_df)

                        pct_cols_excel = [i for i, h in enumerate(headers)
                                        if (isinstance(h, str) and (
                                            h.endswith("_pct") or h.startswith("pct_") or h.endswith("_over_pct") or
                                            h == "Rec Prob %" or h.strip().endswith("%")
                                        ))]

                        for c in pct_cols_excel:
                            ws.conditional_format(1, c, nrows, c, {"type": "cell", "criteria": ">=", "value": HI_PCT, "format": fmt_green_text})
                            ws.conditional_format(1, c, nrows, c, {"type": "cell", "criteria": "<=", "value": LO_PCT, "format": fmt_green_text})
                            ws.conditional_format(1, c, nrows, c, {"type": "cell", "criteria": "between", "minimum": 65.0, "maximum": HI_PCT-1e-9, "format": fmt_purple_text})
                            ws.conditional_format(1, c, nrows, c, {"type": "cell", "criteria": "between", "minimum": LO_PCT+1e-9, "maximum": 35.0, "format": fmt_purple_text})

                        if "Rec Result" in headers:
                            rc = headers.index("Rec Result")
                            ws.conditional_format(1, rc, nrows, rc, {"type": "text", "criteria": "containing", "value": "Win",  "format": fmt_win_fill})
                            ws.conditional_format(1, rc, nrows, rc, {"type": "text", "criteria": "containing", "value": "Loss", "format": fmt_loss_fill})

                    st.download_button(
                        "Download file",
                        data=out_buf.getvalue(),
                        file_name=f"selected_matchups_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_selected_matchups",
                    )

                    st.session_state["download_log"].clear()
                    st.success("Download ready. Your selection log has been cleared.")





    # ---------- RIGHT SIDE ----------
    with colR:
        if selected_eid is None:
            st.info("Pick a matchup on the left to see metrics.")
        else:
            try:
                # --- H2H sample size for this matchup ---
                h2h_n = 0
                if isinstance(agg_summary, pd.DataFrame) and not agg_summary.empty:
                    if "Score" in agg_summary.columns and "Count" in agg_summary.columns:
                        try:
                            h2h_n = int(agg_summary.loc[agg_summary["Score"] == "Total", "Count"].iloc[0])
                        except Exception:
                            h2h_n = int(agg_summary["Count"].sum())

                # Gate the whole UI by Min H2H
                if h2h_n < min_h2h_required:
                    st.warning(
                        f"This matchup has only **{h2h_n}** H2H matches. "
                        f"Minimum required is **{min_h2h_required}** (adjust in the sidebar)."
                    )
                else:
                    # === Your existing Main View content (unchanged) ===
                    st.subheader("Matchup Summary")
                    with st.expander("📊 Matchup Summary (Set-1, Totals & H2H Scorelines)", expanded=True):
                        colA, colB, colC = st.columns(3)
                        with colA:
                            st.markdown(
                                f"""
                                **Set-1 Winner → Wins Match**  
                                - {fmt_pct(metrics.get('s1_winner_wins_match_pct'))}

                                **Set-1 Winner Opponent Wins S2**  
                                - {fmt_pct(metrics.get('s1_winner_opponent_s2_pct'))}

                                **Set-1 Winner Wins S2**  
                                - {fmt_pct(metrics.get('s1_winner_s2_win_pct'))}
                                """,
                                unsafe_allow_html=True,
                            )
                        with colB:
                            over_n = metrics.get("over_75_5_n") or 0
                            over_d = metrics.get("over_75_5_den") or 0
                            st.markdown(
                                f"""
                                **Full-Match Over {float(st.session_state.get('over_total_threshold', DEFAULT_OVER_TOTAL_THRESHOLD)):.1f}**  
                                - {fmt_pct(metrics.get('over_75_5_prob_pct'))} ({over_n}/{over_d})

                                **Set 1 Over {float(st.session_state.get('set1_over_threshold', DEFAULT_SET1_OVER_THRESHOLD)):.1f}**  
                                - {fmt_pct(metrics.get('set1_over_18_5_pct'))}

                                **Set 1 Under {float(st.session_state.get('set1_over_threshold', DEFAULT_SET1_OVER_THRESHOLD)):.1f}**  
                                - {fmt_pct(metrics.get('set1_under_18_5_pct'))}
                                """,
                                unsafe_allow_html=True,
                            )



                            avg_total = metrics.get("avg_total_points")
                            avg_delta = metrics.get("avg_total_vs_line")
                            line_val = float(st.session_state.get('over_total_threshold', DEFAULT_OVER_TOTAL_THRESHOLD))

                            
                            if avg_total is not None and avg_delta is not None:
                                sign = "+" if avg_delta >= 0 else ""
                                avg_line_txt = f"{avg_total:.1f} ({sign}{avg_delta:.1f} vs {line_val:.1f})"
                            else:
                                avg_line_txt = "—"
                            st.markdown(f"**Average Total vs Line**\n- **{avg_line_txt}**")


                        with colC:
                            def _scoreline_block(df, key: str):
                                if df is None or df.empty:
                                    return fmt_pct(None), "0/0"
                                row = df[df["Score"] == key]
                                tot = int(df.loc[df["Score"] == "Total", "Count"].iloc[0]) if "Total" in df["Score"].values else int(df["Count"].sum())
                                if row.empty:
                                    return fmt_pct(0.0), f"0/{tot}"
                                cnt = int(row["Count"].iloc[0])
                                pctv = float(row["Percent"].iloc[0]) if "Percent" in row.columns else (100.0 * cnt / max(1, tot))
                                return fmt_pct(pctv), f"{cnt}/{tot}"
                            st.markdown("**H2H Scorelines (mirrored)**")
                            p30, f30 = _scoreline_block(agg_summary, "3-0")
                            p31, f31 = _scoreline_block(agg_summary, "3-1")
                            p32, f32 = _scoreline_block(agg_summary, "3-2")
                            st.markdown(f"- 3-0: {p30} ({f30})", unsafe_allow_html=True)
                            st.markdown(f"- 3-1: {p31} ({f31})", unsafe_allow_html=True)
                            st.markdown(f"- 3-2: {p32} ({f32})", unsafe_allow_html=True)

                    # Set Over cards
                    st.subheader(f"Set Totals Over {float(st.session_state.get('set1_over_threshold', DEFAULT_SET1_OVER_THRESHOLD)):.1f} ")
                    with st.expander(f"📈 Over {float(st.session_state.get('set1_over_threshold', DEFAULT_SET1_OVER_THRESHOLD)):.1f} by Set (1–5)", expanded=False):
                        per_set_over_df = metrics.get("per_set_over_df", pd.DataFrame())
                        if per_set_over_df.empty:
                            st.info("No set-by-set totals available for per-set Over/Under.")
                        else:
                            cols = st.columns(5)
                            for i, s in enumerate(range(1, 6)):
                                slot = per_set_over_df[per_set_over_df["set_no"] == s]
                                if slot.empty:
                                    cols[i].markdown(
                                        f"**Set {s}**\n\n"
                                        f"- **—**\n"
                                        f"- n/a\n"
                                        f"- Last 1: —\n"
                                        f"- Last 3: —\n"
                                        f"- Last 5: —\n"
                                        f"- Last 10: —"
                                    )
                                    continue
                                pct_val = slot["over_pct"].iloc[0]
                                n_over  = int(slot["n_over"].iloc[0])
                                den     = int(slot["den"].iloc[0])
                                rec = recent_over_counts(df_sets, s, float(st.session_state.get("set1_over_threshold", DEFAULT_SET1_OVER_THRESHOLD)))
                                def fmt_pair(tup): return f"{tup[0]}/{tup[1]}" if tup and tup[1] else "0/0"
                                cols[i].markdown(
                                    f"""
                                    **Set {s}**  
                                    - Over {float(st.session_state.get('set1_over_threshold', DEFAULT_SET1_OVER_THRESHOLD)):.1f}: {fmt_pct(pct_val)}  
                                    - Total: **{n_over}/{den}**
                                    - Last 1:  **{fmt_pair(rec.get('L1'))}**
                                    - Last 3:  **{fmt_pair(rec.get('L3'))}**
                                    - Last 5:  **{fmt_pair(rec.get('L5'))}**
                                    - Last 10: **{fmt_pair(rec.get('L10'))}**
                                    """,
                                    unsafe_allow_html=True,
                                )

                    # -------------------------
                    # Player Summary (PID-based)
                    # -------------------------
                    st.subheader("Player Summary")

                    with st.expander("🏓 Per-Player (when this player wins Set 1)", expanded=False):

                        # Helper: overall win % and win-scoreline split for a PID (robust to name drift)
                        def _player_stats_by_pid(df_h2h: pd.DataFrame, pid: Optional[str]):
                            """
                            Overall & scoreline split using Player ID.
                            - overall_win_pct = wins / appearances for this PID
                            - scorelines computed from wins only (by this PID)
                            """
                            if df_h2h.empty or not pid:
                                return [], 0.0

                            # All matches where this PID participated (either side)
                            pool = df_h2h[(df_h2h["home_id"] == pid) | (df_h2h["away_id"] == pid)]
                            if pool.empty:
                                return [], 0.0

                            wins_mask = (pool["match_winner_pid"] == pid)
                            wins = int(wins_mask.sum())
                            total = int(len(pool))
                            overall_win_pct = (wins / total * 100.0) if total else 0.0

                            wins_df = pool[wins_mask].copy()
                            if "score_norm" not in wins_df.columns:
                                wins_df["score_norm"] = wins_df["score"].apply(normalize_score_str)

                            counts = wins_df["score_norm"].value_counts().sort_index()
                            total_wins = int(counts.sum()) if not counts.empty else 0

                            lines = []
                            for sc in ["3-0", "3-1", "3-2"]:
                                cnt = int(counts.get(sc, 0))
                                pctv = (cnt / total_wins * 100.0) if total_wins else 0.0
                                lines.append((sc, pctv, cnt, total_wins))
                            return lines, overall_win_pct

                        # Pull PIDs & names from overview you already fetched above
                        p1_pid = ov.get("home_id")
                        p2_pid = ov.get("away_id")
                        p1_name = ov.get("home_name", "Player 1")
                        p2_name = ov.get("away_name", "Player 2")

                        # metrics['players'] already carries PID-labeled S1 metrics from build_h2h_report_for_event
                        pp = metrics.get("players", []) or []
                        # Map them to the actual P1/P2 for this event (by PID)
                        a = next((x for x in pp if x.get("pid") == p1_pid), {"pid": p1_pid, "name": p1_name})
                        b = next((x for x in pp if x.get("pid") == p2_pid), {"pid": p2_pid, "name": p2_name})

                        if not p1_pid and not p2_pid:
                            st.info("No per-player breakdown available for this matchup.")
                        else:
                            cA, cB = st.columns(2)

                            # Card A (P1 = HOME)
                            with cA:
                                pname = a.get("name") or p1_name
                                ppid  = a.get("pid") or p1_pid
                                st.markdown(f"**{pname}** (pid: {ppid or '—'})")
                                linesA, overallA = _player_stats_by_pid(df_h2h, ppid)
                                st.markdown(
                                    f"""
                                    - Overall Win %: {fmt_pct(overallA)}
                                    - S1 → Match Win: {fmt_pct(a.get('s1_win_match_pct'))}
                                    - Opponent Wins S2: {fmt_pct(a.get('s1_opponent_s2_pct'))}
                                    - Wins S2: {fmt_pct(a.get('s1_s2_win_pct'))}
                                    """,
                                    unsafe_allow_html=True,
                                )
                                if linesA:
                                    st.markdown("**H2H Scorelines (mirrored)**")
                                    for sc, pctv, cnt, tot in linesA:
                                        st.markdown(f"- {sc}: {fmt_pct(pctv)} ({cnt}/{tot})", unsafe_allow_html=True)

                            # Card B (P2 = AWAY)
                            with cB:
                                pname = b.get("name") or p2_name
                                ppid  = b.get("pid") or p2_pid
                                st.markdown(f"**{pname}** (pid: {ppid or '—'})")
                                linesB, overallB = _player_stats_by_pid(df_h2h, ppid)
                                st.markdown(
                                    f"""
                                    - Overall Win %: {fmt_pct(overallB)}
                                    - S1 → Match Win: {fmt_pct(b.get('s1_win_match_pct'))}
                                    - Opponent Wins S2: {fmt_pct(b.get('s1_opponent_s2_pct'))}
                                    - Wins S2: {fmt_pct(b.get('s1_s2_win_pct'))}
                                    """,
                                    unsafe_allow_html=True,
                                )
                                if linesB:
                                    st.markdown("**H2H Scorelines (mirrored)**")
                                    for sc, pctv, cnt, tot in linesB:
                                        st.markdown(f"- {sc}: {fmt_pct(pctv)} ({cnt}/{tot})", unsafe_allow_html=True)


                        

                    # H2H table
                    st.subheader("H2H Matches")
                    with st.expander("📜 H2H Matches", expanded=False):
                        if df_h2h.empty:
                            st.info("No H2H matches found for this matchup.")
                        else:
                            st.dataframe(df_h2h, use_container_width=True, hide_index=True)

                    # Set Scores
                    st.subheader("Set Scores")
                    with st.expander("🎯 Set-by-Set Scores", expanded=False):
                        if df_sets.empty:
                            st.info("No set-by-set data available.")
                        else:
                            st.dataframe(df_sets, use_container_width=True, hide_index=True)

                    # Match Totals
                    st.subheader("Match Totals")
                    with st.expander("📊 Match Totals (Per Child Event)", expanded=False):
                        if df_totals.empty:
                            st.info("Totals not available for this matchup.")
                        else:
                            st.dataframe(df_totals, use_container_width=True, hide_index=True)

                    # Excel export
                    out_buf = io.BytesIO()
                    per_set_over_df = metrics.get("per_set_over_df", pd.DataFrame())
                    with pd.ExcelWriter(out_buf, engine="xlsxwriter") as writer:
                        if not df_h2h.empty: df_h2h.to_excel(writer, sheet_name="H2H Matches", index=False)
                        if not agg_summary.empty: agg_summary.to_excel(writer, sheet_name="Aggregated Summary", index=False)
                        if not player_agg.empty: player_agg.to_excel(writer, sheet_name="By Player (Aggregated)", index=False)
                        if not player_agg_pivot.empty: player_agg_pivot.to_excel(writer, sheet_name="By Player (Pivot)")
                        if not df_sets.empty: df_sets.to_excel(writer, sheet_name="Set by Set", index=False)
                        if not df_totals.empty: df_totals.to_excel(writer, sheet_name="Match Totals", index=False)
                        if not per_set_over_df.empty: per_set_over_df.to_excel(writer, sheet_name="Over 18.5 by Set", index=False)
                    st.download_button(
                        "⬇️ Download This Matchup (Excel)",
                        data=out_buf.getvalue(),
                        file_name=f"{up_or_end}_{league_name}_event_{selected_eid}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            except Exception as e:
                st.error(f"Error computing matchup package: {e}")

else:
    

    # =========================
    # 🔥 HOT SHEETS (matchups use SAME sidebar filters)
    # =========================
    st.subheader("🔥 Hot Sheets")

    # Threshold controls
    ctl1, ctl2, ctl3, ctl4 = st.columns([1.2, 1.0, 1.0, 1.0])
    hi2_val = ctl1.number_input("High ≥", min_value=0.0, max_value=100.0, value=75.0, step=1.0, key="hot_hi2")
    lo2_val = ctl2.number_input("Low ≤",  min_value=0.0, max_value=100.0, value=25.0, step=1.0, key="hot_lo2")
    hi1_val = ctl3.number_input("Mod-High ≥", min_value=0.0, max_value=100.0, value=65.0, step=1.0, key="hot_hi1")
    lo1_val = ctl4.number_input("Mod-Low ≤",  min_value=0.0, max_value=100.0, value=35.0, step=1.0, key="hot_lo1")

    cb1, cb2, cb3, cb4 = st.columns(4)
    use_high   = cb1.checkbox("Apply High (≥ High)", value=True, key="use_high")
    use_low    = cb2.checkbox("Apply Low (≤ Low)",   value=True, key="use_low")
    use_mhigh  = cb3.checkbox("Apply Mod-High (≥ Mod-High)", value=True, key="use_mhigh")
    use_mlow   = cb4.checkbox("Apply Mod-Low (≤ Mod-Low)",   value=True, key="use_mlow")

    MATCHUP_METRICS = [
        ("Set-1 → Match Win %",          "s1_winner_wins_match_pct"),
        ("Set-1 Opponent Wins S2 %",     "s1_winner_opponent_s2_pct"),
        ("Set-1 Winner Wins S2 %",       "s1_winner_s2_win_pct"),
        ("H2H 3-0 % (mirrored)",         "pct_3_0"),
        ("H2H 3-1 % (mirrored)",         "pct_3_1"),
        ("H2H 3-2 % (mirrored)",         "pct_3_2"),
    ]
    SETS_METRICS = [(f"Set {i} Over %", f"set{i}_over_pct") for i in range(1,6)]
    PLAYER_METRICS = [
        ("P1 Overall Win %",             "p1_overall_win_pct"),
        ("P2 Overall Win %",             "p2_overall_win_pct"),
        ("P1 S1→Match Win %",            "p1_s1_win_match_pct"),
        ("P2 S1→Match Win %",            "p2_s1_win_match_pct"),
        ("P1 S1 Opponent Wins S2 %",     "p1_s1_opponent_s2_pct"),
        ("P2 S1 Opponent Wins S2 %",     "p2_s1_opponent_s2_pct"),
        ("P1 S1 Wins S2 %",              "p1_s1_s2_win_pct"),
        ("P2 S1 Wins S2 %",              "p2_s1_s2_win_pct"),
    ]
    TOTALS_METRICS = [
        (f"Full Over {float(st.session_state.get('over_total_threshold', DEFAULT_OVER_TOTAL_THRESHOLD)):.1f} %", "full_over_pct"),
        (f"Set 1 Over {float(st.session_state.get('set1_over_threshold', DEFAULT_SET1_OVER_THRESHOLD)):.1f} %", "set1_over_18_5_pct"),
        (f"Set 1 Under {float(st.session_state.get('set1_over_threshold', DEFAULT_SET1_OVER_THRESHOLD)):.1f} %", "set1_under_18_5_pct"),
    ]

    m1, m2, m3, m4 = st.columns(4)
    pick_main = m1.multiselect("Main", [x[0] for x in MATCHUP_METRICS], default=[MATCHUP_METRICS[0][0], MATCHUP_METRICS[2][0]])
    pick_sets = m2.multiselect("Sets", [x[0] for x in SETS_METRICS], default=[SETS_METRICS[0][0], SETS_METRICS[1][0], SETS_METRICS[2][0]])
    pick_plrs = m3.multiselect("Player", [x[0] for x in PLAYER_METRICS], default=[PLAYER_METRICS[0][0], PLAYER_METRICS[1][0]])
    pick_totl = m4.multiselect("Totals", [x[0] for x in TOTALS_METRICS], default=[TOTALS_METRICS[0][0]])

    label2key = {lbl: key for (lbl, key) in (MATCHUP_METRICS + SETS_METRICS + PLAYER_METRICS + TOTALS_METRICS)}
    selected_metric_labels = pick_main + pick_sets + pick_plrs + pick_totl
    selected_keys = [label2key[lbl] for lbl in selected_metric_labels]

    oc1, oc2, oc3, oc4 = st.columns([0.9, 0.9, 0.9, 2.3])
    generate_hot  = oc1.button("⚡ Generate / Refresh", key="hot_generate")
    force_api     = oc2.checkbox("Force API refresh", value=True, key="hot_force")
    auto_hot      = oc3.checkbox("Auto-refresh 15s (inplay)", value=False, key="hot_auto")
    show_american = oc4.checkbox("Show American odds columns", value=False, key="hot_show_american")

    oc5, oc6 = st.columns([1.2, 1.8])
    hot_limit   = oc5.number_input("Hot Sheets: limit events (0 = no limit)", min_value=0, value=0, step=1, key="hot_limit")
    show_book    = oc6.checkbox("Show bookmaker/source column", value=True, key="hot_show_book")

    only_actionable = st.checkbox(
        "Only show actionable recs (hide no-rec & show 'pending' instead of N/A)",
        value=True, key="hot_only_actionable"
    )

    # auto-refresh for inplay
    if auto_hot and up_or_end == "inplay":
        do_live_autorefresh(enabled=True, interval_ms=15000, key="hot_auto_refresh")

    should_run = generate_hot or (auto_hot and up_or_end == "inplay")
    # Keep the last generated Hot Sheets in session so downloads / reruns don't wipe the UI
    if "hot_last" not in st.session_state:
        st.session_state["hot_last"] = None

    if "hot_expand_open" not in st.session_state:
        st.session_state["hot_expand_open"] = False


    if should_run and force_api:
        st.cache_data.clear()

    # load matchups using the SAME sidebar controls (mode/league/date/tz)
    with st.spinner("Loading candidate events for Hot Sheets..."):
        date_arg = selected_date if filter_by_date else None
        hot_events = load_events(up_or_end, sport_id, token, league_id, timezone_local, date_arg)

    if filter_by_date:
        st.caption(f"Hot Sheets date filter: **{selected_date.strftime('%Y-%m-%d')}** (local TZ)")

    # --- Player search for Hot Sheets ---
    import unicodedata, re
    def _norm_txt(s: str) -> str:
        try:
            s = "" if s is None else str(s)
            s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
            return s.lower().strip()
        except Exception:
            return ""

    player_query_hot = st.text_input(
        "🔎 Search player (Hot Sheets)",
        value=st.session_state.get("player_query_hot", ""),
        key="player_query_hot",
        placeholder="Type one or more names, e.g. ivan, petrov"
    )

    hot_events_display = hot_events.copy()
    qraw_hot = (player_query_hot or "").strip()
    if qraw_hot and not hot_events_display.empty:
        terms = [t for t in re.split(r"[,\s;]+", qraw_hot) if t]
        hn = hot_events_display["home_name"].map(_norm_txt) if "home_name" in hot_events_display.columns else ""
        an = hot_events_display["away_name"].map(_norm_txt) if "away_name" in hot_events_display.columns else ""
        mask_any = None
        for t in terms:
            tnorm = _norm_txt(t)
            m = hn.str.contains(tnorm, na=False) | an.str.contains(tnorm, na=False)
            mask_any = m if mask_any is None else (mask_any | m)
        hot_events_display = hot_events_display[mask_any].copy() if mask_any is not None else hot_events_display

    # limit after filtering (0 = unlimited)
    if hot_limit and hot_limit > 0 and not hot_events_display.empty and len(hot_events_display) > hot_limit:
        hot_events_display = hot_events_display.head(hot_limit).copy()

    

    if not should_run:
        # If we have a previous result, render it instead of bailing out
        if st.session_state["hot_last"] is not None:
            hot_payload = st.session_state["hot_last"]
            # --- render from cached payload ---
            per_metric_panels = hot_payload["per_metric_panels"]
            styled_display = hot_payload["styled_display"]
            display_df = hot_payload["display_df"]
            csv_bytes = hot_payload["csv_bytes"]
            excel_bytes = hot_payload["excel_bytes"]

            # panels
            for lbl in selected_metric_labels:
                panel = per_metric_panels.get(lbl) or []
                with st.expander(f"📌 {lbl} — recommendations & last results", expanded=st.session_state.get("hot_expand_open", False)):
                    if not panel:
                        st.write("_No hits for this metric (or no recommendation matched thresholds)._")
                    else:
                        st.markdown("\n".join(panel))

            # table
            if display_df.empty:
                st.info("No rows matched your selections/thresholds.")
            else:
                st.dataframe(styled_display, use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇️ Download Hot Sheet (CSV)",
                    data=csv_bytes,
                    file_name=hot_payload["csv_name"],
                    mime="text/csv",
                    key="hot_cached_csv"
                )
                st.download_button(
                    "⬇️ Download Hot Sheet (Excel)",
                    data=excel_bytes,
                    file_name=hot_payload["xlsx_name"],
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="hot_cached_xlsx"
                )
            st.stop()
        else:
            st.info("Configure thresholds/metrics and click **⚡ Generate / Refresh**.")
            st.stop()


    if hot_events_display.empty:
        st.info("No events match your Hot Sheets filters/search.")
        st.stop()

    # helpers
    def any_threshold_hit(p: float) -> bool:
        if p is None or pd.isna(p): return False
        checks = []
        if use_high:  checks.append(p >= hi2_val)
        if use_mhigh: checks.append(p >= hi1_val)
        if use_low:   checks.append(p <= lo2_val)
        if use_mlow:  checks.append(p <= lo1_val)
        return any(checks) if checks else True

    def recommend_direction(metric_key: str, pct_val: float):
        if pct_val is None or pd.isna(pct_val):
            return None, None
        def band_of(x):
            if use_high  and x >= hi2_val: return "HIGH"
            if use_mhigh and x >= hi1_val: return "MHIGH"
            if use_low   and x <= lo2_val: return "LOW"
            if use_mlow  and x <= lo1_val: return "MLOW"
            return None
        band = band_of(float(pct_val))
        if not band: return None, None
        if metric_key == "full_over_pct":
            th = float(st.session_state.get("over_total_threshold", DEFAULT_OVER_TOTAL_THRESHOLD))
            return (f"Play Full **Over {th:.1f}**", "FULL_OVER") if band in ("HIGH","MHIGH") else (f"Play Full **Under {th:.1f}**", "FULL_UNDER")
        if metric_key == "set1_over_18_5_pct":
            th = float(st.session_state.get("set1_over_threshold", DEFAULT_SET1_OVER_THRESHOLD))
            return (f"Play **Set 1 Over {th:.1f}**", "SET1_OVER") if band in ("HIGH","MHIGH") else (f"Play **Set 1 Under {th:.1f}**", "SET1_UNDER")
        if metric_key == "set1_under_18_5_pct":
            th = float(st.session_state.get("set1_over_threshold", DEFAULT_SET1_OVER_THRESHOLD))
            return (f"Play **Set 1 Under {th:.1f}**", "SET1_UNDER")
        if metric_key.startswith("set") and metric_key.endswith("_over_pct"):
            n = int(metric_key[3])
            th = float(st.session_state.get("set1_over_threshold", DEFAULT_SET1_OVER_THRESHOLD))
            return (f"Play **Set {n} Over {th:.1f}**", f"SET{n}_OVER") if band in ("HIGH","MHIGH") else (f"Play **Set {n} Under {th:.1f}**", f"SET{n}_UNDER")
        if metric_key in ("pct_3_0","pct_3_1","pct_3_2"):
            if band in ("HIGH","MHIGH"):
                final = metric_key.replace("pct_","").replace("_","-")
                return (f"Play **Exact {final}**", f"EXACT_{final}")
            return None, None
        if metric_key == "s1_winner_wins_match_pct":
            return ("Back **S1 winner to win the match**", "S1_WINS_MATCH") if band in ("HIGH","MHIGH") else (None, None)
        if metric_key == "s1_winner_opponent_s2_pct":
            return ("Back **S1 opponent to win Set 2**", "S1_OPP_WINS_S2") if band in ("HIGH","MHIGH") else (None, None)
        if metric_key == "s1_winner_s2_win_pct":
            return ("Back **S1 winner to win Set 2**", "S1_WINS_S2") if band in ("HIGH","MHIGH") else (None, None)
        if metric_key == "p1_overall_win_pct":
            return ("Back **Player 1 to win**", "P1_WIN") if band in ("HIGH","MHIGH") else (None, None)
        if metric_key == "p2_overall_win_pct":
            return ("Back **Player 2 to win**", "P2_WIN") if band in ("HIGH","MHIGH") else (None, None)
        return None, None

    

    # generate
    if not should_run:
        st.info("Configure thresholds/metrics and click **⚡ Generate / Refresh**.")
    else:
        with st.spinner("Building Hot Sheet across all matching events..."):
            rows = []
            per_metric_panels = {lbl: [] for lbl in selected_metric_labels}
            progress = st.progress(0.0, text="Computing per-event metrics...")
            totalN = len(hot_events_display)

            for i, (_, r) in enumerate(hot_events_display.iterrows(), start=1):
                eid = int(r["event_id"]) if str(r["event_id"]).isdigit() else r["event_id"]
                home_nm = r.get("home_name", "")
                away_nm = r.get("away_name", "")

                base = {
                    "event_id": str(r["event_id"]),
                    "match": f"{home_nm or '?'} vs {away_nm or '?'}",
                    "time": r.get("time"),
                    "league": r.get("league.name") or ("ALL" if league_id is None else league_name),
                }
                event_time = r.get("time")

                # Get home/away PIDs for this event (for PID-accurate stats)
                # Get home/away PIDs from the load_events call (r = row)
                p1_pid = r.get("home_id")  # HOME = Player 1
                p2_pid = r.get("away_id")  # AWAY = Player 2

                # We ONLY need to call get_event_overview if the match is "ended"
                # so we can check the final score for "Win/Loss".
                # For "upcoming" or "inplay", this API call is skipped.
                ov_for_pids = {}
                if up_or_end == "ended":
                    try:
                        # This call is now ONLY for outcome checking, not for PIDs.
                        ov_for_pids = get_event_overview(int(eid), sport_id, token)
                    except Exception:
                        ov_for_pids = {} # Continue with empty dict on error

                try:
                    (df_h2h_ev, agg_summary_ev, _pa, _piv, df_sets_ev, _totals_ev, met) = compute_event_package(
                    eid, timezone_local, sport_id, token,
                    history_qty,
                    float(st.session_state.get("over_total_threshold", DEFAULT_OVER_TOTAL_THRESHOLD)),
                    float(st.session_state.get("set1_over_threshold", DEFAULT_SET1_OVER_THRESHOLD)),
                    rate_limit_seconds,
                    event_cutoff_dt=event_time,   # <-- NEW
                    )

                    # H2H scoreline % helpers
                    def _score_pct_and_total(agg_df, key):
                        if agg_df is None or agg_df.empty:
                            return None, 0, 0
                        tot = int(agg_df.loc[agg_df["Score"]=="Total","Count"].iloc[0]) \
                            if "Total" in agg_df["Score"].values else int(agg_df["Count"].sum())
                        rowk = agg_df[agg_df["Score"]==key]
                        if rowk.empty:
                            return 0.0, 0, tot
                        cnt = int(rowk["Count"].iloc[0])
                        pct = float(rowk["Percent"].iloc[0]) if "Percent" in rowk.columns else (100.0*cnt/max(1,tot))
                        return pct, cnt, tot

                    p30,c30,totGames = _score_pct_and_total(agg_summary_ev, "3-0")
                    p31,c31,_        = _score_pct_and_total(agg_summary_ev, "3-1")
                    p32,c32,_        = _score_pct_and_total(agg_summary_ev, "3-2")
                    h2h_total_games  = totGames

                    # respect min H2H sample from the Main View selector
                    if h2h_total_games < min_h2h_required:
                        progress.progress(i/totalN, text=f"Computing per-event metrics... {i}/{totalN}")
                        continue

                    row = dict(base)
                    row.update({
                        "s1_winner_wins_match_pct":  met.get("s1_winner_wins_match_pct"),
                        "s1_winner_opponent_s2_pct": met.get("s1_winner_opponent_s2_pct"),
                        "s1_winner_s2_win_pct":      met.get("s1_winner_s2_win_pct"),
                        "pct_3_0": p30, "pct_3_1": p31, "pct_3_2": p32,
                        "H2H_n": h2h_total_games,
                        "set1_over_18_5_pct": met.get("set1_over_18_5_pct"),
                        "set1_under_18_5_pct": met.get("set1_under_18_5_pct"),
                        "full_over_pct":       met.get("over_75_5_prob_pct"),
                    })

                    # Per-set Over table
                    per_set = met.get("per_set_over_df", pd.DataFrame())
                    if not per_set.empty:
                        for s in range(1,6):
                            slot = per_set[per_set["set_no"]==s]
                            row[f"set{s}_over_pct"] = (float(slot["over_pct"].iloc[0])
                                                    if not slot.empty and pd.notna(slot["over_pct"].iloc[0]) else None)

                    # ===== P1/P2 mapping: P1=HOME, P2=AWAY =====
                    # NEW (PID-based):
                    row["P1_Name"] = home_nm or ""
                    row["P2_Name"] = away_nm or ""
                    row["P1_PID"]  = p1_pid or ""
                    row["P2_PID"]  = p2_pid or ""

                    def _overall_win_pct_by_pid(df_h2h_local: pd.DataFrame, pid: Optional[str]):
                        """
                        Overall win% for a player PID across the H2H pool:
                        wins / appearances, counting matches where (home_id==pid or away_id==pid),
                        and wins where match_winner_pid==pid.
                        """
                        if df_h2h_local.empty or not pid:
                            return None
                        pool = df_h2h_local[(df_h2h_local["home_id"] == pid) | (df_h2h_local["away_id"] == pid)]
                        if pool.empty:
                            return None
                        wins = int((pool["match_winner_pid"] == pid).sum())
                        den  = int(len(pool))
                        return (wins/den*100.0) if den else None


                    row["p1_overall_win_pct"] = _overall_win_pct_by_pid(df_h2h_ev, p1_pid)
                    row["p2_overall_win_pct"] = _overall_win_pct_by_pid(df_h2h_ev, p2_pid)

                    # Decide which side leads on overall win %
                    p1p = row.get("p1_overall_win_pct")
                    p2p = row.get("p2_overall_win_pct")
                    leader_side = None
                    try:
                        if p1p is not None and p2p is not None:
                            leader_side = "p1" if float(p1p) >= float(p2p) else "p2"
                        elif p1p is not None:
                            leader_side = "p1"
                        elif p2p is not None:
                            leader_side = "p2"
                    except Exception:
                        leader_side = None

                    # Odds snapshot (robust parser)
                    try:
                        odds_summary = get_odds_summary_cached(int(eid), sport_id, token)
                        stage = "end" if up_or_end == "inplay" else "start"
                        totals_odds, match_ml, set1_ml = parse_event_odds_summary(
                            odds_summary, home_nm, away_nm, preferred_stage=stage
                        )
                    except Exception:
                        totals_odds, match_ml, set1_ml = {}, {}, {}

                    row["OU_Line"]   = totals_odds.get("line")
                    row["Over_Dec"]  = totals_odds.get("over_dec")
                    row["Under_Dec"] = totals_odds.get("under_dec")
                    if show_american:
                        row["Over_Amer"]  = totals_odds.get("over_amer")
                        row["Under_Amer"] = totals_odds.get("under_amer")
                    if show_book:
                        row["Odds_Book"]  = totals_odds.get("book") or match_ml.get("book") or set1_ml.get("book")

                    row["P1_ML_Dec"] = match_ml.get("home_dec")
                    row["P2_ML_Dec"] = match_ml.get("away_dec")
                    if show_american:
                        row["P1_ML_Amer"] = match_ml.get("home_amer")
                        row["P2_ML_Amer"] = match_ml.get("away_amer")

                    row["S1_P1_ML_Dec"] = set1_ml.get("home_dec")
                    row["S1_P2_ML_Dec"] = set1_ml.get("away_dec")
                    if show_american:
                        row["S1_P1_ML_Amer"] = set1_ml.get("home_amer")
                        row["S1_P2_ML_Amer"] = set1_ml.get("away_amer")

                    # threshold gating
                    passed = (not selected_keys) or any(any_threshold_hit(row.get(k)) for k in selected_keys)
                    if not passed:
                        progress.progress(i/totalN, text=f"Computing per-event metrics... {i}/{totalN}")
                        continue

                    # quick outcome badge if finished/live partial
                    #ov = get_event_overview(int(eid), sport_id, token)

                    # ---------------------------
                    # Per-metric panel lines + collect ALL actionable recs, then EXPLODE rows
                    # ---------------------------
                    all_recs = []  # each rec ⇒ one output row

                    for lbl in selected_metric_labels:
                        k = label2key[lbl]
                        v = row.get(k)

                        # suppress P1/P2 recs for the non-leader side (keeps P1/P2 sensible)
                        if k == "p1_overall_win_pct" and leader_side == "p2":
                            rec, desired = (None, None)
                        elif k == "p2_overall_win_pct" and leader_side == "p1":
                            rec, desired = (None, None)
                        else:
                            rec, desired = recommend_direction(k, v)

                        # No recommendation for this metric
                        if not rec:
                            if not only_actionable:
                                pct_txt = f"{v:.1f}%" if v is not None and not pd.isna(v) else "—"
                                per_metric_panels[lbl].append(
                                    f"- **{row['match']}** (H2H {h2h_total_games} g) — _no rec_ ({pct_txt})"
                                )
                            continue

                        # units/pct (panel text only)
                        u = units_for_prob(v)
                        units_txt = (
                            f" · **{int(u)} unit{'s' if int(u) != 1 else ''}**" if (u and float(u).is_integer())
                            else (f" · **{u:.1f} units**" if u else "")
                        )
                        pct_txt = f"{v:.1f}%" if v is not None and not pd.isna(v) else "—"

                        # compute final value/result if ended; otherwise Pending
                        if up_or_end == "ended":
                            outcome, fval = evaluate_outcome_and_value(desired, ov_for_pids)
                            final_txt = f" · Final: **{fval}**" if fval is not None else ""
                            if outcome == "N/A":
                                per_metric_panels[lbl].append(
                                    f"- **{row['match']}** (H2H {h2h_total_games} g) — {rec} ({pct_txt}{units_txt}){final_txt} — ➖ N/A"
                                )
                            else:
                                badge = "✅ **Win**" if outcome == "Win" else "❌ **Loss**"
                                per_metric_panels[lbl].append(
                                    f"- **{row['match']}** (H2H {h2h_total_games} g) — {rec} ({pct_txt}{units_txt}){final_txt} → {badge}"
                                )
                            rec_result = outcome if outcome in ("Win","Loss") else "N/A"
                            final_value = fval
                        else:
                            per_metric_panels[lbl].append(
                                f"- **{row['match']}** (H2H {h2h_total_games} g) — {rec} ({pct_txt}{units_txt}) — _Pending_"
                            )
                            rec_result = "Pending"
                            final_value = None

                        # Collect this recommendation as ONE output row
                        all_recs.append({
                            "Rec_Metric":  lbl,
                            "Rec_Text":    rec,
                            "Rec_Key":     desired,
                            "Rec_ProbPct": v,
                            "Rec_Units":   u,
                            "Final_Value": final_value,
                            "Rec_Result":  rec_result,
                        })

                    # === EXPLODE: one row PER recommendation ===
                    if all_recs:
                        for rline in all_recs:
                            exploded = dict(row)   # base event columns (event_id, match, time, league, H2H_n, odds…)
                            exploded.update(rline) # add this single recommendation’s fields
                            rows.append(exploded)
                    else:
                        # Only add a single “no-rec” row when the user UNchecks "Only show actionable recs"
                        if not only_actionable:
                            rows.append(dict(row))

                except Exception as ex:
                    rows.append({**base, "error": str(ex)})

                progress.progress(i/totalN, text=f"Computing per-event metrics... {i}/{totalN}")

            hot_df = pd.DataFrame(rows)

            # panels
            for lbl in selected_metric_labels:
                panel = per_metric_panels.get(lbl) or []
                with st.expander(f"📌 {lbl} — recommendations & last results", expanded=False):
                    if not panel:
                        st.write("_No hits for this metric (or no recommendation matched thresholds)._")
                    else:
                        st.markdown("\n".join(panel))

            # table
            if hot_df.empty:
                st.info("No rows matched your selections/thresholds.")
            else:
                # Ensure Final Value is visible and nicely formatted
                id_cols  = ["event_id", "match", "time", "league", "H2H_n"]
                rec_cols = ["Rec_Metric", "Rec_Text", "Rec_ProbPct", "Rec_Units", "Final_Value", "Rec_Result"]

                odds_cols = ["OU_Line", "Over_Dec", "Under_Dec", "P1_ML_Dec", "P2_ML_Dec", "S1_P1_ML_Dec", "S1_P2_ML_Dec"]
                if show_american:
                    odds_cols += ["Over_Amer", "Under_Amer", "P1_ML_Amer", "P2_ML_Amer", "S1_P1_ML_Amer", "S1_P2_ML_Amer"]
                if show_book:
                    odds_cols += ["Odds_Book"]

                keep_cols = [c for c in (id_cols + rec_cols + selected_keys + odds_cols) if c in hot_df.columns]
                hot_df = hot_df[keep_cols].copy() if keep_cols else hot_df

                # ---------- Styling & formatting helpers ----------
                def _style_pct(col):
                    styles = []
                    for v in col:
                        if v is None or (isinstance(v, float) and pd.isna(v)):
                            styles.append("")
                            continue
                        try:
                            x = float(v)
                        except Exception:
                            styles.append("")
                            continue
                        if (use_high and x >= hi2_val) or (use_low and x <= lo2_val):
                            styles.append("color:#2ecc71; font-weight:bold;")
                        elif (use_mhigh and x >= hi1_val) or (use_mlow and x <= lo1_val):
                            styles.append("color:#9b59b6; font-weight:bold;")
                        else:
                            styles.append("")
                    return styles

                def _style_result(col):
                    return [color_style_for_result(v) for v in col]

                def mk_formatter(fmt_spec: str):
                    def _fmt(x):
                        if x is None: return ""
                        try:
                            if pd.isna(x): return ""
                        except Exception:
                            pass
                        try:
                            return format(float(x), fmt_spec)
                        except Exception:
                            return str(x)
                    return _fmt

                # Percent columns (include Rec_ProbPct)
                pct_cols = [c for c in hot_df.columns if c.endswith("_pct") or c.startswith("pct_") or c.endswith("_over_pct")]
                if "Rec_ProbPct" in hot_df.columns:
                    pct_cols.append("Rec_ProbPct")

                def pct_fmt(x):
                    try:
                        return "" if pd.isna(x) else f"{float(x):.1f}%"
                    except Exception:
                        return ""

                # Nice units formatting: show "1" if integer else "1.5"
                def units_fmt(u):
                    if u is None: return ""
                    try:
                        fu = float(u)
                        return f"{int(fu)}" if float(fu).is_integer() else f"{fu:.1f}"
                    except Exception:
                        return str(u)

                # Final value formatting:
                # - Numbers (totals) displayed as integer (e.g., 78)
                # - Strings (like "3-1" or "S1W:HOME, S2W:AWAY") passed through
                def final_fmt(v):
                    if v is None: return ""
                    try:
                        if isinstance(v, (int, np.integer)):
                            return str(int(v))
                        fv = float(v)
                        return f"{int(fv)}" if float(fv).is_integer() else f"{fv:.0f}"
                    except Exception:
                        return str(v)

                fmt_funcs = {}
                for c in pct_cols:
                    fmt_funcs[c] = pct_fmt

                # Odds & line formatters
                for oc in ["Over_Dec","Under_Dec","P1_ML_Dec","P2_ML_Dec","S1_P1_ML_Dec","S1_P2_ML_Dec"]:
                    if oc in hot_df.columns:
                        fmt_funcs[oc] = mk_formatter(".2f")
                if "OU_Line" in hot_df.columns:
                    fmt_funcs["OU_Line"] = mk_formatter(".1f")

                if "Rec_Units" in hot_df.columns:
                    fmt_funcs["Rec_Units"] = units_fmt
                if "Final_Value" in hot_df.columns:
                    fmt_funcs["Final_Value"] = final_fmt

                styled = (hot_df.style
                        .format(fmt_funcs)
                        .apply(_style_pct, subset=pct_cols))

                if "Rec_Result" in hot_df.columns:
                    styled = styled.apply(_style_result, subset=["Rec_Result"])

                # Friendlier column names for display
                display_df = hot_df.rename(columns={
                    "Rec_ProbPct": "Rec Prob %",
                    "Rec_Units": "Units",
                    "Final_Value": "Final Total/Score",
                    "Rec_Result": "Rec Result"
                })
                display_fmt_funcs = dict(fmt_funcs)
                if "Final Total/Score" in display_df.columns and "Final_Value" in fmt_funcs:
                    display_fmt_funcs["Final Total/Score"] = display_fmt_funcs.pop("Final_Value")
                if "Units" in display_df.columns and "Rec_Units" in display_fmt_funcs:
                    display_fmt_funcs["Units"] = display_fmt_funcs.pop("Rec_Units")
                if "Rec Prob %" in display_df.columns and "Rec_ProbPct" in display_fmt_funcs:
                    display_fmt_funcs["Rec Prob %"] = display_fmt_funcs.pop("Rec_ProbPct")

                display_pct_cols = [c if c != "Rec_ProbPct" else "Rec Prob %" for c in pct_cols]

                styled_display = (display_df.style
                                .format(display_fmt_funcs)
                                .apply(_style_pct, subset=[c for c in display_pct_cols if c in display_df.columns]))
                if "Rec Result" in display_df.columns:
                    styled_display = styled_display.apply(_style_result, subset=["Rec Result"])

                st.dataframe(styled_display, use_container_width=True, hide_index=True)

                # ---------- CSV export (use display_df so headers match what you see) ----------
                csv_bytes = display_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Hot Sheet (CSV)",
                    data=csv_bytes,
                    file_name=f"hot_sheet_{up_or_end}{'' if league_id else '_all'}_exploded.csv",
                    mime="text/csv"
                )

                # ---------- Excel export with simple inline formatting (xlsxwriter -> openpyxl fallback) ----------
                out_xlsx = io.BytesIO()
                export_df = display_df.copy()

                # Pre-format Units and Final columns as strings
                if "Units" in export_df.columns:
                    export_df["Units"] = export_df["Units"].apply(units_fmt)
                if "Final Total/Score" in export_df.columns:
                    export_df["Final Total/Score"] = export_df["Final Total/Score"].apply(final_fmt)

                engine = "xlsxwriter"
                try:
                    with pd.ExcelWriter(out_xlsx, engine=engine) as writer:
                        export_df.to_excel(writer, sheet_name="Hot Sheets (Exploded)", index=False)
                        wb  = writer.book
                        ws  = writer.sheets["Hot Sheets (Exploded)"]

                        fmt_green_text  = wb.add_format({"font_color": "#2ecc71", "bold": True})
                        fmt_purple_text = wb.add_format({"font_color": "#9b59b6", "bold": True})
                        fmt_win_fill    = wb.add_format({"bg_color": "#e6ffe6", "font_color": "#077307", "bold": True})
                        fmt_loss_fill   = wb.add_format({"bg_color": "#ffe6e6", "font_color": "#8a0000", "bold": True})

                        headers = list(export_df.columns)
                        nrows, ncols = export_df.shape

                        def col_idx(name):
                            try:
                                return headers.index(name)
                            except ValueError:
                                return None

                        pct_like_headers = headers[:]  # already display names
                        pct_cols_excel = [i for i,h in enumerate(pct_like_headers)
                                        if (h.endswith("_pct") or h.startswith("pct_") or h.endswith("_over_pct") or h == "Rec Prob %")]

                        for c in pct_cols_excel:
                            ws.conditional_format(1, c, nrows, c, {
                                "type": "cell", "criteria": ">=", "value": hi2_val, "format": fmt_green_text
                            })
                            ws.conditional_format(1, c, nrows, c, {
                                "type": "cell", "criteria": "<=", "value": lo2_val, "format": fmt_green_text
                            })
                            ws.conditional_format(1, c, nrows, c, {
                                "type": "cell", "criteria": "between", "minimum": hi1_val, "maximum": hi2_val - 1e-9, "format": fmt_purple_text
                            })
                            ws.conditional_format(1, c, nrows, c, {
                                "type": "cell", "criteria": "between", "minimum": lo2_val + 1e-9, "maximum": lo1_val, "format": fmt_purple_text
                            })

                        rec_res_col = col_idx("Rec Result")
                        if rec_res_col is not None:
                            ws.conditional_format(1, rec_res_col, nrows, rec_res_col, {
                                "type": "text", "criteria": "containing", "value": "Win", "format": fmt_win_fill
                            })
                            ws.conditional_format(1, rec_res_col, nrows, rec_res_col, {
                                "type": "text", "criteria": "containing", "value": "Loss", "format": fmt_loss_fill
                            })
                except Exception:
                    out_xlsx = io.BytesIO()
                    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
                        export_df.to_excel(writer, sheet_name="Hot Sheets (Exploded)", index=False)
                    # (openpyxl fallback without conditional formatting)

                st.download_button(
                    "⬇️ Download Hot Sheet (Excel)",
                    data=out_xlsx.getvalue(),
                    file_name=f"hot_sheet_{up_or_end}{'' if league_id else '_all'}_exploded.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )


                # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                # 3) SAVE WHAT YOU JUST RENDERED so a rerun (e.g., clicking Download) re-shows it
                # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                st.session_state["hot_expand_open"] = True  # optional: reopen expanders after rerun
                st.session_state["hot_last"] = {
                    "per_metric_panels": per_metric_panels,         # dict[label] -> list[str]
                    "display_df": display_df,                       # user-facing headers
                    "styled_display": styled_display,               # styled DataFrame you showed
                    "csv_bytes": csv_bytes,
                    "excel_bytes": out_xlsx.getvalue(),
                    "csv_name": f"hot_sheet_{up_or_end}{'' if league_id else '_all'}_exploded.csv",
                    "xlsx_name": f"hot_sheet_{up_or_end}{'' if league_id else '_all'}_exploded.xlsx",

                }
                # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<






# Footer
st.markdown("---")
st.caption("Tip: Turn on the date filter in the sidebar to pull ALL events for that calendar day (local TZ).")

# Auto-refresh for live
if st.session_state.get("mode_radio") == "inplay":
    do_live_autorefresh(enabled=True, interval_ms=15000, key="live_refresh")
