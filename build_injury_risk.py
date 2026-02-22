# build_injury_risk.py
# ---------------------------------------------------------
# NFL workload (snap-count) -> next-week limited/injury proxy
# Data: nflverse via nflreadpy
# ---------------------------------------------------------

import os
import subprocess
import sys

def _install_deps():
    """Install requirements if missing, then re-run this script."""
    try:
        import numpy
        import pandas
        import sklearn
        import nflreadpy
        return
    except ImportError:
        pass
    req_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    if os.path.isfile(req_file):
        print("Installing dependencies from requirements.txt...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
        print("Restarting script...")
        os.execv(sys.executable, [sys.executable] + sys.argv)
    sys.exit(1)

_install_deps()

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression


def pick_player_id_col(df: pd.DataFrame) -> str:
    for c in ["player_id", "gsis_id", "pfr_player_id", "nfl_player_id"]:
        if c in df.columns:
            return c
    raise ValueError(f"No recognizable player id column found. Columns: {df.columns.tolist()[:80]}")


def load_data(seasons):
    import nflreadpy as nfl
    snaps = nfl.load_snap_counts(seasons).to_pandas()
    injuries = nfl.load_injuries(seasons).to_pandas()
    players = nfl.load_players().to_pandas()
    return snaps, injuries, players


def build_weekly_snaps(snaps: pd.DataFrame) -> pd.DataFrame:
    pid = pick_player_id_col(snaps)

    snap_candidates = [
        "offense_snaps", "defense_snaps", "st_snaps",
        "snaps", "total_snaps", "snap_counts"
    ]
    snap_col = next((c for c in snap_candidates if c in snaps.columns), None)

    if snap_col is None:
        cols = snaps.columns.tolist()
        off = "offense_snaps" if "offense_snaps" in cols else None
        de = "defense_snaps" if "defense_snaps" in cols else None
        if off or de:
            snaps["_snaps_total"] = (snaps[off] if off else 0) + (snaps[de] if de else 0)
            snap_col = "_snaps_total"
        else:
            raise ValueError(f"Could not find snaps column. Columns: {snaps.columns.tolist()[:120]}")

    needed = [pid, "season", "week", snap_col]
    missing = [c for c in needed if c not in snaps.columns]
    if missing:
        raise ValueError(f"Missing required columns in snaps table: {missing}")

    # Optional: restrict to REG; include team for UI
    use_cols = needed.copy()
    if "team" in snaps.columns:
        use_cols.append("team")
    for gcol in ["game_type", "season_type"]:
        if gcol in snaps.columns:
            use_cols.append(gcol)
            break
    df = snaps[use_cols].copy()
    for gcol in ["game_type", "season_type"]:
        if gcol in df.columns:
            df = df[df[gcol].astype(str).str.upper().eq("REG")].drop(columns=[gcol], errors="ignore")
            break

    df = df.rename(columns={pid: "player_id", snap_col: "snaps"})
    df["player_id"] = df["player_id"].astype(str)
    df["season"] = pd.to_numeric(df["season"], errors="coerce").dropna().astype(int)
    df["week"] = pd.to_numeric(df["week"], errors="coerce").dropna().astype(int)
    df["snaps"] = pd.to_numeric(df["snaps"], errors="coerce").fillna(0.0)
    if "team" in df.columns:
        df = df.groupby(["player_id", "season", "week"], as_index=False).agg(
            snaps=("snaps", "sum"), team=("team", "first")
        )
    else:
        df = df.groupby(["player_id", "season", "week"], as_index=False)["snaps"].sum()
    return df


def _find_texty_cols(df: pd.DataFrame) -> list:
    """Heuristic: columns likely to contain injury/practice info."""
    keys = ["status", "practice", "report", "injury", "game"]
    cols = []
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in keys):
            cols.append(c)
    return cols


def build_injury_label(injuries: pd.DataFrame, debug: bool = True) -> pd.DataFrame:
    """
    Create a weekly 'availability limitation' proxy.
    We aim to flag weeks where the player is limited/out/questionable/IR/etc.
    """
    pid = pick_player_id_col(injuries)
    required = [pid, "season", "week"]
    for c in required:
        if c not in injuries.columns:
            raise ValueError(f"Injuries table missing {c}. Columns: {injuries.columns.tolist()[:120]}")

    df = injuries.copy().rename(columns={pid: "player_id"})
    df["player_id"] = df["player_id"].astype(str)
    df["season"] = pd.to_numeric(df["season"], errors="coerce").dropna().astype(int)
    df["week"] = pd.to_numeric(df["week"], errors="coerce").dropna().astype(int)
    for col in ["game_type", "season_type"]:
        if col in df.columns:
            df = df[df[col].astype(str).str.upper().eq("REG")]
            break

    status_cols = _find_texty_cols(df)
    if debug:
        print("\n[DEBUG] injuries columns (first 40):")
        print(df.columns.tolist()[:40])
        print(f"[DEBUG] candidate status/practice columns found: {status_cols[:25]}{'...' if len(status_cols) > 25 else ''}")

    if not status_cols:
        status_cols = [c for c in df.columns if df[c].dtype == "object"]
        if debug:
            print(f"[DEBUG] fallback to object columns, count={len(status_cols)}")

    tmp = df[status_cols].copy()
    tmp = tmp.fillna("").astype(str)
    blob = tmp.agg(" | ".join, axis=1).str.lower()

    tokens = [
        "out", "doubtful", "questionable",
        "did not practice", "dnp",
        "limited", "lp",
        "injured reserve", "ir",
        "inactive"
    ]
    df["inj_limited"] = blob.apply(lambda s: int(any(t in s for t in tokens)))

    out = df.groupby(["player_id", "season", "week"], as_index=False)["inj_limited"].max()

    if debug:
        vc = out["inj_limited"].value_counts(dropna=False).to_dict()
        print(f"[DEBUG] inj_limited value counts: {vc}")
        if vc.get(1, 0) == 0:
            print("[DEBUG] No positives detected.")

    return out


def add_workload_features(weekly: pd.DataFrame) -> pd.DataFrame:
    weekly = weekly.sort_values(["player_id", "season", "week"]).copy()

    weekly["acute_1g"] = weekly["snaps"]

    weekly["chronic_4g"] = (
        weekly.groupby(["player_id", "season"])["snaps"]
        .apply(lambda s: s.shift(1).rolling(4, min_periods=2).mean())
        .reset_index(level=[0, 1], drop=True)
    )
    weekly["chronic_6g"] = (
        weekly.groupby(["player_id", "season"])["snaps"]
        .apply(lambda s: s.shift(1).rolling(6, min_periods=3).mean())
        .reset_index(level=[0, 1], drop=True)
    )

    weekly["acr_4g"] = weekly["acute_1g"] / weekly["chronic_4g"].replace(0, np.nan)
    weekly["acr_6g"] = weekly["acute_1g"] / weekly["chronic_6g"].replace(0, np.nan)

    weekly["snap_delta_4g"] = weekly["acute_1g"] - weekly["chronic_4g"]
    weekly["snap_delta_6g"] = weekly["acute_1g"] - weekly["chronic_6g"]

    prior3 = (
        weekly.groupby(["player_id", "season"])["snaps"]
        .apply(lambda s: s.shift(1).rolling(3, min_periods=2).mean())
        .reset_index(level=[0, 1], drop=True)
    )
    weekly["delta_vs_prior3"] = weekly["acute_1g"] - prior3

    return weekly


def attach_player_metadata(df: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    if players is None or players.empty:
        return df

    pid = pick_player_id_col(players)
    p = players.copy().rename(columns={pid: "player_id"})
    p["player_id"] = p["player_id"].astype(str)

    keep = [c for c in ["player_id", "position", "full_name", "first_name", "last_name"] if c in p.columns]
    p = p[keep].drop_duplicates("player_id")

    if "full_name" not in p.columns:
        fn = p["first_name"] if "first_name" in p.columns else ""
        ln = p["last_name"] if "last_name" in p.columns else ""
        p["full_name"] = (fn.astype(str) + " " + ln.astype(str)).str.strip()

    return df.merge(p, on="player_id", how="left")


def get_latest_risk_table(seasons=None, debug=False):
    """
    Build model and return latest week risk table with team, player, position, snaps, risk.
    Returns (dataframe, season, week) or (None, None, None) if training fails.
    """
    if seasons is None:
        seasons = list(range(2018, 2025))
    snaps, injuries, players = load_data(seasons)

    weekly_snaps = build_weekly_snaps(snaps)
    injury_week = build_injury_label(injuries, debug=debug)

    # Snap counts use pfr_player_id; injuries use gsis_id. Map via players table.
    pid_snaps = pick_player_id_col(snaps)
    if pid_snaps == "pfr_player_id" and "pfr_id" in players.columns and "gsis_id" in players.columns:
        player_ids = players[["pfr_id", "gsis_id"]].dropna(subset=["gsis_id", "pfr_id"]).drop_duplicates("pfr_id")
        player_ids["pfr_id"] = player_ids["pfr_id"].astype(str)
        weekly_snaps = weekly_snaps.merge(player_ids, left_on="player_id", right_on="pfr_id", how="left")
        weekly_snaps = weekly_snaps.dropna(subset=["gsis_id"]).drop(columns=["pfr_id"], errors="ignore")
        weekly_snaps["gsis_id"] = weekly_snaps["gsis_id"].astype(str)
    elif "gsis_id" in weekly_snaps.columns:
        pass
    else:
        weekly_snaps["gsis_id"] = weekly_snaps["player_id"].astype(str)

    injury_week = injury_week.copy()
    injury_week["season"] = injury_week["season"].astype(int)
    injury_week["week"] = injury_week["week"].astype(int)
    weekly_snaps = weekly_snaps.copy()
    weekly_snaps["season"] = weekly_snaps["season"].astype(int)
    weekly_snaps["week"] = weekly_snaps["week"].astype(int)

    df = weekly_snaps.merge(
        injury_week,
        left_on=["gsis_id", "season", "week"],
        right_on=["player_id", "season", "week"],
        how="left",
    )
    for c in ["player_id_x", "player_id_y", "player_id"]:
        df = df.drop(columns=[c], errors="ignore")
    df["player_id"] = df["gsis_id"].astype(str)
    df["inj_limited"] = df["inj_limited"].fillna(0).astype(int)

    if debug:
        print(f"[DEBUG] after merge, inj_limited value counts: {df['inj_limited'].value_counts().to_dict()}")

    df = df.sort_values(["player_id", "season", "week"]).copy()
    df["inj_next_week"] = df.groupby(["player_id", "season"])["inj_limited"].shift(-1)

    df = add_workload_features(df)
    df = attach_player_metadata(df, players)

    model_df = df.dropna(subset=["inj_next_week"]).copy()
    model_df["inj_next_week"] = model_df["inj_next_week"].astype(int)

    dist = model_df["inj_next_week"].value_counts().to_dict()
    if debug:
        print(f"\n[DEBUG] inj_next_week distribution: {dist}")
    if len(dist) < 2:
        if debug:
            print("\nERROR: inj_next_week has only one class. Model cannot train.")
        return None, None, None

    feature_cols = ["snaps", "acr_4g", "acr_6g", "snap_delta_4g", "snap_delta_6g", "delta_vs_prior3"]
    X = model_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = model_df["inj_next_week"]

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    use_stratify = n_pos >= 2 and n_neg >= 2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if use_stratify else None
    )

    clf = LogisticRegression(max_iter=3000)
    clf.fit(X_train, y_train)

    pred = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, pred)
    if debug:
        print(f"\nBaseline Logistic Regression AUC: {auc:.3f}")
        print("\nFeature weights:")
        for name, w in sorted(zip(feature_cols, clf.coef_[0]), key=lambda t: abs(t[1]), reverse=True):
            print(f"  {name:>15s}: {w:+.4f}")

    latest_season = int(model_df["season"].max())
    latest_week = int(model_df.loc[model_df["season"] == latest_season, "week"].max())
    latest = model_df[(model_df["season"] == latest_season) & (model_df["week"] == latest_week)].copy()

    latest_X = latest[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    latest["risk"] = clf.predict_proba(latest_X)[:, 1]

    if debug:
        show_cols = ["season", "week", "player_id", "team", "position", "full_name",
                     "snaps", "acr_4g", "snap_delta_4g", "risk"]
        show_cols = [c for c in show_cols if c in latest.columns]
        top = latest.sort_values("risk", ascending=False).head(20)[show_cols]
        print(f"\nTop 20 risk flags (season {latest_season}, week {latest_week}):")
        print(top.to_string(index=False))
    return latest, latest_season, latest_week


def main():
    latest, season, week = get_latest_risk_table(debug=True)
    if latest is None:
        return
    print(f"\nRisk table ready: season {season}, week {week} ({len(latest)} players).")


if __name__ == "__main__":
    main()
