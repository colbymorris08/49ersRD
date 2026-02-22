"""
Streamlit app: NFL injury risk by team and player.
Uses workload (snap count) model from build_injury_risk.
"""

import streamlit as st
import pandas as pd

# Page config: simple and clean
st.set_page_config(
    page_title="NFL Injury Risk",
    page_icon="🏈",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Hide streamlit chrome for cleaner look
st.markdown("""
<style>
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 720px; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def load_risk_data():
    """Load latest risk table (cached 1 hr)."""
    from build_injury_risk import get_latest_risk_table
    latest, season, week = get_latest_risk_table(debug=False)
    if latest is None:
        return None, None, None
    return latest, season, week


def main():
    st.title("NFL injury risk by workload")
    st.caption("Next-week limited/out risk from snap-count and workload features (nflverse data).")

    st.markdown(
        "**Data & project:** This app uses NFL snap counts and injury reports (2018–2024) from "
        "nflverse. A simple logistic model predicts next-week limited/questionable/out status from "
        "current-week snaps and acute:chronic workload ratios."
    )
    with st.expander("How AC ratio & risk are calculated"):
        st.markdown(
            "**AC ratio (4-game):** This week's snaps ÷ rolling average of the *previous* 4 weeks' snaps. "
            "So it's *acute workload / chronic workload*. Values &gt;1 mean more snaps than recent baseline; "
            "&lt;1 means less. If this week's snaps are 0, the ratio is 0 (or undefined if they had no prior snaps)."
        )
        st.markdown(
            "**Risk:** The model uses 6 features: snaps, AC ratio (4g and 6g), and *snap deltas* "
            "(this week minus recent average). So risk isn't only from high workload—it also reacts to **sudden drops**. "
            "A player with **0 snaps and 0 AC ratio but high risk** (e.g. Stone Forsythe) is there because their "
            "*deltas* are large negative: they usually had snaps in prior weeks and now have zero. In the training data, "
            "that pattern often goes with being limited or out next week (already injured or ramping down). So the model "
            "flags them as higher risk even though AC ratio is 0."
        )
    st.divider()

    with st.spinner("Loading model and data…"):
        latest, season, week = load_risk_data()

    if latest is None:
        st.error("Could not build the risk model. Check the console or run `python build_injury_risk.py`.")
        return

    if "team" not in latest.columns:
        latest["team"] = "—"

    teams = sorted(latest["team"].dropna().astype(str).unique())
    if not teams:
        st.warning("No team information in the data.")
        team_options = ["All"]
    else:
        team_options = ["All"] + teams

    team_choice = st.selectbox(
        "Select team",
        options=team_options,
        index=0,
        key="team",
    )

    if team_choice == "All":
        subset = latest.copy()
    else:
        subset = latest[latest["team"].astype(str) == team_choice].copy()
    subset = subset.sort_values("risk", ascending=False).reset_index(drop=True)

    display_cols = ["full_name", "position", "snaps", "acr_4g", "risk"]
    display_cols = [c for c in display_cols if c in subset.columns]
    out = subset[display_cols].copy()
    if "risk" in out.columns:
        out["risk"] = out["risk"].round(3)
    if "snaps" in out.columns:
        out["snaps"] = out["snaps"].astype(int)
    if "acr_4g" in out.columns:
        out["acr_4g"] = out["acr_4g"].round(2)

    st.subheader(f"{'All teams' if team_choice == 'All' else team_choice} — Season {season}, Week {week}")
    st.dataframe(
        out,
        use_container_width=True,
        hide_index=True,
        column_config={
            "full_name": "Player",
            "position": "Pos",
            "snaps": "Snaps",
            "acr_4g": st.column_config.NumberColumn("AC ratio", format="%.2f", help="Acute:chronic workload (4-game)"),
            "risk": st.column_config.NumberColumn("Risk", format="%.2f"),
        },
    )
    st.caption(f"{len(out)} players · Risk = predicted probability of being limited/out next week.")


if __name__ == "__main__":
    main()
