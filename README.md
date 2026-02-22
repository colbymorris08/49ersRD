# 49ers R&D — NFL Injury Risk

NFL next-week injury/availability risk from snap-count workload (acute:chronic ratio, deltas). Data: [nflverse](https://nflverse.r-universe.dev/) (2018–2024).

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Or run the CLI model only:

```bash
python build_injury_risk.py
```

## Deploy Streamlit app (public)

1. Push this repo to GitHub (e.g. `YOUR_USERNAME/49ersRD`).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. **New app** → choose repo `49ersRD`, branch `main`, main file path `app.py`.
4. Click **Deploy**. First run will install deps and build the model (may take a few minutes).

No secrets or env vars required; the app loads nflverse data at runtime.
