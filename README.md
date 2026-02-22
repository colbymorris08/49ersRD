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

## Push to GitHub (repo: 49ersRD)

1. **Create the repo on GitHub:** [github.com/new](https://github.com/new) → name it `49ersRD`, leave it empty (no README/license).
2. **Set remote and push:**

   ```bash
   cd /Users/colbymorris/nflinjury
   git remote set-url origin https://github.com/colbymorris08/49ersRD.git
   git push -u origin main
   ```

## Deploy Streamlit app (public)

1. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
2. **New app** → choose repo `49ersRD`, branch `main`, main file path `app.py`.
3. Click **Deploy**. First run will install deps and build the model (may take a few minutes).

No secrets or env vars required; the app loads nflverse data at runtime.
