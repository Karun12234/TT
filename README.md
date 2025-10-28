# Streamlit App (TT Dashboard)

This repo hosts a Streamlit app and is ready for deployment on Streamlit Community Cloud.

## Files
- `tt_dashboard_v24.py` — the app entrypoint (or rename to `streamlit_app.py`).
- `requirements.txt` — Python dependencies.
- `runtime.txt` — Python version pin for Streamlit Cloud.
- `.gitignore` — avoids committing secrets and venvs.
- `.streamlit/secrets.example.toml` — template for local development. **Do not commit real secrets.**

## Local run
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run tt_dashboard_v24.py
```

For local secrets, copy the template and fill in your token:
```bash
mkdir -p .streamlit
cp .streamlit/secrets.example.toml .streamlit/secrets.toml
# edit .streamlit/secrets.toml and set BETSAPI_TOKEN
```

## Deploy to Streamlit Community Cloud
1. Push this folder to a new GitHub repo.
2. Visit https://share.streamlit.io/ and click **New app**.
3. Choose your repo, branch, and **app file path** (`tt_dashboard_v24.py` or `streamlit_app.py`).
4. Click **Advanced settings → Secrets** and add:
   ```toml
   BETSAPI_TOKEN = "your-real-token"
   APP_ENV = "cloud"
   ```
5. Deploy. The app will build using `requirements.txt` and `runtime.txt`.
