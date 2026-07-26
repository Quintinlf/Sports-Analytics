# Render Deployment

## 1. Render project setup

1. Go to [render.com](https://render.com) → **New** → **Blueprint** (or **Web Service** + **PostgreSQL**)
2. Connect this GitHub repository and select the deploy branch
3. Create a **PostgreSQL** database (Render Dashboard → **New** → **PostgreSQL**)
4. Create a **Web Service** from the same repo (or use `render.yaml` Blueprint)
5. Link the web service to the Postgres instance via `DATABASE_URL`
6. Set the public URL in `FEEDBACK_BASE_URL` (no trailing slash)

### Web service settings

| Setting | Value |
|---|---|
| **Runtime** | Python 3 |
| **Build Command** | `pip install -r requirements-web.txt` |
| **Pre-Deploy Command** | `python -m scripts.init_database` |
| **Start Command** | `uvicorn backend.main:app --host 0.0.0.0 --port $PORT` |
| **Health Check Path** | `/feedback` |

### Environment variables

| Variable | Value |
|---|---|
| `DATABASE_URL` | Internal connection string from Render PostgreSQL (or external URL for GitHub Actions) |
| `SCHEMA_AUTO_MIGRATE` | `false` on production web (default for Postgres). Schema is applied by the pre-deploy command, not on every boot. |
| `FEEDBACK_BASE_URL` | `https://<your-service>.onrender.com` |
| `SMTP_HOST` | `smtp.gmail.com` |
| `SMTP_PORT` | `587` |
| `SMTP_USER` | Your Gmail address |
| `SMTP_PASS` | Gmail App Password |
| `FEEDBACK_EMAIL_FROM` | Same as `SMTP_USER` |

> **Note:** Render Postgres provides `postgres://` URLs. The app normalizes these to `postgresql+psycopg2://` automatically in `scripts/db_utils.py`.

## 2. Migrate SQLite data to PostgreSQL

Run once from your local machine (use the Render Postgres **external** connection string for GitHub Actions; internal URL works from the Render shell):

```powershell
$env:DATABASE_URL="postgresql://..."   # Render PostgreSQL URL
$env:SQLITE_DATABASE_URL="sqlite:///./sports_analytics.db"
python scripts/migrate_sqlite_to_postgres.py
```

## 3. GitHub Secrets (Settings → Secrets → Actions)

| Secret | Value |
|---|---|
| `DATABASE_URL` | Render PostgreSQL **external** URL (reachable from GitHub Actions runners) |
| `FEEDBACK_BASE_URL` | `https://<your-service>.onrender.com` |
| `SMTP_HOST` | `smtp.gmail.com` |
| `SMTP_PORT` | `587` |
| `SMTP_USER` | Gmail address |
| `SMTP_PASS` | Gmail App Password |
| `FEEDBACK_EMAIL_FROM` | Same as SMTP_USER |

## 4. Validate deployment

```powershell
$env:FEEDBACK_BASE_URL="https://<your-service>.onrender.com"
python scripts/validate_deployment.py
```

## 5. Send test email to Quintin

GitHub → **Actions** → **Weekly Feedback Test (Quintin Only)** → **Run workflow**

Email link format:
`https://<your-service>.onrender.com/feedback?reviewer_id=quintin&sport=MLB`

## 6. Production workflows

| Workflow | Schedule | Purpose |
|---|---|---|
| `daily-predictions.yml` | `0 0 * * *` UTC | Ingest MLB/NBA/FIFA predictions |
| `weekly-feedback.yml` | `0 0 * * 0` UTC | Weekly digest to all reviewers |
| `weekly-feedback-test.yml` | Manual only | Test email to `quintinlf7@gmail.com` |

## Gmail App Password

Google Account → Security → 2-Step Verification → App Passwords → create for "Sports Analytics"
