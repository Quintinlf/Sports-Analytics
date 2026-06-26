# Railway External Tester Deployment

## 1. Railway project setup

1. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo**
2. Select this repository and deploy branch
3. **+ New** → **Database** → **PostgreSQL**
4. Web service → **Variables** → **Add Reference** → link PostgreSQL `DATABASE_URL`
5. **Settings** → **Networking** → **Generate Domain**
6. Set web service variables:

| Variable | Value |
|---|---|
| `DATABASE_URL` | Reference from PostgreSQL plugin |
| `FEEDBACK_BASE_URL` | `https://<your-domain>.up.railway.app` |
| `SMTP_HOST` | `smtp.gmail.com` |
| `SMTP_PORT` | `587` |
| `SMTP_USER` | Your Gmail address |
| `SMTP_PASS` | Gmail App Password |
| `FEEDBACK_EMAIL_FROM` | Same as `SMTP_USER` |

7. **Settings** → **Deploy**:
   - Start command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
   - Install command: `pip install -r requirements-web.txt`

## 2. Migrate SQLite data to PostgreSQL

Run once from your local machine (with Railway Postgres **public** URL):

```powershell
$env:DATABASE_URL="postgresql://..."   # Railway public DATABASE_URL
$env:SQLITE_DATABASE_URL="sqlite:///./sports_analytics.db"
python scripts/migrate_sqlite_to_postgres.py
```

## 3. GitHub Secrets (Settings → Secrets → Actions)

| Secret | Value |
|---|---|
| `DATABASE_URL` | Railway PostgreSQL **public** URL |
| `FEEDBACK_BASE_URL` | `https://<your-domain>.up.railway.app` |
| `SMTP_HOST` | `smtp.gmail.com` |
| `SMTP_PORT` | `587` |
| `SMTP_USER` | Gmail address |
| `SMTP_PASS` | Gmail App Password |
| `FEEDBACK_EMAIL_FROM` | Same as SMTP_USER |

## 4. Validate deployment

```powershell
$env:FEEDBACK_BASE_URL="https://<your-domain>.up.railway.app"
python scripts/validate_deployment.py
```

## 5. Send test email to Quintin

GitHub → **Actions** → **Weekly Feedback Test (Quintin Only)** → **Run workflow**

Email link format:
`https://<domain>/feedback?reviewer_id=quintin&sport=MLB`

## 6. Production workflows

| Workflow | Schedule | Purpose |
|---|---|---|
| `daily-predictions.yml` | `0 0 * * *` UTC | Ingest MLB/NBA/FIFA predictions |
| `weekly-feedback.yml` | `0 0 * * 0` UTC | Weekly digest to all reviewers |
| `weekly-feedback-test.yml` | Manual only | Test email to `quintinlf7@gmail.com` |

## Gmail App Password

Google Account → Security → 2-Step Verification → App Passwords → create for "Sports Analytics"
