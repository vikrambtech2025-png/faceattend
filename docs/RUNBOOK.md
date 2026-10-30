# Runbook

## Deploy
`docker compose up -d --build`

## Rollback
Redeploy previous image tag.

## Restore DB
`sqlite3 data/attendance.db ".restore backup.db"`

## Handle camera offline
Check /health camera_ok; restart container; verify device permissions.

## Purge data
`POST /admin/purge` or run retention job.