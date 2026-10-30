# Privacy

FaceAttend stores biometric data (face embeddings). Treat as sensitive.

## Data inventory
- persons: name, employee_id, consent metadata
- embeddings: 512-d ArcFace vectors (biometric)
- attendance: timestamps, camera, confidence, optional snapshot path
- alerts: snapshots for unknown persons
- audit_log: actions

## Rights
- Access: `GET /persons/{id}/export`
- Erasure: `DELETE /persons/{id}` (removes embeddings, anonymizes attendance, deletes snapshots, audit entry)
- Rectification: re-enroll

## Retention
- Snapshots: RETENTION_DAYS (default 30)
- Attendance anonymized after ANONYMIZE_DAYS (default 90)
- Audit log: 1 year

## Controls
- Consent required at enrollment
- No raw images stored by default (only on unknown alerts)
- PII redacted in logs (IDs only)
- Webhook opt-in

## Lawful basis
Explicit consent. Consult counsel before production deployment (GDPR/BIPA/CCPA may apply).