-- One human inbox → one reviewer row.
-- Partial unique index keeps NULL emails allowed for local/demo seeds.

CREATE UNIQUE INDEX IF NOT EXISTS unique_reviewer_email
ON reviewers (lower(email))
WHERE email IS NOT NULL AND btrim(email) <> '';
