-- Persist settled winner on review_outcomes for analyst challenge tracking.
-- Safe to run multiple times.
--
-- Existing columns already cover:
--   agree_with_model (prediction_reviews) → analyst_disagreed
--   pregame_notes / missing_factors → analyst_reasoning
--   reviewer_correct → analyst_was_correct
--   model_correct → ai_was_correct
--   reviewer_beat_model → successful_analyst_override

ALTER TABLE review_outcomes ADD COLUMN IF NOT EXISTS final_result TEXT;
