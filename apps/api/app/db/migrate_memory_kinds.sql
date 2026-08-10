-- Add text/voice memory kinds (safe to re-run)
ALTER TABLE memories ADD COLUMN IF NOT EXISTS kind TEXT;
ALTER TABLE memories ADD COLUMN IF NOT EXISTS text_content TEXT;

UPDATE memories SET kind = 'voice' WHERE kind IS NULL;
ALTER TABLE memories ALTER COLUMN kind SET DEFAULT 'voice';
ALTER TABLE memories ALTER COLUMN kind SET NOT NULL;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'memories_kind_check'
  ) THEN
    ALTER TABLE memories
      ADD CONSTRAINT memories_kind_check CHECK (kind IN ('text', 'voice'));
  END IF;
END $$;

ALTER TABLE memories ALTER COLUMN audio_uri DROP NOT NULL;