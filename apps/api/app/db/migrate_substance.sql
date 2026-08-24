-- Substance layer: transcripts, chunk metadata, knowledge facts (safe to re-run)

ALTER TABLE memories ADD COLUMN IF NOT EXISTS raw_transcript TEXT;
ALTER TABLE memories ADD COLUMN IF NOT EXISTS cleaned_transcript TEXT;

ALTER TABLE chunks ADD COLUMN IF NOT EXISTS meta JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS salience REAL NOT NULL DEFAULT 0.5;

CREATE TABLE IF NOT EXISTS knowledge_facts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    statement TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'attribute'
        CHECK (category IN (
            'preference', 'relationship', 'belief', 'attribute', 'event_summary'
        )),
    people TEXT[] NOT NULL DEFAULT '{}',
    places TEXT[] NOT NULL DEFAULT '{}',
    topics TEXT[] NOT NULL DEFAULT '{}',
    confidence REAL NOT NULL DEFAULT 0.5,
    evidence_count INTEGER NOT NULL DEFAULT 1,
    supporting_chunk_ids UUID[] NOT NULL DEFAULT '{}',
    supporting_memory_ids UUID[] NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'candidate'
        CHECK (status IN ('candidate', 'established', 'disputed')),
    salience REAL NOT NULL DEFAULT 0.5,
    conflict_note TEXT,
    embedding VECTOR(1024),
    tsv TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(statement, ''))
    ) STORED,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS knowledge_facts_agent_id_idx ON knowledge_facts(agent_id);
CREATE INDEX IF NOT EXISTS knowledge_facts_tsv_idx ON knowledge_facts USING GIN (tsv);
CREATE INDEX IF NOT EXISTS knowledge_facts_embedding_idx ON knowledge_facts
    USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_meta_gin_idx ON chunks USING GIN (meta);

ALTER TABLE knowledge_facts ENABLE ROW LEVEL SECURITY;
ALTER TABLE knowledge_facts FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS knowledge_facts_all ON knowledge_facts;
CREATE POLICY knowledge_facts_all ON knowledge_facts
    FOR ALL USING (
        COALESCE(current_setting('app.current_agent_id', true), '') = ''
        OR agent_id::text = current_setting('app.current_agent_id', true)
    )
    WITH CHECK (
        COALESCE(current_setting('app.current_agent_id', true), '') = ''
        OR agent_id::text = current_setting('app.current_agent_id', true)
    );
