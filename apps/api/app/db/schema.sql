CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- App settings for RLS
-- SET LOCAL app.current_agent_id = '<uuid>';

CREATE TABLE IF NOT EXISTS agents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    display_name TEXT NOT NULL,
    share_token_hash TEXT NOT NULL,
    elevenlabs_voice_id TEXT,
    status TEXT NOT NULL DEFAULT 'ready'
        CHECK (status IN ('ready', 'cloning', 'error')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    kind TEXT NOT NULL DEFAULT 'voice'
        CHECK (kind IN ('text', 'voice')),
    text_content TEXT,
    audio_uri TEXT,
    duration_ms INTEGER,
    assemblyai_transcript_id TEXT,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'transcribing', 'embedding', 'indexed', 'error')),
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS memories_agent_id_idx ON memories(agent_id);

CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    memory_id UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    speaker TEXT,
    ts_start DOUBLE PRECISION,
    ts_end DOUBLE PRECISION,
    tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', coalesce(text, ''))) STORED,
    embedding VECTOR(1024),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chunks_agent_id_idx ON chunks(agent_id);
CREATE INDEX IF NOT EXISTS chunks_memory_id_idx ON chunks(memory_id);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING GIN (tsv);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks
    USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS chat_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    citations JSONB,
    audio_uri TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS messages_session_id_idx ON messages(session_id);
CREATE INDEX IF NOT EXISTS messages_agent_id_idx ON messages(agent_id);

-- RLS (FORCE so table owner is also constrained when app.current_agent_id is set)
ALTER TABLE agents ENABLE ROW LEVEL SECURITY;
ALTER TABLE agents FORCE ROW LEVEL SECURITY;
ALTER TABLE memories ENABLE ROW LEVEL SECURITY;
ALTER TABLE memories FORCE ROW LEVEL SECURITY;
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks FORCE ROW LEVEL SECURITY;
ALTER TABLE chat_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_sessions FORCE ROW LEVEL SECURITY;
ALTER TABLE messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages FORCE ROW LEVEL SECURITY;

-- Policies (drop+create for idempotent migrate)
DROP POLICY IF EXISTS agents_insert ON agents;
DROP POLICY IF EXISTS agents_select ON agents;
DROP POLICY IF EXISTS agents_update ON agents;
DROP POLICY IF EXISTS memories_all ON memories;
DROP POLICY IF EXISTS chunks_all ON chunks;
DROP POLICY IF EXISTS chat_sessions_all ON chat_sessions;
DROP POLICY IF EXISTS messages_all ON messages;

CREATE POLICY agents_insert ON agents
    FOR INSERT WITH CHECK (true);

CREATE POLICY agents_select ON agents
    FOR SELECT USING (id::text = current_setting('app.current_agent_id', true));

CREATE POLICY agents_update ON agents
    FOR UPDATE USING (id::text = current_setting('app.current_agent_id', true));

CREATE POLICY memories_all ON memories
    FOR ALL USING (agent_id::text = current_setting('app.current_agent_id', true))
    WITH CHECK (agent_id::text = current_setting('app.current_agent_id', true));

CREATE POLICY chunks_all ON chunks
    FOR ALL USING (agent_id::text = current_setting('app.current_agent_id', true))
    WITH CHECK (agent_id::text = current_setting('app.current_agent_id', true));

CREATE POLICY chat_sessions_all ON chat_sessions
    FOR ALL USING (agent_id::text = current_setting('app.current_agent_id', true))
    WITH CHECK (agent_id::text = current_setting('app.current_agent_id', true));

CREATE POLICY messages_all ON messages
    FOR ALL USING (agent_id::text = current_setting('app.current_agent_id', true))
    WITH CHECK (agent_id::text = current_setting('app.current_agent_id', true));