"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { Recorder } from "@/components/Recorder";
import {
  Agent,
  Memory,
  cloneVoice,
  createAgent,
  getAgent,
  listMemories,
  uploadMemory,
} from "@/lib/api";

const STORAGE_KEY = "ovyu_maker_agent";

type Stored = { id: string; token: string; display_name: string };

export default function MakerPage() {
  const [name, setName] = useState("");
  const [agent, setAgent] = useState<Agent | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [memories, setMemories] = useState<Memory[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const keeperPath = useMemo(() => {
    if (!agent || !token) return null;
    return `/k/${agent.id}?token=${encodeURIComponent(token)}`;
  }, [agent, token]);

  const refresh = useCallback(async (id: string, tok: string) => {
    const [a, m] = await Promise.all([getAgent(id, tok), listMemories(id, tok)]);
    setAgent(a);
    setMemories(m);
  }, []);

  useEffect(() => {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    try {
      const stored = JSON.parse(raw) as Stored;
      setToken(stored.token);
      setName(stored.display_name);
      void refresh(stored.id, stored.token).catch((e) => setError(String(e)));
    } catch {
      localStorage.removeItem(STORAGE_KEY);
    }
  }, [refresh]);

  useEffect(() => {
    if (!agent || !token) return;
    const hasPending = memories.some((m) =>
      ["pending", "transcribing", "embedding", "cloning"].includes(m.status),
    ) || agent.status === "cloning";
    if (!hasPending) return;
    const t = window.setInterval(() => {
      void refresh(agent.id, token).catch(() => undefined);
    }, 2500);
    return () => window.clearInterval(t);
  }, [agent, token, memories, refresh]);

  async function onCreate(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      const a = await createAgent(name.trim() || "Maker");
      if (!a.share_token) throw new Error("Missing share token");
      const stored: Stored = {
        id: a.id,
        token: a.share_token,
        display_name: a.display_name,
      };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(stored));
      setToken(a.share_token);
      setAgent(a);
      setMemories([]);
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  }

  async function onRecorded(blob: Blob, durationMs: number) {
    if (!agent || !token) return;
    setError(null);
    try {
      await uploadMemory(agent.id, token, blob, durationMs);
      await refresh(agent.id, token);
    } catch (err) {
      setError(String(err));
    }
  }

  async function onClone() {
    if (!agent || !token) return;
    setError(null);
    setBusy(true);
    try {
      await cloneVoice(agent.id, token);
      await refresh(agent.id, token);
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  }

  function reset() {
    localStorage.removeItem(STORAGE_KEY);
    setAgent(null);
    setToken(null);
    setMemories([]);
  }

  return (
    <main style={{ maxWidth: 720, margin: "0 auto", padding: "2rem 1.25rem 4rem" }}>
      <header className="rise" style={{ marginBottom: "2.5rem" }}>
        <Link href="/" style={{ color: "var(--muted)", fontSize: "0.9rem" }}>
          ← Ovyu
        </Link>
        <h1
          style={{
            fontFamily: "var(--font-display)",
            fontSize: "clamp(2.4rem, 6vw, 3.4rem)",
            margin: "0.75rem 0 0.35rem",
            letterSpacing: "-0.02em",
          }}
        >
          Maker
        </h1>
        <p style={{ color: "var(--muted)", margin: 0, lineHeight: 1.55 }}>
          Record memories. Clone your voice. Share a keeper link.
        </p>
      </header>

      {!agent || !token ? (
        <form
          onSubmit={onCreate}
          className="rise"
          style={{ display: "grid", gap: "0.85rem", maxWidth: 420 }}
        >
          <label style={{ display: "grid", gap: "0.4rem" }}>
            <span style={{ color: "var(--muted)", fontSize: "0.9rem" }}>Your name</span>
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Alex"
              style={inputStyle}
            />
          </label>
          <button type="submit" disabled={busy} style={primaryBtn}>
            Create agent
          </button>
        </form>
      ) : (
        <div style={{ display: "grid", gap: "2.25rem" }}>
          <section className="rise" style={{ display: "grid", gap: "1rem" }}>
            <StatusRow agent={agent} memories={memories} />
            <Recorder onRecorded={onRecorded} />
            <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
              <button
                type="button"
                onClick={onClone}
                disabled={busy || (agent.indexed_memories ?? 0) < 1}
                style={primaryBtn}
              >
                Clone my voice
              </button>
              <button type="button" onClick={reset} style={ghostBtn}>
                New maker
              </button>
            </div>
          </section>

          {keeperPath && (
            <section className="rise" style={{ borderTop: "1px solid var(--line)", paddingTop: "1.5rem" }}>
              <h2 style={{ fontFamily: "var(--font-display)", fontSize: "1.4rem", margin: "0 0 0.5rem" }}>
                Keeper link
              </h2>
              <p style={{ color: "var(--muted)", margin: "0 0 0.75rem", fontSize: "0.95rem" }}>
                Share this so someone can ask about your memories.
              </p>
              <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", alignItems: "center" }}>
                <code
                  style={{
                    flex: 1,
                    minWidth: 200,
                    padding: "0.7rem 0.85rem",
                    border: "1px solid var(--line)",
                    borderRadius: 10,
                    color: "var(--ink)",
                    background: "rgba(0,0,0,0.2)",
                    wordBreak: "break-all",
                    fontSize: "0.85rem",
                  }}
                >
                  {typeof window !== "undefined" ? `${window.location.origin}${keeperPath}` : keeperPath}
                </code>
                <Link href={keeperPath} style={{ ...primaryBtn, display: "inline-block" }}>
                  Open
                </Link>
              </div>
            </section>
          )}

          <section className="rise" style={{ borderTop: "1px solid var(--line)", paddingTop: "1.5rem" }}>
            <h2 style={{ fontFamily: "var(--font-display)", fontSize: "1.4rem", margin: "0 0 1rem" }}>
              Memories
            </h2>
            {memories.length === 0 ? (
              <p style={{ color: "var(--muted)" }}>No memories yet — record your first one.</p>
            ) : (
              <ul style={{ listStyle: "none", margin: 0, padding: 0, display: "grid", gap: "0.65rem" }}>
                {memories.map((m) => (
                  <li
                    key={m.id}
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      gap: "1rem",
                      padding: "0.85rem 0",
                      borderBottom: "1px solid var(--line)",
                      fontSize: "0.95rem",
                    }}
                  >
                    <span style={{ color: "var(--muted)" }}>
                      {new Date(m.created_at).toLocaleString()}
                      {m.duration_ms != null ? ` · ${Math.round(m.duration_ms / 1000)}s` : ""}
                    </span>
                    <span style={{ color: statusColor(m.status) }}>{m.status}</span>
                  </li>
                ))}
              </ul>
            )}
          </section>
        </div>
      )}

      {error && (
        <p style={{ color: "var(--danger)", marginTop: "1.5rem", whiteSpace: "pre-wrap" }}>{error}</p>
      )}
    </main>
  );
}

function StatusRow({ agent, memories }: { agent: Agent; memories: Memory[] }) {
  const indexed = agent.indexed_memories ?? memories.filter((m) => m.status === "indexed").length;
  const voiceReady = Boolean(agent.elevenlabs_voice_id) && agent.status === "ready";
  return (
    <div style={{ display: "flex", gap: "1.25rem", flexWrap: "wrap", color: "var(--muted)" }}>
      <span>
        <strong style={{ color: "var(--ink)", fontWeight: 600 }}>{agent.display_name}</strong>
      </span>
      <span>
        Memories indexed:{" "}
        <strong style={{ color: "var(--ink)" }}>{indexed}</strong>
      </span>
      <span>
        Voice:{" "}
        <strong style={{ color: voiceReady ? "var(--ok)" : "var(--accent)" }}>
          {agent.status === "cloning" ? "cloning…" : voiceReady ? "ready" : "not cloned"}
        </strong>
      </span>
    </div>
  );
}

function statusColor(status: string): string {
  if (status === "indexed") return "var(--ok)";
  if (status === "error") return "var(--danger)";
  return "var(--accent)";
}

const inputStyle: React.CSSProperties = {
  padding: "0.8rem 0.95rem",
  borderRadius: 12,
  border: "1px solid var(--line)",
  background: "rgba(0,0,0,0.25)",
  color: "var(--ink)",
  outline: "none",
};

const primaryBtn: React.CSSProperties = {
  background: "var(--accent)",
  color: "#1a140c",
  border: "none",
  padding: "0.75rem 1.2rem",
  borderRadius: 999,
  fontWeight: 600,
  cursor: "pointer",
};

const ghostBtn: React.CSSProperties = {
  background: "transparent",
  color: "var(--muted)",
  border: "1px solid var(--line)",
  padding: "0.75rem 1.2rem",
  borderRadius: 999,
  cursor: "pointer",
};
