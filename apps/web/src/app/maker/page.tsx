"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { Recorder } from "@/components/Recorder";
import {
  Agent,
  Memory,
  STORAGE_KEY,
  createAgent,
  createTextMemory,
  getAgent,
  listMemories,
  uploadMemory,
} from "@/lib/api";

type Stored = { id: string; token: string; display_name: string };
type Mode = "choose" | "text" | "voice";

export default function MakerPage() {
  const [agent, setAgent] = useState<Agent | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [memories, setMemories] = useState<Memory[]>([]);
  const [mode, setMode] = useState<Mode>("choose");
  const [text, setText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [booting, setBooting] = useState(true);

  const textCount = useMemo(
    () => memories.filter((m) => m.kind === "text").length,
    [memories],
  );
  const voiceCount = useMemo(
    () => memories.filter((m) => m.kind === "voice").length,
    [memories],
  );
  const textIndexed = agent?.text_indexed ?? 0;
  const voiceIndexed = agent?.voice_indexed ?? 0;
  const ready = Boolean(agent?.ready_for_keeper);

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
    async function boot() {
      setBooting(true);
      setError(null);
      try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (raw) {
          const stored = JSON.parse(raw) as Stored;
          setToken(stored.token);
          await refresh(stored.id, stored.token);
        } else {
          const a = await createAgent("Maker");
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
        }
      } catch (err) {
        setError(String(err));
      } finally {
        setBooting(false);
      }
    }
    void boot();
  }, [refresh]);

  useEffect(() => {
    if (!agent || !token) return;
    const pending =
      memories.some((m) => ["pending", "transcribing", "embedding"].includes(m.status)) ||
      agent.status === "cloning";
    if (!pending) return;
    const t = window.setInterval(() => {
      void refresh(agent.id, token).catch(() => undefined);
    }, 2000);
    return () => window.clearInterval(t);
  }, [agent, token, memories, refresh]);

  async function onSaveText(e: React.FormEvent) {
    e.preventDefault();
    if (!agent || !token || !text.trim() || textCount >= 3) return;
    setBusy(true);
    setError(null);
    try {
      await createTextMemory(agent.id, token, text.trim());
      setText("");
      await refresh(agent.id, token);
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  }

  async function onRecorded(blob: Blob, durationMs: number) {
    if (!agent || !token || voiceCount >= 3) return;
    setError(null);
    try {
      await uploadMemory(agent.id, token, blob, durationMs);
      await refresh(agent.id, token);
    } catch (err) {
      setError(String(err));
    }
  }

  function reset() {
    localStorage.removeItem(STORAGE_KEY);
    window.location.reload();
  }

  if (booting) {
    return (
      <main style={{ maxWidth: 720, margin: "0 auto", padding: "2rem" }}>
        <p style={{ color: "var(--muted)" }}>Starting maker…</p>
      </main>
    );
  }

  return (
    <main style={{ maxWidth: 720, margin: "0 auto", padding: "2rem 1.25rem 4rem" }}>
      <header className="rise" style={{ marginBottom: "2rem" }}>
        <Link href="/" style={{ color: "var(--muted)", fontSize: "0.9rem" }}>
          ← MIRA
        </Link>
        <h1
          style={{
            fontFamily: "var(--font-display)",
            fontSize: "clamp(2.4rem, 6vw, 3.4rem)",
            margin: "0.75rem 0 0.35rem",
          }}
        >
          Maker
        </h1>
        <p style={{ color: "var(--muted)", margin: 0 }}>
          Add 3 text memories and 3 voice memories to unlock the keeper.
        </p>
        <div
          style={{
            display: "flex",
            gap: "1rem",
            marginTop: "1rem",
            color: "var(--muted)",
            flexWrap: "wrap",
          }}
        >
          <span>
            Text:{" "}
            <strong style={{ color: textIndexed >= 3 ? "var(--ok)" : "var(--ink)" }}>
              {textIndexed}/3 indexed
            </strong>{" "}
            ({textCount} added)
          </span>
          <span>
            Voice:{" "}
            <strong style={{ color: voiceIndexed >= 3 ? "var(--ok)" : "var(--ink)" }}>
              {voiceIndexed}/3 indexed
            </strong>{" "}
            ({voiceCount} added)
          </span>
        </div>
      </header>

      {mode === "choose" && (
        <section className="rise" style={{ display: "grid", gap: "0.85rem", maxWidth: 420 }}>
          <button
            type="button"
            onClick={() => setMode("text")}
            disabled={textCount >= 3}
            style={primaryBtn}
          >
            Create text memories
          </button>
          <button
            type="button"
            onClick={() => setMode("voice")}
            disabled={voiceCount >= 3}
            style={primaryBtn}
          >
            Create voice memories
          </button>
          <button type="button" onClick={reset} style={ghostBtn}>
            Reset maker
          </button>
        </section>
      )}

      {mode === "text" && (
        <section className="rise" style={{ display: "grid", gap: "1rem" }}>
          <button type="button" onClick={() => setMode("choose")} style={ghostBtn}>
            ← Back
          </button>
          {textCount >= 3 ? (
            <p style={{ color: "var(--ok)" }}>All 3 text memories added.</p>
          ) : (
            <form onSubmit={onSaveText} style={{ display: "grid", gap: "0.75rem" }}>
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                rows={6}
                placeholder="Write a memory…"
                style={{ ...inputStyle, resize: "vertical", borderRadius: 14 }}
              />
              <button type="submit" disabled={busy || !text.trim()} style={primaryBtn}>
                Save text memory ({textCount + 1}/3)
              </button>
            </form>
          )}
          <MemoryList memories={memories.filter((m) => m.kind === "text")} />
        </section>
      )}

      {mode === "voice" && (
        <section className="rise" style={{ display: "grid", gap: "1rem" }}>
          <button type="button" onClick={() => setMode("choose")} style={ghostBtn}>
            ← Back
          </button>
          {voiceCount >= 3 ? (
            <p style={{ color: "var(--ok)" }}>All 3 voice memories added.</p>
          ) : (
            <Recorder disabled={busy} onRecorded={onRecorded} />
          )}
          <MemoryList memories={memories.filter((m) => m.kind === "voice")} />
        </section>
      )}

      {ready && keeperPath && (
        <section
          className="rise"
          style={{ marginTop: "2.5rem", borderTop: "1px solid var(--line)", paddingTop: "1.5rem" }}
        >
          <h2 style={{ fontFamily: "var(--font-display)", fontSize: "1.4rem", margin: "0 0 0.5rem" }}>
            Keeper unlocked
          </h2>
          <p style={{ color: "var(--muted)", margin: "0 0 0.85rem" }}>
            You have 3 text and 3 voice memories. Share this link:
          </p>
          <Link href={keeperPath} style={{ ...primaryBtn, display: "inline-block" }}>
            Open keeper chat
          </Link>
        </section>
      )}

      {error && (
        <p style={{ color: "var(--danger)", marginTop: "1.5rem", whiteSpace: "pre-wrap" }}>{error}</p>
      )}
    </main>
  );
}

function MemoryList({ memories }: { memories: Memory[] }) {
  if (memories.length === 0) return null;
  return (
    <ul style={{ listStyle: "none", margin: "0.5rem 0 0", padding: 0, display: "grid", gap: "0.5rem" }}>
      {memories.map((m) => (
        <li
          key={m.id}
          style={{
            display: "flex",
            justifyContent: "space-between",
            gap: "1rem",
            padding: "0.7rem 0",
            borderBottom: "1px solid var(--line)",
            fontSize: "0.92rem",
          }}
        >
          <span style={{ color: "var(--muted)", overflow: "hidden", textOverflow: "ellipsis" }}>
            {m.kind === "text"
              ? (m.text_content || "Text memory").slice(0, 80)
              : `Voice · ${m.duration_ms != null ? `${Math.round(m.duration_ms / 1000)}s` : "audio"}`}
          </span>
          <span style={{ color: m.status === "indexed" ? "var(--ok)" : "var(--accent)" }}>
            {m.status}
          </span>
        </li>
      ))}
    </ul>
  );
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
  width: "fit-content",
};
