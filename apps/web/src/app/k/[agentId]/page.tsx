"use client";

import Link from "next/link";
import { useParams, useSearchParams } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { Citation, ChatResponse, apiUrl, getAgent, sendChat } from "@/lib/api";

type Turn = {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  audioUrl?: string | null;
  refused?: boolean;
};

export default function KeeperPage() {
  const params = useParams<{ agentId: string }>();
  const search = useSearchParams();
  const agentId = params.agentId;
  const token = search.get("token") || "";

  const [name, setName] = useState("Memory agent");
  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [locked, setLocked] = useState(false);
  const [lockDetail, setLockDetail] = useState("");
  const [listening, setListening] = useState(false);
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!token) {
      setError("Missing token in URL");
      return;
    }
    void getAgent(agentId, token)
      .then((a) => {
        setName(a.display_name);
        if (!a.ready_for_keeper) {
          setLocked(true);
          setLockDetail(
            `Need ${a.text_required} text + ${a.voice_required} voice memories. ` +
              `Have ${a.text_indexed} text, ${a.voice_indexed} voice indexed.`,
          );
        }
      })
      .catch((e) => setError(String(e)));
  }, [agentId, token]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns]);

  async function ask(message: string) {
    const text = message.trim();
    if (!text || !token || busy || locked) return;
    setBusy(true);
    setError(null);
    setTurns((t) => [...t, { role: "user", content: text }]);
    setInput("");
    try {
      const res: ChatResponse = await sendChat({
        agentId,
        token,
        message: text,
        sessionId,
        speak: true,
      });
      setSessionId(res.session_id);
      const audioUrl = res.audio_url ? apiUrl(res.audio_url) : null;
      setTurns((t) => [
        ...t,
        {
          role: "assistant",
          content: res.answer,
          citations: res.citations,
          audioUrl,
          refused: res.refused,
        },
      ]);
      if (audioUrl) {
        const audio = new Audio(audioUrl);
        void audio.play().catch(() => undefined);
      }
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  }

  function toggleMic() {
    const SR =
      typeof window !== "undefined"
        ? window.SpeechRecognition || window.webkitSpeechRecognition
        : undefined;
    if (!SR) {
      setError("Speech recognition not supported in this browser");
      return;
    }
    if (listening && recognitionRef.current) {
      recognitionRef.current.stop();
      setListening(false);
      return;
    }
    const recognition = new SR();
    recognition.lang = "en-US";
    recognition.interimResults = false;
    recognition.onresult = (event: SpeechRecognitionEvent) => {
      const transcript = event.results[0]?.[0]?.transcript;
      if (transcript) void ask(transcript);
    };
    recognition.onerror = () => setListening(false);
    recognition.onend = () => setListening(false);
    recognitionRef.current = recognition;
    setListening(true);
    recognition.start();
  }

  return (
    <main
      style={{
        minHeight: "100vh",
        display: "grid",
        gridTemplateRows: "auto 1fr auto",
        maxWidth: 720,
        margin: "0 auto",
        padding: "1.5rem 1.25rem 1rem",
      }}
    >
      <header className="rise">
        <Link href="/" style={{ color: "var(--muted)", fontSize: "0.9rem" }}>
          MIRA
        </Link>
        <h1
          style={{
            fontFamily: "var(--font-display)",
            fontSize: "clamp(2rem, 5vw, 2.8rem)",
            margin: "0.5rem 0 0.25rem",
          }}
        >
          {name}
        </h1>
        <p style={{ color: "var(--muted)", margin: 0 }}>
          They&apos;ll answer in their own voice, only from what they recorded.
        </p>
      </header>

      <section
        style={{
          overflowY: "auto",
          padding: "1.5rem 0",
          display: "flex",
          flexDirection: "column",
          gap: "1.1rem",
        }}
      >
        {locked && (
          <p className="rise" style={{ color: "var(--accent)", lineHeight: 1.55 }}>
            Keeper chat is locked until the maker finishes 3 text and 3 voice memories.
            <br />
            {lockDetail}
          </p>
        )}
        {!locked && turns.length === 0 && (
          <p className="rise" style={{ color: "var(--muted)" }}>
            Try: “What was your childhood home like?”
          </p>
        )}
        {turns.map((t, i) => (
          <article
            key={`${t.role}-${i}`}
            className="rise"
            style={{
              alignSelf: t.role === "user" ? "flex-end" : "flex-start",
              maxWidth: "92%",
              animationDelay: "0.05s",
            }}
          >
            <p
              style={{
                margin: 0,
                padding: t.role === "user" ? "0.75rem 1rem" : 0,
                borderRadius: t.role === "user" ? 16 : 0,
                background: t.role === "user" ? "rgba(212,163,92,0.16)" : "transparent",
                lineHeight: 1.55,
                color: t.refused ? "var(--muted)" : "var(--ink)",
              }}
            >
              {t.content}
            </p>
          </article>
        ))}
        <div ref={bottomRef} />
      </section>

      <footer
        style={{
          borderTop: "1px solid var(--line)",
          paddingTop: "0.9rem",
          display: "grid",
          gap: "0.65rem",
        }}
      >
        {error && <p style={{ color: "var(--danger)", margin: 0, fontSize: "0.9rem" }}>{error}</p>}
        <form
          onSubmit={(e) => {
            e.preventDefault();
            void ask(input);
          }}
          style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}
        >
          <button
            type="button"
            onClick={toggleMic}
            aria-label="Speak"
            style={{
              width: 44,
              height: 44,
              borderRadius: "50%",
              border: "1px solid var(--line)",
              background: listening ? "var(--danger)" : "rgba(0,0,0,0.25)",
              color: "var(--ink)",
              cursor: "pointer",
              flexShrink: 0,
            }}
          >
            mic
          </button>
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Talk with them…"
            disabled={busy || !token || locked}
            style={{
              flex: 1,
              padding: "0.8rem 1rem",
              borderRadius: 999,
              border: "1px solid var(--line)",
              background: "rgba(0,0,0,0.25)",
              color: "var(--ink)",
              outline: "none",
            }}
          />
          <button
            type="submit"
            disabled={busy || !input.trim() || locked}
            style={{
              background: "var(--accent)",
              color: "#1a140c",
              border: "none",
              borderRadius: 999,
              padding: "0.75rem 1.1rem",
              fontWeight: 600,
              cursor: "pointer",
              opacity: busy || !input.trim() ? 0.5 : 1,
            }}
          >
            Ask
          </button>
        </form>
      </footer>
    </main>
  );
}
