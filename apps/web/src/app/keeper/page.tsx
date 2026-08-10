"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { STORAGE_KEY, getAgent } from "@/lib/api";

type Stored = { id: string; token: string; display_name: string };

export default function KeeperEntryPage() {
  const [status, setStatus] = useState<"loading" | "ready" | "locked" | "missing">("loading");
  const [detail, setDetail] = useState("");

  useEffect(() => {
    async function load() {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) {
        setStatus("missing");
        return;
      }
      try {
        const stored = JSON.parse(raw) as Stored;
        const agent = await getAgent(stored.id, stored.token);
        const path = `/k/${stored.id}?token=${encodeURIComponent(stored.token)}`;
        if (agent.ready_for_keeper) {
          setStatus("ready");
          window.location.replace(path);
        } else {
          setStatus("locked");
          setDetail(
            `Need ${agent.text_required} text + ${agent.voice_required} voice memories. ` +
              `Have ${agent.text_indexed} text, ${agent.voice_indexed} voice indexed.`,
          );
        }
      } catch (err) {
        setStatus("missing");
        setDetail(String(err));
      }
    }
    void load();
  }, []);

  return (
    <main style={{ maxWidth: 640, margin: "0 auto", padding: "2rem 1.25rem" }}>
      <Link href="/" style={{ color: "var(--muted)", fontSize: "0.9rem" }}>
        ← MIRA
      </Link>
      <h1
        style={{
          fontFamily: "var(--font-display)",
          fontSize: "clamp(2.4rem, 6vw, 3.2rem)",
          margin: "0.75rem 0 0.5rem",
        }}
      >
        Keeper
      </h1>

      {status === "loading" && <p style={{ color: "var(--muted)" }}>Checking memories…</p>}

      {status === "ready" && <p style={{ color: "var(--muted)" }}>Opening chat…</p>}

      {status === "locked" && (
        <div className="rise" style={{ display: "grid", gap: "1rem" }}>
          <p style={{ color: "var(--muted)", lineHeight: 1.55 }}>
            The maker must finish 3 text memories and 3 voice memories before you can chat.
          </p>
          <p style={{ color: "var(--accent)" }}>{detail}</p>
          <Link
            href="/maker"
            style={{
              width: "fit-content",
              background: "var(--accent)",
              color: "#1a140c",
              padding: "0.75rem 1.2rem",
              borderRadius: 999,
              fontWeight: 600,
            }}
          >
            Go to maker
          </Link>
        </div>
      )}

      {status === "missing" && (
        <div className="rise" style={{ display: "grid", gap: "1rem" }}>
          <p style={{ color: "var(--muted)", lineHeight: 1.55 }}>
            No maker profile found in this browser. Create memories as the maker first, then come
            back — or open a keeper link the maker shared.
          </p>
          {detail && <p style={{ color: "var(--danger)" }}>{detail}</p>}
          <Link
            href="/maker"
            style={{
              width: "fit-content",
              background: "var(--accent)",
              color: "#1a140c",
              padding: "0.75rem 1.2rem",
              borderRadius: 999,
              fontWeight: 600,
            }}
          >
            Go to maker
          </Link>
        </div>
      )}
    </main>
  );
}
