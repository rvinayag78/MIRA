import Link from "next/link";
import type { CSSProperties } from "react";

const btn: CSSProperties = {
  display: "inline-block",
  minWidth: 180,
  textAlign: "center",
  background: "var(--accent)",
  color: "#1a140c",
  padding: "0.95rem 1.5rem",
  borderRadius: 999,
  fontWeight: 600,
};

const ghost: CSSProperties = {
  ...btn,
  background: "transparent",
  color: "var(--ink)",
  border: "1px solid var(--line)",
};

export default function HomePage() {
  return (
    <main
      style={{
        minHeight: "100vh",
        display: "grid",
        placeItems: "center",
        padding: "2rem",
        position: "relative",
        overflow: "hidden",
      }}
    >
      <div
        aria-hidden
        style={{
          position: "absolute",
          inset: "18% 20%",
          borderRadius: "50%",
          background: "radial-gradient(circle, rgba(212,163,92,0.16), transparent 70%)",
          animation: "breathe 7s ease-in-out infinite",
          pointerEvents: "none",
        }}
      />
      <section className="rise" style={{ maxWidth: 640, textAlign: "center", zIndex: 1 }}>
        <p
          style={{
            fontFamily: "var(--font-display)",
            fontSize: "clamp(3.5rem, 10vw, 6rem)",
            margin: 0,
            letterSpacing: "-0.03em",
            lineHeight: 1,
          }}
        >
          MIRA
        </p>
        <p style={{ color: "var(--muted)", margin: "1.25rem 0 2rem", lineHeight: 1.6 }}>
          Add memories as the maker. Talk with them as the keeper.
        </p>
        <div style={{ display: "flex", gap: "0.85rem", justifyContent: "center", flexWrap: "wrap" }}>
          <Link href="/maker" style={btn}>
            Maker
          </Link>
          <Link href="/keeper" style={ghost}>
            Keeper
          </Link>
        </div>
      </section>
    </main>
  );
}
