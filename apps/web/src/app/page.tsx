import Link from "next/link";

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
          Ovyu
        </p>
        <h1
          style={{
            fontWeight: 500,
            fontSize: "clamp(1.1rem, 2.5vw, 1.35rem)",
            margin: "1.25rem 0 0.5rem",
            color: "var(--ink)",
          }}
        >
          Memories that speak — and stay honest.
        </h1>
        <p style={{ color: "var(--muted)", margin: "0 0 2rem", lineHeight: 1.6 }}>
          Record what matters. Clone the voice. Let a keeper ask — answers only from what was said.
        </p>
        <div style={{ display: "flex", gap: "0.75rem", justifyContent: "center", flexWrap: "wrap" }}>
          <Link
            href="/maker"
            style={{
              background: "var(--accent)",
              color: "#1a140c",
              padding: "0.85rem 1.4rem",
              borderRadius: 999,
              fontWeight: 600,
              transition: "transform 0.2s ease",
            }}
          >
            Start as maker
          </Link>
        </div>
      </section>
    </main>
  );
}