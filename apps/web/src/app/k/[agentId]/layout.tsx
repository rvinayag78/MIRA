import { Suspense } from "react";

export default function KeeperLayout({ children }: { children: React.ReactNode }) {
  return <Suspense fallback={<main style={{ padding: "2rem" }}>Loading…</main>}>{children}</Suspense>;
}