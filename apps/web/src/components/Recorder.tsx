"use client";

import { useEffect, useRef, useState } from "react";

type Props = {
  disabled?: boolean;
  onRecorded: (blob: Blob, durationMs: number) => Promise<void> | void;
};

export function Recorder({ disabled, onRecorded }: Props) {
  const [recording, setRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [busy, setBusy] = useState(false);
  const mediaRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const startedAt = useRef<number>(0);
  const timerRef = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      if (timerRef.current) window.clearInterval(timerRef.current);
      mediaRef.current?.stream.getTracks().forEach((t) => t.stop());
    };
  }, []);

  async function start() {
    if (disabled || busy) return;
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const recorder = new MediaRecorder(stream);
    chunksRef.current = [];
    recorder.ondataavailable = (e) => {
      if (e.data.size) chunksRef.current.push(e.data);
    };
    recorder.onstop = async () => {
      const blob = new Blob(chunksRef.current, { type: "audio/webm" });
      const durationMs = Date.now() - startedAt.current;
      stream.getTracks().forEach((t) => t.stop());
      setBusy(true);
      try {
        await onRecorded(blob, durationMs);
      } finally {
        setBusy(false);
      }
    };
    mediaRef.current = recorder;
    startedAt.current = Date.now();
    setSeconds(0);
    timerRef.current = window.setInterval(() => {
      setSeconds(Math.floor((Date.now() - startedAt.current) / 1000));
    }, 250);
    recorder.start();
    setRecording(true);
  }

  function stop() {
    if (!mediaRef.current || mediaRef.current.state === "inactive") return;
    mediaRef.current.stop();
    setRecording(false);
    if (timerRef.current) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "1rem" }}>
      <div style={{ position: "relative", width: 96, height: 96 }}>
        {recording && (
          <span
            aria-hidden
            style={{
              position: "absolute",
              inset: 0,
              borderRadius: "50%",
              border: "2px solid var(--accent)",
              animation: "pulse-ring 1.4s ease-out infinite",
            }}
          />
        )}
        <button
          type="button"
          disabled={disabled || busy}
          onClick={recording ? stop : start}
          aria-label={recording ? "Stop recording" : "Start recording"}
          style={{
            width: 96,
            height: 96,
            borderRadius: "50%",
            border: "1px solid var(--line)",
            background: recording ? "var(--danger)" : "rgba(232,226,214,0.06)",
            color: "var(--ink)",
            cursor: disabled || busy ? "not-allowed" : "pointer",
            opacity: disabled || busy ? 0.5 : 1,
            transition: "background 0.2s ease, transform 0.15s ease",
          }}
        >
          {busy ? "…" : recording ? "Stop" : "Rec"}
        </button>
      </div>
      <p style={{ margin: 0, color: "var(--muted)", fontSize: "0.95rem" }}>
        {recording ? `Recording ${seconds}s` : busy ? "Uploading…" : "Tap to record a memory"}
      </p>
    </div>
  );
}
