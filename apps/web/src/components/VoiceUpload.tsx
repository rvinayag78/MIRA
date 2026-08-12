"use client";

import { useRef, useState } from "react";

type Props = {
  disabled?: boolean;
  onUploaded: (blob: Blob, filename: string, durationMs?: number) => Promise<void> | void;
};

const ACCEPT = "audio/*,.webm,.wav,.mp3,.m4a,.ogg,.aac,.flac";

export function VoiceUpload({ disabled, onUploaded }: Props) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [busy, setBusy] = useState(false);
  const [fileName, setFileName] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function onChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file || disabled || busy) return;

    setBusy(true);
    setFileName(file.name);
    setError(null);
    try {
      const durationMs = await readAudioDurationMs(file);
      await onUploaded(file, file.name, durationMs);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div style={{ display: "grid", gap: "0.65rem", justifyItems: "center", width: "100%" }}>
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT}
        hidden
        disabled={disabled || busy}
        onChange={onChange}
      />
      <button
        type="button"
        disabled={disabled || busy}
        onClick={() => inputRef.current?.click()}
        style={{
          background: "transparent",
          color: "var(--ink)",
          border: "1px solid var(--line)",
          padding: "0.75rem 1.2rem",
          borderRadius: 999,
          fontWeight: 600,
          cursor: disabled || busy ? "not-allowed" : "pointer",
          opacity: disabled || busy ? 0.5 : 1,
        }}
      >
        {busy ? "Uploading…" : "Upload voice recording"}
      </button>
      <p style={{ margin: 0, color: "var(--muted)", fontSize: "0.9rem", textAlign: "center" }}>
        {fileName ? fileName : "webm, wav, mp3, m4a, and similar"}
      </p>
      {error && (
        <p style={{ margin: 0, color: "var(--danger)", fontSize: "0.85rem", textAlign: "center" }}>
          {error}
        </p>
      )}
    </div>
  );
}

function readAudioDurationMs(file: File): Promise<number | undefined> {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file);
    const audio = new Audio();
    audio.preload = "metadata";
    const finish = (ms?: number) => {
      URL.revokeObjectURL(url);
      resolve(ms);
    };
    audio.onloadedmetadata = () => {
      const seconds = audio.duration;
      if (!Number.isFinite(seconds) || seconds <= 0) finish(undefined);
      else finish(Math.round(seconds * 1000));
    };
    audio.onerror = () => finish(undefined);
    audio.src = url;
  });
}
