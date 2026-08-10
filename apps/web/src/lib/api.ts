const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type Agent = {
  id: string;
  display_name: string;
  elevenlabs_voice_id: string | null;
  status: string;
  created_at: string;
  share_token?: string | null;
  keeper_url?: string | null;
  indexed_memories?: number | null;
  text_indexed: number;
  voice_indexed: number;
  text_required: number;
  voice_required: number;
  ready_for_keeper: boolean;
};

export type Memory = {
  id: string;
  agent_id: string;
  kind: "text" | "voice";
  text_content: string | null;
  audio_uri: string | null;
  duration_ms: number | null;
  status: string;
  error_message: string | null;
  created_at: string;
};

export type Citation = { chunk_id: string; quote: string };

export type ChatResponse = {
  session_id: string;
  answer: string;
  citations: Citation[];
  confidence: number;
  refused: boolean;
  intent: string;
  audio_url: string | null;
};

export const STORAGE_KEY = "mira_maker_agent";

export function apiUrl(path: string): string {
  return `${API_URL}${path}`;
}

export async function createAgent(displayName: string): Promise<Agent> {
  const res = await fetch(apiUrl("/agents"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ display_name: displayName }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getAgent(agentId: string, token: string): Promise<Agent> {
  const res = await fetch(apiUrl(`/agents/${agentId}?token=${encodeURIComponent(token)}`), {
    headers: { "X-Agent-Token": token },
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function listMemories(agentId: string, token: string): Promise<Memory[]> {
  const res = await fetch(
    apiUrl(`/memories?agent_id=${agentId}&token=${encodeURIComponent(token)}`),
    { headers: { "X-Agent-Token": token } },
  );
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function createTextMemory(
  agentId: string,
  token: string,
  text: string,
): Promise<Memory> {
  const res = await fetch(apiUrl("/memories/text"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ agent_id: agentId, token, text }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function uploadMemory(
  agentId: string,
  token: string,
  blob: Blob,
  durationMs?: number,
): Promise<Memory> {
  const form = new FormData();
  form.append("agent_id", agentId);
  form.append("token", token);
  if (durationMs != null) form.append("duration_ms", String(durationMs));
  form.append("file", blob, "recording.webm");
  const res = await fetch(apiUrl("/memories"), { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function sendChat(params: {
  agentId: string;
  token: string;
  message: string;
  sessionId?: string | null;
  speak?: boolean;
}): Promise<ChatResponse> {
  const res = await fetch(apiUrl("/chat"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      agent_id: params.agentId,
      token: params.token,
      message: params.message,
      session_id: params.sessionId || null,
      speak: params.speak ?? true,
    }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
