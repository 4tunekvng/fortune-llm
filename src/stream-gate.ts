/**
 * Holds a free tier's Anthropic-shaped stream until its first words, so a
 * tier that fails before saying anything is treated like one that failed
 * before replying: the chain moves on to the next tier (fortune-llm#52).
 * Until then nothing has gone to the caller, so nothing is lost.
 *
 * "First words" is the first non-blank text delta, tool-call argument
 * fragment, or tool_use block. An error event, a message_stop, or the end
 * of the stream before that fails the tier.
 *
 * Once words have gone out no other tier can take over, so a tier that
 * then goes quiet for `idleMs` is cut: the stream ends on an error event
 * the caller can retry, and the upstream call is stopped.
 */

import { splitSseFrames } from "./sse.js";

const encoder = new TextEncoder();

type FrameKind =
  | { kind: "words" | "stop" | "other" }
  | { kind: "error"; message: string };

function frameKind(frame: string): FrameKind {
  const line = frame.split("\n").find((l) => l.startsWith("data:"));
  if (!line) return { kind: "other" };
  let data: {
    type?: string;
    delta?: { text?: unknown; partial_json?: unknown };
    content_block?: { type?: string };
    error?: { message?: unknown };
  };
  try {
    data = JSON.parse(line.slice(5).trim());
  } catch {
    return { kind: "other" };
  }
  if (data.type === "error") {
    const message = data.error?.message;
    const text = typeof message === "string" ? message : "error";
    return { kind: "error", message: text };
  }
  if (data.type === "message_stop") return { kind: "stop" };
  const block = data.content_block?.type;
  if (data.type === "content_block_start" && block === "tool_use") {
    return { kind: "words" };
  }
  if (data.type === "content_block_delta") {
    const { text, partial_json: json } = data.delta ?? {};
    if (typeof text === "string" && text.trim() !== "") return { kind: "words" };
    if (typeof json === "string" && json !== "") return { kind: "words" };
  }
  return { kind: "other" };
}

/** One Anthropic error event, in the shape the translators write. */
function errorEvent(message: string): Uint8Array {
  const data = { type: "error", error: { type: "api_error", message } };
  return encoder.encode(`event: error\ndata: ${JSON.stringify(data)}\n\n`);
}

export interface HoldOptions {
  /** Longest quiet gap allowed once words have gone out. */
  idleMs: number;
  /** Aborts while held (the tier's time limit) cancel the read. */
  signal: AbortSignal;
  /** Stops the upstream call once the caller no longer needs it. */
  stopUpstream: () => void;
}

/**
 * Reads `resp` until its first words and returns a response that replays
 * them and passes the rest through. Throws, so the chain tries the next
 * tier, when the stream fails, finishes or ends before its first words.
 */
export async function holdUntilFirstWords(
  resp: Response,
  label: string,
  opts: HoldOptions,
): Promise<Response> {
  if (!resp.body) throw new Error(`${label} stream had no body`);
  const reader = resp.body.getReader();
  const cancel = () => {
    opts.stopUpstream();
    reader.cancel().catch(() => {});
  };
  opts.signal.addEventListener("abort", cancel, { once: true });
  const held: Uint8Array[] = [];
  const decoder = new TextDecoder();
  let buffer = "";
  let spoke = false;
  try {
    while (!spoke) {
      const { value, done } = await reader.read();
      buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
      if (value) held.push(value);
      const { frames, rest } = splitSseFrames(buffer);
      buffer = rest;
      if (done && rest.trim()) frames.push(rest);
      for (const frame of frames) {
        const seen = frameKind(frame);
        if (seen.kind === "error") {
          const when = spoke
            ? "before its words were sent"
            : "before its first words";
          throw new Error(`${label} stream failed ${when}: ${seen.message}`);
        }
        if (seen.kind === "words") spoke = true;
        if (seen.kind === "stop" && !spoke) {
          throw new Error(`${label} returned an empty reply`);
        }
      }
      if (done && !spoke) {
        throw new Error(`${label} stream ended before its first words`);
      }
    }
  } catch (err) {
    cancel();
    throw err;
  } finally {
    opts.signal.removeEventListener("abort", cancel);
  }

  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of held) controller.enqueue(chunk);
    },
    async pull(controller) {
      let timer: ReturnType<typeof setTimeout> | undefined;
      const quiet = new Promise<"quiet">((resolve) => {
        timer = setTimeout(() => resolve("quiet"), opts.idleMs);
      });
      const next = await Promise.race([reader.read(), quiet]).finally(() =>
        clearTimeout(timer),
      );
      if (next === "quiet") {
        cancel();
        controller.enqueue(errorEvent(`${label} went quiet for ${opts.idleMs} ms`));
        controller.close();
      } else if (next.done) {
        controller.close();
      } else {
        controller.enqueue(next.value);
      }
    },
    cancel() {
      cancel();
    },
  });
  return new Response(body, {
    status: resp.status,
    statusText: resp.statusText,
    headers: resp.headers,
  });
}
