/**
 * Gemini path. Translates an Anthropic Messages request into Google's
 * generateContent API, runs it (against the free tier), and translates
 * the response back into Anthropic's message / SSE format.
 *
 * Used for capabilities Workers AI can't do well today:
 *   - vision (image content blocks)
 *   - very long contexts beyond Workers AI's reliable window
 *
 * Free tier docs: https://ai.google.dev/pricing — the gateway picks
 * gemini-2.5-flash by default (highest free-tier RPD as of writing).
 */

import type {
  AnthropicMessagesRequest,
  AnthropicMessageResponse,
  AnthropicMessage,
  AnthropicContentBlock,
  AnthropicResponseContentBlock,
  AnthropicStopReason,
  AnthropicTool,
  GeminiGenerateRequest,
  GeminiContent,
  GeminiPart,
  GeminiResponse,
} from "./types.js";

const GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta";
export const DEFAULT_GEMINI_MODEL = "gemini-2.5-flash";

export interface CallGeminiOptions {
  apiKey: string;
  model: string;
}

/**
 * Build the Gemini generateContent request body from an Anthropic
 * request, including tool declarations, image parts, and tool_use /
 * tool_result history.
 */
export function buildGeminiInput(req: AnthropicMessagesRequest): GeminiGenerateRequest {
  const systemParts: Array<{ text: string }> = [];
  if (typeof req.system === "string" && req.system) {
    systemParts.push({ text: req.system });
  } else if (Array.isArray(req.system)) {
    for (const block of req.system as AnthropicContentBlock[]) {
      if (block.type === "text" && "text" in block && typeof block.text === "string") {
        systemParts.push({ text: block.text });
      }
    }
  }

  // Pre-scan: build a tool_use_id → tool_name map across all prior
  // assistant messages so we can attach the right name to each
  // tool_result on the user side. Gemini's functionResponse expects the
  // tool name, not the Anthropic id.
  const toolNameById = buildToolUseIdMap(req.messages);

  const contents: GeminiContent[] = [];
  for (const m of req.messages) {
    const translated = translateMessageToGemini(m, toolNameById);
    if (translated) contents.push(translated);
  }

  const out: GeminiGenerateRequest = {
    contents,
    generationConfig: {
      maxOutputTokens: clampMaxTokens(req.max_tokens),
      ...(typeof req.temperature === "number" ? { temperature: req.temperature } : {}),
      ...(typeof req.top_p === "number" ? { topP: req.top_p } : {}),
      ...(typeof req.top_k === "number" ? { topK: req.top_k } : {}),
      ...(Array.isArray(req.stop_sequences) && req.stop_sequences.length > 0
        ? { stopSequences: req.stop_sequences }
        : {}),
    },
  };

  if (systemParts.length > 0) {
    out.systemInstruction = { parts: systemParts };
  }

  if (Array.isArray(req.tools) && req.tools.length > 0) {
    out.tools = [{ functionDeclarations: translateToolsToGemini(req.tools) }];
  }

  return out;
}

export function translateToolsToGemini(tools: AnthropicTool[]) {
  return tools.map((t) => ({
    name: t.name,
    description: t.description,
    parameters: sanitizeJsonSchemaForGemini(
      (t.input_schema as Record<string, unknown>) ?? { type: "object", properties: {} },
    ),
  }));
}

/**
 * Gemini's function-calling parameter schema is a strict subset of JSON
 * Schema (modeled after OpenAPI 3.0's Schema object). Anthropic-format
 * tool schemas frequently include keywords Gemini rejects with HTTP 400:
 * `$schema`, `additionalProperties`, `exclusiveMinimum`,
 * `exclusiveMaximum`, `propertyNames`, `patternProperties`, `$defs`, etc.
 *
 * We use an *allowlist* of fields Gemini documents support for — anything
 * not on the list is dropped on the way through. This is more durable
 * than blocklisting because new Anthropic / JSON-Schema keywords don't
 * silently break us in the future.
 *
 * Reference: https://ai.google.dev/api/caching#Schema
 */
const GEMINI_SCHEMA_ALLOWED_FIELDS = new Set([
  "type",
  "format",
  "description",
  "nullable",
  "enum",
  "items",
  "properties",
  "required",
  "minItems",
  "maxItems",
  "minLength",
  "maxLength",
  "pattern",
  "minimum",
  "maximum",
  "multipleOf",
  "title",
  "default",
  "example",
  "anyOf",
]);

/**
 * Walk a JSON Schema node, keep only Gemini-allowed schema keywords, and
 * recursively sanitize the schemas embedded as values:
 *   - `properties`: arbitrary property-name keys; values are sub-schemas
 *   - `items`: single sub-schema
 *   - `anyOf` / `allOf` / `oneOf`: array of sub-schemas
 *
 * The allowlist applies to *schema-keyword* keys, not to arbitrary
 * property names — otherwise we'd erase the user's property names too.
 */
export function sanitizeJsonSchemaForGemini(schema: Record<string, unknown>): Record<string, unknown> {
  return sanitizeSchemaNode(schema) as Record<string, unknown>;
}

function sanitizeSchemaNode(node: unknown): unknown {
  if (node === null || typeof node !== "object" || Array.isArray(node)) return node;
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(node as Record<string, unknown>)) {
    if (!GEMINI_SCHEMA_ALLOWED_FIELDS.has(k)) continue;
    if (k === "properties") {
      out[k] = sanitizePropertiesMap(v);
    } else if (k === "items") {
      out[k] = sanitizeSchemaNode(v);
    } else if (k === "anyOf") {
      out[k] = Array.isArray(v) ? v.map(sanitizeSchemaNode) : v;
    } else {
      // type, description, enum (array of primitives), required (array of
      // strings), pattern, minimum/maximum/etc. — pass through unchanged.
      out[k] = v;
    }
  }
  return out;
}

function sanitizePropertiesMap(node: unknown): Record<string, unknown> {
  if (node === null || typeof node !== "object" || Array.isArray(node)) return {};
  const out: Record<string, unknown> = {};
  for (const [propName, propSchema] of Object.entries(node as Record<string, unknown>)) {
    out[propName] = sanitizeSchemaNode(propSchema);
  }
  return out;
}

function buildToolUseIdMap(messages: AnthropicMessage[]): Map<string, string> {
  const map = new Map<string, string>();
  for (const m of messages) {
    if (m.role !== "assistant" || !Array.isArray(m.content)) continue;
    for (const block of m.content) {
      if (block.type === "tool_use") {
        const tu = block as { id: string; name: string };
        if (typeof tu.id === "string" && typeof tu.name === "string") {
          map.set(tu.id, tu.name);
        }
      }
    }
  }
  return map;
}

function translateMessageToGemini(m: AnthropicMessage, toolNameById: Map<string, string>): GeminiContent | null {
  const role = m.role === "assistant" ? "model" : "user";
  const parts: GeminiPart[] = [];

  if (typeof m.content === "string") {
    if (m.content) parts.push({ text: m.content });
  } else {
    for (const block of m.content) {
      if (block.type === "text" && "text" in block && typeof block.text === "string") {
        if (block.text) parts.push({ text: block.text });
      } else if (block.type === "image") {
        const src = (block as { source?: unknown }).source as
          | { type?: string; media_type?: string; data?: string }
          | undefined;
        if (src && src.type === "base64" && typeof src.data === "string") {
          parts.push({
            inlineData: {
              mimeType: src.media_type ?? "image/png",
              data: src.data,
            },
          });
        }
      } else if (block.type === "tool_use") {
        const tu = block as { name: string; input: unknown };
        const args =
          typeof tu.input === "object" && tu.input !== null
            ? (tu.input as Record<string, unknown>)
            : {};
        parts.push({ functionCall: { name: tu.name, args } });
      } else if (block.type === "tool_result") {
        const tr = block as { tool_use_id: string; content: unknown };
        let response: Record<string, unknown>;
        if (typeof tr.content === "string") {
          response = { result: tr.content };
        } else if (Array.isArray(tr.content)) {
          response = {
            result: tr.content
              .map((c) =>
                typeof c === "object" && c && "text" in c && typeof (c as { text: unknown }).text === "string"
                  ? (c as { text: string }).text
                  : "",
              )
              .join("\n"),
          };
        } else if (tr.content && typeof tr.content === "object") {
          response = tr.content as Record<string, unknown>;
        } else {
          response = {};
        }
        // Gemini's functionResponse uses the tool *name*. Look it up
        // from the prior assistant tool_use blocks; fall back to the id
        // if we somehow can't find it (the model can still associate
        // via positional order in that case).
        const toolName = toolNameById.get(tr.tool_use_id) ?? tr.tool_use_id;
        parts.push({
          functionResponse: {
            name: toolName,
            response,
          },
        });
      }
    }
  }

  if (parts.length === 0) return null;
  return { role, parts };
}

function clampMaxTokens(requested: number): number {
  if (!Number.isFinite(requested) || requested <= 0) return 1024;
  return requested;
}

export async function callGemini(
  opts: CallGeminiOptions,
  req: AnthropicMessagesRequest,
): Promise<Response> {
  const body = buildGeminiInput(req);
  const wantsStream = Boolean(req.stream);
  const action = wantsStream ? "streamGenerateContent" : "generateContent";
  const url = `${GEMINI_API_BASE}/models/${encodeURIComponent(opts.model)}:${action}${
    wantsStream ? "?alt=sse&" : "?"
  }key=${encodeURIComponent(opts.apiKey)}`;

  const upstream = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!upstream.ok) {
    const errText = await upstream.text().catch(() => "");
    throw new Error(`Gemini ${upstream.status}: ${errText.slice(0, 400)}`);
  }

  if (wantsStream) {
    if (!upstream.body) {
      throw new Error("Gemini stream had no body");
    }
    return streamGeminiAsAnthropic(upstream.body, req.model || opts.model);
  }

  const parsed = (await upstream.json()) as GeminiResponse;
  const anthropicShape = geminiToAnthropicMessage(parsed, req.model || opts.model);
  return new Response(JSON.stringify(anthropicShape), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}

export function geminiToAnthropicMessage(
  result: GeminiResponse,
  modelLabel: string,
): AnthropicMessageResponse {
  const candidate = result.candidates?.[0];
  const parts = candidate?.content?.parts ?? [];

  const content: AnthropicResponseContentBlock[] = [];
  let textBuf = "";
  let hasToolCall = false;

  for (const part of parts) {
    if ("text" in part && typeof part.text === "string") {
      textBuf += part.text;
    } else if ("functionCall" in part) {
      if (textBuf) {
        content.push({ type: "text", text: textBuf });
        textBuf = "";
      }
      const fc = part.functionCall;
      content.push({
        type: "tool_use",
        id: `toolu_${randomId()}`,
        name: fc.name,
        input: fc.args ?? {},
      });
      hasToolCall = true;
    }
  }
  if (textBuf) {
    content.push({ type: "text", text: textBuf });
  }
  if (content.length === 0) {
    content.push({ type: "text", text: "" });
  }

  let stopReason: AnthropicStopReason = "end_turn";
  if (hasToolCall) {
    stopReason = "tool_use";
  } else if (candidate?.finishReason === "MAX_TOKENS") {
    stopReason = "max_tokens";
  } else if (candidate?.finishReason === "STOP" || candidate?.finishReason === undefined) {
    stopReason = "end_turn";
  }

  return {
    id: `msg_gm_${randomId()}`,
    type: "message",
    role: "assistant",
    model: modelLabel,
    content,
    stop_reason: stopReason,
    stop_sequence: null,
    usage: {
      input_tokens: result.usageMetadata?.promptTokenCount ?? 0,
      output_tokens: result.usageMetadata?.candidatesTokenCount ?? 0,
    },
  };
}

/**
 * Translate Gemini's SSE stream (each `data:` frame is a partial
 * GeminiResponse JSON) into Anthropic's SSE event sequence.
 *
 * Gemini's streaming protocol emits incremental candidate.content.parts
 * — we accumulate per-part state so each text or functionCall part
 * becomes one Anthropic content block.
 */
export function streamGeminiAsAnthropic(
  upstream: ReadableStream<Uint8Array>,
  modelLabel: string,
): Response {
  const messageId = `msg_gm_${randomId()}`;
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();

  const downstream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const writeEvent = (event: string, data: unknown) => {
        controller.enqueue(
          encoder.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`),
        );
      };

      writeEvent("message_start", {
        type: "message_start",
        message: {
          id: messageId,
          type: "message",
          role: "assistant",
          model: modelLabel,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: 0, output_tokens: 0 },
        },
      });

      let blockIndex = -1;
      let currentBlockKind: "text" | "tool_use" | null = null;
      // Anthropic's protocol expects stop_reason="tool_use" when the
      // assistant turn contains any tool_use block. The runner on the
      // consumer side branches on this exact value to decide whether to
      // execute the tool — without it, the chat appears to stall after
      // the model decides to call something. Track whether we ever
      // emitted a tool_use so the finalizer can shape the right reason.
      let emittedToolUse = false;
      let outputTokens = 0;
      let inputTokens = 0;
      let finishReason: string | undefined;
      let streamError = false;
      let buffer = "";

      const closeCurrentBlock = () => {
        if (currentBlockKind !== null) {
          writeEvent("content_block_stop", { type: "content_block_stop", index: blockIndex });
          currentBlockKind = null;
        }
      };

      const openTextBlock = () => {
        if (currentBlockKind !== "text") {
          closeCurrentBlock();
          blockIndex += 1;
          currentBlockKind = "text";
          writeEvent("content_block_start", {
            type: "content_block_start",
            index: blockIndex,
            content_block: { type: "text", text: "" },
          });
        }
      };

      const reader = upstream.getReader();
      try {
        while (true) {
          const { value, done } = await reader.read();
          if (done) {
            const tail = decoder.decode();
            if (tail) buffer += tail;
            if (buffer.trim()) processFrame(buffer.trim());
            buffer = "";
            break;
          }
          buffer += decoder.decode(value, { stream: true });
          let cut: number;
          while ((cut = buffer.indexOf("\n\n")) !== -1) {
            const frame = buffer.slice(0, cut);
            buffer = buffer.slice(cut + 2);
            processFrame(frame);
          }
        }
      } catch (err) {
        streamError = true;
        writeEvent("error", {
          type: "error",
          error: {
            type: "api_error",
            message: err instanceof Error ? err.message : "Gemini stream failed",
          },
        });
      } finally {
        reader.releaseLock();
        if (!streamError) {
          closeCurrentBlock();
          let stopReason: AnthropicStopReason = "end_turn";
          if (finishReason === "MAX_TOKENS") stopReason = "max_tokens";
          // tool_use takes precedence over end_turn. Consumer runners
          // (e.g. Anthropic SDK's `messages.stream()`) branch on the
          // accumulated stop_reason to decide whether to execute the
          // tool and re-loop; mis-reporting end_turn here causes the
          // chat to silently stall after a tool call.
          if (emittedToolUse) stopReason = "tool_use";
          writeEvent("message_delta", {
            type: "message_delta",
            delta: { stop_reason: stopReason, stop_sequence: null },
            usage: { output_tokens: outputTokens, input_tokens: inputTokens },
          });
          writeEvent("message_stop", { type: "message_stop" });
        }
        try {
          controller.close();
        } catch {
          // already closed
        }
      }

      function processFrame(frame: string) {
        const payload = parseSseFrame(frame);
        if (!payload || payload === "[DONE]") return;
        let parsed: GeminiResponse;
        try {
          parsed = JSON.parse(payload) as GeminiResponse;
        } catch {
          return;
        }
        const candidate = parsed.candidates?.[0];
        const parts = candidate?.content?.parts ?? [];
        for (const part of parts) {
          if ("text" in part && typeof part.text === "string" && part.text) {
            openTextBlock();
            outputTokens += Math.max(1, Math.ceil(part.text.length / 4));
            writeEvent("content_block_delta", {
              type: "content_block_delta",
              index: blockIndex,
              delta: { type: "text_delta", text: part.text },
            });
          } else if ("functionCall" in part) {
            closeCurrentBlock();
            blockIndex += 1;
            currentBlockKind = "tool_use";
            emittedToolUse = true;
            const fc = part.functionCall;
            writeEvent("content_block_start", {
              type: "content_block_start",
              index: blockIndex,
              content_block: {
                type: "tool_use",
                id: `toolu_${randomId()}`,
                name: fc.name,
                input: {},
              },
            });
            const argsJson = JSON.stringify(fc.args ?? {});
            writeEvent("content_block_delta", {
              type: "content_block_delta",
              index: blockIndex,
              delta: { type: "input_json_delta", partial_json: argsJson },
            });
            writeEvent("content_block_stop", { type: "content_block_stop", index: blockIndex });
            currentBlockKind = null;
          }
        }
        if (candidate?.finishReason) finishReason = candidate.finishReason;
        if (parsed.usageMetadata?.promptTokenCount) {
          inputTokens = parsed.usageMetadata.promptTokenCount;
        }
        if (parsed.usageMetadata?.candidatesTokenCount) {
          outputTokens = parsed.usageMetadata.candidatesTokenCount;
        }
      }
    },
  });

  return new Response(downstream, {
    status: 200,
    headers: {
      "content-type": "text/event-stream; charset=utf-8",
      "cache-control": "no-cache, no-transform",
      "x-accel-buffering": "no",
    },
  });
}

function parseSseFrame(frame: string): string | null {
  const lines = frame.split("\n");
  for (const line of lines) {
    if (line.startsWith("data:")) return line.slice(5).trim();
  }
  return null;
}

function randomId(): string {
  const buf = new Uint8Array(6);
  crypto.getRandomValues(buf);
  return Array.from(buf, (b) => b.toString(16).padStart(2, "0")).join("");
}

/**
 * Quick capability sniff: does the request need vision support?
 * Used by the router to prefer Gemini for image-bearing calls.
 */
export function requestHasImage(req: AnthropicMessagesRequest): boolean {
  for (const m of req.messages) {
    if (Array.isArray(m.content)) {
      for (const block of m.content) {
        if (block.type === "image") return true;
      }
    }
  }
  return false;
}
