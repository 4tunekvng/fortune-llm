/**
 * Gemma 4's larger models can write a thinking block into a reply even with
 * thinking turned off; Google's guide calls these "ghost" thought channels
 * (https://ai.google.dev/gemma/docs/capabilities/thinking). The block is
 * `<|channel>thought …<channel|>`, usually empty. Some servers pass it on
 * whole, sometimes mid-reply just before a tool call; others drop its two
 * special tokens and leave the bare word "thought" on the reply's first
 * line (leggo#170). Either way consumers read it as part of the answer.
 *
 * What is removed, and why each is safe:
 * - A closed block anywhere, from any model: its tokens are Gemma's
 *   special tokens, not words.
 * - The bare lowercase word alone on the first line, from Gemma only: that
 *   is the token-stripped block, but from another model it could be a real
 *   answer's first line (a list of past tenses, say).
 * - Blank lines left at the start of a reply once a ghost there is gone.
 *
 * A closed block is also removed from an answer that quotes Gemma's own
 * template; real chat answers don't, and leaking the block costs more.
 */
const OPEN = "<|channel>";
const CLOSE = "<channel|>";
const BARE = /^\s*thought[ \t]*\r?\n/;
// Blank lines a dropped ghost leaves before the answer; indentation stays.
const BLANK_LINES = /^(?:[ \t]*\r?\n)+/;

/** Models whose server may leave the bare word behind. */
function mayLeaveBareWord(model: string): boolean {
  return /gemma/i.test(model);
}

/** True while `held` could still grow into the bare word and its line end. */
function couldBeBare(held: string): boolean {
  const s = held.trimStart();
  return "thought".startsWith(s) || /^thought[ \t]*\r?$/.test(s);
}

/** Length of the longest end of `text` that `OPEN` starts with. */
function partialOpenAtEnd(text: string): number {
  for (let n = Math.min(OPEN.length - 1, text.length); n > 0; n--) {
    if (OPEN.startsWith(text.slice(-n))) return n;
  }
  return 0;
}

export interface GhostThoughtFilter {
  /** Takes the next piece of reply text; returns the text to send now. */
  push(delta: string): string;
  /** Returns whatever is still held once the reply text ends. */
  flush(): string;
}

/**
 * Removes ghost thought blocks from a reply's text as it streams. It holds
 * text back only while it could still be part of a ghost: the reply's
 * opening (blank space, or the start of the bare word) and anything from a
 * block's opening token until its closing one. A block that never closes is
 * sent on as it came when the reply ends.
 */
export function ghostThoughtFilter(model: string): GhostThoughtFilter {
  let held = "";
  let bareChecked = !mayLeaveBareWord(model);
  // No answer text sent yet; blank space so far waits in `leadingSpace`.
  let atStart = true;
  let leadingSpace = "";
  let droppedAtStart = false;

  const send = (text: string): string => {
    if (!atStart) return text;
    if (!text.trim()) {
      leadingSpace += text;
      return "";
    }
    atStart = false;
    const opening = leadingSpace + text;
    const out = droppedAtStart ? opening.replace(BLANK_LINES, "") : opening;
    leadingSpace = "";
    return out;
  };

  const drain = (final: boolean): string => {
    let out = "";
    while (true) {
      if (!bareChecked) {
        const bare = BARE.exec(held);
        if (bare) {
          held = held.slice(bare[0].length);
          droppedAtStart = true;
        } else if (!final && couldBeBare(held)) {
          return out;
        }
        bareChecked = true;
      }
      const open = held.indexOf(OPEN);
      if (open >= 0) {
        const close = held.indexOf(CLOSE, open + OPEN.length);
        if (close < 0 && !final) {
          out += send(held.slice(0, open));
          held = held.slice(open);
          return out;
        }
        if (close >= 0) {
          out += send(held.slice(0, open));
          if (atStart) droppedAtStart = true;
          held = held.slice(close + CLOSE.length);
          continue;
        }
      }
      const keep = final ? 0 : partialOpenAtEnd(held);
      out += send(held.slice(0, held.length - keep));
      held = held.slice(held.length - keep);
      if (final && atStart && !droppedAtStart) {
        out += leadingSpace;
        leadingSpace = "";
      }
      return out;
    }
  };

  return {
    push(delta) {
      held += delta;
      return drain(false);
    },
    flush() {
      return drain(true);
    },
  };
}

/** The same filter run over a whole reply at once. */
export function stripGhostThought(text: string, model: string): string {
  const filter = ghostThoughtFilter(model);
  return filter.push(text) + filter.flush();
}
