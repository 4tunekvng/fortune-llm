import { describe, it, expect } from "vitest";
import { ghostThoughtFilter, stripGhostThought } from "../src/ghost-thought.js";

const GEMMA = "@cf/google/gemma-4-26b-a4b-it";
const LLAMA = "llama-3.3-70b-versatile";
// The reply Leggo saved on 2026-11-24 (leggo#170).
const ANSWER = "I've drafted a weekend drinks ask for the College Group.";
const SEEN = `thought\n${ANSWER}`;
// Gemma 4 31B on OpenRouter, mid-reply before a tool call (a Lena eval).
const MID = "I need a few details:\n\n<|channel>thought\n<channel|>";

describe("stripGhostThought", () => {
  it.each([
    ["the bare word from Gemma", SEEN, GEMMA, ANSWER],
    ["the bare word with a Windows line end", `thought\r\n${ANSWER}`, GEMMA, ANSWER],
    ["the bare word after blank space", `\n thought\n\n${ANSWER}`, GEMMA, ANSWER],
    ["an empty block", `<|channel>thought\n<channel|>${ANSWER}`, LLAMA, ANSWER],
    [
      "a block with reasoning inside",
      `<|channel>thought\nThe user wants a plan.<channel|>\n\n${ANSWER}`,
      GEMMA,
      ANSWER,
    ],
    ["a block in the middle of a reply", MID, GEMMA, "I need a few details:\n\n"],
    [
      "two blocks",
      "Hi <|channel>a<channel|>there<|channel>b<channel|>.",
      LLAMA,
      "Hi there.",
    ],
    [
      "the word and keeps the answer's indentation",
      "thought\n\n  indented code",
      GEMMA,
      "  indented code",
    ],
  ])("drops %s", (_name, input, model, expected) => {
    expect(stripGhostThought(input, model)).toBe(expected);
  });

  it.each([
    ["the bare word from another model", "thought\nbrought\ncaught", LLAMA],
    ["an answer about a thought", "thought about it: brunch works", GEMMA],
    ["a capitalised first word", "Thought you'd like this.\nBrunch?", GEMMA],
    ["a word that starts with thought", "thoughtful idea\nbrunch", GEMMA],
    ["the bare word later in the reply", `Hi\nthought\n${ANSWER}`, GEMMA],
    ["a block that never closes", "<|channel>thought\nstill going", GEMMA],
    ["blank space before an ordinary answer", "\n  Brunch?", GEMMA],
    ["an indented answer", "  indented code\nmore", GEMMA],
    ["an empty reply", "", GEMMA],
  ])("keeps %s", (_name, input, model) => {
    expect(stripGhostThought(input, model)).toBe(input);
  });
});

describe("ghostThoughtFilter", () => {
  const run = (deltas: string[], model = GEMMA) => {
    const filter = ghostThoughtFilter(model);
    const shown = deltas.map((d) => filter.push(d));
    return { shown, flushed: filter.flush() };
  };

  it("drops the bare word when it arrives in pieces", () => {
    const deltas = ["tho", "ught", "\n", "I've drafted", " a plan."];

    const { shown, flushed } = run(deltas);

    expect(shown).toEqual(["", "", "", "I've drafted", " a plan."]);
    expect(flushed).toBe("");
  });

  it("drops the word and keeps the answer sent in the same piece", () => {
    const { shown } = run([SEEN]);

    expect(shown).toEqual([ANSWER]);
  });

  it("drops the blank line that follows the word in a later piece", () => {
    const { shown } = run(["thought\n", "\n", "Brunch?"]);

    expect(shown.join("")).toBe("Brunch?");
  });

  it("drops a block whose tokens arrive in pieces", () => {
    const deltas = ["<|chan", "nel>thought\nplanning", "<chan", "nel|>", "Brunch?"];

    const { shown } = run(deltas, LLAMA);

    expect(shown).toEqual(["", "", "", "", "Brunch?"]);
  });

  it("sends the words before a block in the middle of a reply", () => {
    const deltas = [
      "I need a few",
      " details:\n\n<|chan",
      "nel>thought\n",
      "<channel|>",
    ];

    const { shown, flushed } = run(deltas);

    expect(shown).toEqual(["I need a few", " details:\n\n", "", ""]);
    expect(flushed).toBe("");
  });

  it("passes an ordinary answer straight through", () => {
    const { shown, flushed } = run(["Hello", " there"]);

    expect(shown).toEqual(["Hello", " there"]);
    expect(flushed).toBe("");
  });

  it("passes the bare word through from a model that is not Gemma", () => {
    const { shown } = run(["thought\n", "brought"], LLAMA);

    expect(shown).toEqual(["thought\n", "brought"]);
  });

  it("hands back an answer that only looked like the word at first", () => {
    const { shown, flushed } = run(["thought", " about it", " more"]);

    expect(shown).toEqual(["", "thought about it", " more"]);
    expect(flushed).toBe("");
  });

  it("hands back what it held when the reply ends early", () => {
    const { shown, flushed } = run(["Hi <|chan"]);

    expect(shown).toEqual(["Hi "]);
    expect(flushed).toBe("<|chan");
  });

  it.each([
    [["thought\n\n", "Brunch?"]],
    [["thought", "\n", "\n", "Bru", "nch?"]],
    [["\n", "<|channel>thought\n<channel|>", "\n\nBrunch?"]],
    [["<|channel>thought\n<channel|>\n", "\n", "Brunch?"]],
  ])("gives the same text as the whole reply for pieces %j", (deltas) => {
    const { shown, flushed } = run(deltas);

    expect(shown.join("") + flushed).toBe(stripGhostThought(deltas.join(""), GEMMA));
  });
});
