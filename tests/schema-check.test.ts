import { describe, expect, it } from "vitest";
import { zodOutputFormat } from "@anthropic-ai/sdk/helpers/zod";
import { z } from "zod";
import { schemaMismatch } from "../src/schema-check.js";

// Schemas built the way the apps build them, with the SDK's own helper, so
// the checker sees exactly what reaches the gateway.
const letter = zodOutputFormat(
  z.object({
    docType: z.enum(["eob", "denial", "other"]),
    tag: z.literal("v1"),
    billedCents: z.number().int(),
    note: z.string().nullable(),
    facts: z.array(z.string()),
    isActionable: z.boolean(),
  }),
).schema as Record<string, unknown>;

const good = {
  docType: "denial",
  tag: "v1",
  billedCents: 245000,
  note: null,
  facts: ["MRI denied"],
  isActionable: true,
};

describe("schemaMismatch on the SDK's output", () => {
  it("accepts a reply that matches", () => {
    // Act
    const miss = schemaMismatch(good, letter);

    // Assert
    expect(miss).toBeNull();
  });

  it("names an enum value outside the list", () => {
    // Arrange: Workers AI's failure in issue #28.
    const reply = { ...good, docType: "Denial Letter" };

    // Act
    const miss = schemaMismatch(reply, letter);

    // Assert
    expect(miss).toBe("$.docType is not one of the allowed values");
  });

  it("names a missing required field", () => {
    // Arrange
    const { facts: _facts, ...reply } = good;

    // Act
    const miss = schemaMismatch(reply, letter);

    // Assert
    expect(miss).toBe("$.facts is missing");
  });

  it("allows an extra key, which zod drops", () => {
    // Arrange: the SDK sends the object closed, but messages.parse accepts.
    const reply = { ...good, extra: 1 };

    // Act
    const miss = schemaMismatch(reply, letter);

    // Assert
    expect(miss).toBeNull();
  });

  it("allows a missing field that has a default, which zod fills in", () => {
    // Arrange: Maestro's skill kind defaults to "practice".
    const schema = zodOutputFormat(
      z.object({
        kind: z.enum(["speaking", "practice"]).default("practice"),
        prompt: z.string().default(""),
        name: z.string(),
      }),
    ).schema as Record<string, unknown>;

    // Act
    const misses = [schemaMismatch({ name: "x" }, schema), schemaMismatch({}, schema)];

    // Assert
    expect(misses).toEqual([null, "$.name is missing"]);
  });

  it("allows a filled z.record map, which the SDK sends as a closed object", () => {
    // Arrange
    const schema = zodOutputFormat(
      z.object({ counts: z.record(z.string(), z.number()) }),
    ).schema as Record<string, unknown>;

    // Act
    const miss = schemaMismatch({ counts: { apples: 3 } }, schema);

    // Assert
    expect(miss).toBeNull();
  });

  it("refuses a fraction where an integer is required", () => {
    // Arrange
    const reply = { ...good, billedCents: 2450.5 };

    // Act
    const miss = schemaMismatch(reply, letter);

    // Assert
    expect(miss).toBe("$.billedCents is number, expected integer");
  });

  it("refuses a wrong literal", () => {
    // Arrange
    const reply = { ...good, tag: "v2" };

    // Act
    const miss = schemaMismatch(reply, letter);

    // Assert
    expect(miss).toBe("$.tag is not the one allowed value");
  });

  it("refuses a wrong type inside an array", () => {
    // Arrange
    const reply = { ...good, facts: ["ok", 3] };

    // Act
    const miss = schemaMismatch(reply, letter);

    // Assert
    expect(miss).toBe("$.facts[1] is integer, expected string");
  });
});

// Shapes written by hand, none of them used while writing the checker.
describe("schemaMismatch on shapes beyond the apps' schemas", () => {
  const defs = {
    $defs: {
      step: {
        type: "object",
        properties: { n: { type: "integer" } },
        required: ["n"],
      },
    },
    type: "object",
    properties: { steps: { type: "array", items: { $ref: "#/$defs/step" } } },
  };

  it("follows a local $ref", () => {
    // Act
    const miss = schemaMismatch({ steps: [{ n: 1 }, {}] }, defs);

    // Assert
    expect(miss).toBe("$.steps[1].n is missing");
  });

  it("accepts a value that fits one branch of a union", () => {
    // Arrange
    const union = { anyOf: [{ type: "string" }, { type: "integer" }] };

    // Act
    const misses = [schemaMismatch("a", union), schemaMismatch(2, union)];

    // Assert
    expect(misses).toEqual([null, null]);
  });

  it("refuses a value that fits no branch of a union", () => {
    // Arrange
    const union = { oneOf: [{ type: "string" }, { type: "integer" }] };

    // Act
    const miss = schemaMismatch(true, union);

    // Assert
    expect(miss).toBe("$ is boolean, expected string");
  });

  it("checks tuple positions and an allOf", () => {
    // Arrange
    const schema = {
      allOf: [{ type: "array" }],
      prefixItems: [{ type: "string" }, { type: "number" }],
    };

    // Act
    const misses = [
      schemaMismatch(["a", 1], schema),
      schemaMismatch([1, 1], schema),
    ];

    // Assert
    expect(misses).toEqual([null, "$[0] is integer, expected string"]);
  });

  it("checks the values of a map", () => {
    // Arrange
    const map = { type: "object", additionalProperties: { type: "number" } };

    // Act
    const misses = [schemaMismatch({ a: 1 }, map), schemaMismatch({ a: "x" }, map)];

    // Assert
    expect(misses).toEqual([null, "$.a is string, expected number"]);
  });

  it("ignores keywords it does not know", () => {
    // Arrange
    const schema = { type: "string", format: "date", minLength: 99, "x-hint": 1 };

    // Act
    const miss = schemaMismatch("2026-11-21", schema);

    // Assert
    expect(miss).toBeNull();
  });

  it("takes boolean schemas and nullable", () => {
    // Act
    const misses = [
      schemaMismatch(1, true),
      schemaMismatch(1, false),
      schemaMismatch(null, { type: "string", nullable: true }),
    ];

    // Assert
    expect(misses).toEqual([null, "$ is not allowed", null]);
  });

  it("stops on a $ref that points back at itself", () => {
    // Arrange
    const loop = { $defs: { a: { $ref: "#/$defs/a" } }, $ref: "#/$defs/a" };

    // Act
    const miss = schemaMismatch({ any: 1 }, loop);

    // Assert
    expect(miss).toBeNull();
  });

  it("follows a recursive tree", () => {
    // Arrange: what a z.lazy() tree turns into.
    const tree = {
      $defs: {
        node: {
          type: "object",
          properties: {
            name: { type: "string" },
            kids: { type: "array", items: { $ref: "#/$defs/node" } },
          },
          required: ["name"],
        },
      },
      $ref: "#/$defs/node",
    };

    // Act
    const reply = { name: "a", kids: [{ name: "b", kids: [{}] }] };
    const miss = schemaMismatch(reply, tree);

    // Assert
    expect(miss).toBe("$.kids[0].kids[0].name is missing");
  });
});
