/**
 * Checks a free tier's structured reply against the JSON schema the caller
 * sent in `output_config`. The Anthropic SDK's `messages.parse` throws on a
 * reply that misses the schema, so passing one on as a success only makes
 * the caller wait and then start again; a mismatch is a tier failure.
 *
 * Covers the JSON Schema the SDK's `zodOutputFormat` sends: boolean schemas,
 * `type` (one or a list), `enum`, `const`, `properties`, `required`,
 * `additionalProperties` given as a schema, `items`, `prefixItems`, `anyOf`,
 * `oneOf` (checked as anyOf), `allOf`, `nullable`, and local `$ref` into
 * `$defs` or `definitions`. Every other keyword is ignored, so a keyword
 * this file does not know can never fail a good reply.
 *
 * The bar is what the app's own parser accepts, not the letter of the
 * schema. Zod, behind the SDK's `messages.parse`, drops keys it doesn't know
 * and fills in fields that have a default, while the SDK sends every object
 * as closed (`additionalProperties: false`, which is also how it writes a
 * `z.record` map) and every field as required, with the default moved into
 * the description as `{default: ...}`. So extra keys are allowed, and a
 * field with a default may be missing. Length and range limits are not
 * checked either: the SDK moves those into `description` as hints.
 */

type Schema = boolean | Record<string, unknown>;

/**
 * A `$ref` that points back at itself without going deeper into the value
 * (`{"$defs": {"A": {"$ref": "#/$defs/A"}}}`) would recurse forever. Real
 * replies are never this deep, so past this many nested checks the reply is
 * taken as matching rather than failing a good one.
 */
const MAX_DEPTH = 64;

/** The first place `value` misses `schema`, or null when it matches. */
export function schemaMismatch(value: unknown, schema: Schema): string | null {
  const root = typeof schema === "object" ? schema : {};
  return check(value, schema, root, "$", 0);
}

function check(
  value: unknown,
  schema: Schema,
  root: Record<string, unknown>,
  path: string,
  depth: number,
): string | null {
  if (depth > MAX_DEPTH || schema === true) return null;
  if (schema === false) return `${path} is not allowed`;
  if (typeof schema !== "object" || schema === null) return null;
  const ref = schema.$ref;
  if (typeof ref === "string") {
    const target = resolveRef(ref, root);
    if (target !== undefined) {
      const miss = check(value, target, root, path, depth + 1);
      if (miss) return miss;
    }
  }
  if (value === null && schema.nullable === true) return null;
  return (
    checkType(value, schema.type, path) ??
    checkEnum(value, schema, path) ??
    checkCombinators(value, schema, root, path, depth) ??
    checkObject(value, schema, root, path, depth) ??
    checkArray(value, schema, root, path, depth)
  );
}

function resolveRef(ref: string, root: Record<string, unknown>): Schema | undefined {
  const match = /^#\/(\$defs|definitions)\/(.+)$/.exec(ref);
  if (!match) return ref === "#" ? root : undefined;
  const defs = root[match[1] as string] as Record<string, Schema> | undefined;
  return defs?.[decodeURIComponent(match[2] as string)];
}

function typeOf(value: unknown): string {
  if (value === null) return "null";
  if (Array.isArray(value)) return "array";
  if (typeof value === "number") {
    return Number.isInteger(value) ? "integer" : "number";
  }
  return typeof value;
}

function checkType(value: unknown, type: unknown, path: string): string | null {
  if (type === undefined) return null;
  const allowed = Array.isArray(type) ? type : [type];
  const actual = typeOf(value);
  const ok = allowed.some(
    (t) => t === actual || (t === "number" && actual === "integer"),
  );
  return ok ? null : `${path} is ${actual}, expected ${allowed.join(" or ")}`;
}

function checkEnum(
  value: unknown,
  schema: Record<string, unknown>,
  path: string,
): string | null {
  if ("const" in schema && !sameJson(value, schema.const)) {
    return `${path} is not the one allowed value`;
  }
  const options = schema.enum;
  if (Array.isArray(options) && !options.some((o) => sameJson(value, o))) {
    return `${path} is not one of the allowed values`;
  }
  return null;
}

function sameJson(a: unknown, b: unknown): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

function checkCombinators(
  value: unknown,
  schema: Record<string, unknown>,
  root: Record<string, unknown>,
  path: string,
  depth: number,
): string | null {
  for (const sub of asSchemas(schema.allOf)) {
    const miss = check(value, sub, root, path, depth + 1);
    if (miss) return miss;
  }
  for (const key of ["anyOf", "oneOf"]) {
    const options = asSchemas(schema[key]);
    if (options.length === 0) continue;
    const misses = options.map((s) => check(value, s, root, path, depth + 1));
    if (misses.every((m) => m !== null)) return misses[0] as string;
  }
  return null;
}

function asSchemas(v: unknown): Schema[] {
  return Array.isArray(v) ? (v as Schema[]) : [];
}

function checkObject(
  value: unknown,
  schema: Record<string, unknown>,
  root: Record<string, unknown>,
  path: string,
  depth: number,
): string | null {
  if (typeOf(value) !== "object") return null;
  const obj = value as Record<string, unknown>;
  const props = (schema.properties ?? {}) as Record<string, Schema>;
  const required = Array.isArray(schema.required) ? schema.required : [];
  for (const key of required) {
    if (!(key in obj) && !hasDefault(props[String(key)])) {
      return `${path}.${String(key)} is missing`;
    }
  }
  // `false` closes the object, but zod drops the extra keys rather than
  // failing, so only a schema for extra values is checked.
  const extra = schema.additionalProperties;
  const extraSchema =
    typeof extra === "object" && extra !== null ? (extra as Schema) : undefined;
  for (const [key, v] of Object.entries(obj)) {
    const sub = key in props ? props[key] : extraSchema;
    if (sub === undefined) continue;
    const miss = check(v, sub, root, `${path}.${key}`, depth + 1);
    if (miss) return miss;
  }
  return null;
}

/** A field zod fills in when the reply leaves it out. */
function hasDefault(schema: Schema | undefined): boolean {
  if (typeof schema !== "object" || schema === null) return false;
  if ("default" in schema) return true;
  const hint = schema.description;
  return typeof hint === "string" && hint.includes("{default:");
}

function checkArray(
  value: unknown,
  schema: Record<string, unknown>,
  root: Record<string, unknown>,
  path: string,
  depth: number,
): string | null {
  if (!Array.isArray(value)) return null;
  const tuple = asSchemas(schema.prefixItems);
  const rest = schema.items as Schema | undefined;
  for (let i = 0; i < value.length; i++) {
    const sub = i < tuple.length ? tuple[i] : rest;
    if (sub === undefined) continue;
    const miss = check(value[i], sub, root, `${path}[${i}]`, depth + 1);
    if (miss) return miss;
  }
  return null;
}
