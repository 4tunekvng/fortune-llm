/**
 * A provider's HTTP error, with its status kept as a number so the gateway
 * decides what a failure means from the status, not from the wording of
 * the message. The message keeps the old `<label> <status>: <body>` text,
 * so logs and the prior-errors header read the same as before.
 */
export class UpstreamError extends Error {
  constructor(
    readonly label: string,
    readonly status: number,
    readonly body: string,
    readonly retryAfterMs?: number,
  ) {
    super(`${label} ${status}: ${body.slice(0, 400)}`);
    this.name = "UpstreamError";
  }

  /** Builds the error from a failed upstream response, reading its body once. */
  static async from(label: string, resp: Response): Promise<UpstreamError> {
    const body = await resp.text().catch(() => "");
    const retryAfter = parseRetryAfter(resp.headers.get("retry-after"));
    return new UpstreamError(label, resp.status, body, retryAfter);
  }
}

/** Retry-After is either whole seconds or an HTTP date. */
function parseRetryAfter(value: string | null): number | undefined {
  if (!value) return undefined;
  const seconds = Number(value);
  if (Number.isFinite(seconds) && seconds >= 0) return seconds * 1000;
  const at = Date.parse(value);
  return Number.isNaN(at) ? undefined : Math.max(0, at - Date.now());
}
