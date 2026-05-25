#!/usr/bin/env bash
# Comprehensive HTTP-level test matrix for the live deployed gateway.
# Verifies every documented surface: endpoints, headers, routing
# overrides, cache semantics, stream-from-cache, error shapes. Run
# against prod; complements the SDK-level tests-integration/sdk-e2e.mjs.
#
# Exit 0 if everything passes; non-zero with a summary if any fails.

set -uo pipefail

GATEWAY="${GATEWAY_URL:-https://fortune-llm.fortunee.workers.dev}"
TOKEN="${GATEWAY_TOKEN:-KtfvUb0dLwGu7NSQxyKlvbkse2hFBvK1ZPf9RwDSIfo}"
RUN_TAG="run-$(date +%s)"

PASS=0
FAIL=0
FAIL_NAMES=()

green() { printf '\033[32m%s\033[0m\n' "$1"; }
red() { printf '\033[31m%s\033[0m\n' "$1"; }

pass() {
  local name="$1"; shift
  green "✓ $name"
  for line in "$@"; do echo "    $line"; done
  echo ""
  PASS=$((PASS + 1))
}

fail() {
  local name="$1"; local reason="$2"; shift 2
  red "✗ $name"
  echo "    REASON: $reason"
  for line in "$@"; do echo "    $line"; done
  echo ""
  FAIL=$((FAIL + 1))
  FAIL_NAMES+=("$name")
}

# Helper to extract a response header (case-insensitive).
hdr() {
  local file="$1"; local name="$2"
  grep -i "^${name}:" "$file" | head -1 | sed 's/^[^:]*: *//' | tr -d '\r'
}

post_messages() {
  local payload="$1"
  local out_dir="$2"
  curl -sS -i -X POST "$GATEWAY/v1/messages" \
    -H "x-api-key: $TOKEN" \
    -H "content-type: application/json" \
    --data "$payload" 2>/dev/null > "$out_dir/full"
  # Split headers and body.
  awk 'BEGIN{h=1} /^\r?$/ {h=0; next} h{print > "'"$out_dir"'/headers"; next} !h{print > "'"$out_dir"'/body"}' "$out_dir/full"
}

WORK=$(mktemp -d)
trap "rm -rf $WORK" EXIT

# ────────────────────────────────────────────────────────────────
# T1 — /healthz returns ok=true and lists configured backends
# ────────────────────────────────────────────────────────────────
T1=$WORK/t1; mkdir -p "$T1"
curl -sS "$GATEWAY/healthz" > "$T1/body"
ok=$(python3 -c "import sys,json;print(json.load(open('$T1/body')).get('ok'))" 2>/dev/null || echo "")
groq_cfg=$(python3 -c "import sys,json;print(json.load(open('$T1/body'))['backends'].get('groq'))" 2>/dev/null || echo "")
or_cfg=$(python3 -c "import sys,json;print(json.load(open('$T1/body'))['backends'].get('openrouter'))" 2>/dev/null || echo "")
if [[ "$ok" == "True" && "$groq_cfg" == "True" && "$or_cfg" == "True" ]]; then
  pass "T1 /healthz reports configured backends" "ok=$ok groq=$groq_cfg openrouter=$or_cfg"
else
  fail "T1 /healthz" "unexpected payload" "$(cat $T1/body | head -c 200)"
fi

# ────────────────────────────────────────────────────────────────
# T2 — /stats returns a sane shape
# ────────────────────────────────────────────────────────────────
T2=$WORK/t2; mkdir -p "$T2"
curl -sS "$GATEWAY/stats" > "$T2/body"
has_totals=$(python3 -c "import json;d=json.load(open('$T2/body'));print('requests' in d.get('totals',{}))" 2>/dev/null)
if [[ "$has_totals" == "True" ]]; then
  reqs=$(python3 -c "import json;print(json.load(open('$T2/body'))['totals']['requests'])" 2>/dev/null)
  hits=$(python3 -c "import json;print(json.load(open('$T2/body'))['totals']['cache_hits'])" 2>/dev/null)
  pass "T2 /stats returns daily counters" "requests=$reqs cache_hits=$hits"
else
  fail "T2 /stats" "missing totals.requests" "$(cat $T2/body | head -c 200)"
fi

# ────────────────────────────────────────────────────────────────
# T3 — Authentication: missing token → 401
# ────────────────────────────────────────────────────────────────
T3=$WORK/t3; mkdir -p "$T3"
curl -sS -i -X POST "$GATEWAY/v1/messages" \
  -H "content-type: application/json" \
  --data '{"model":"claude-sonnet-4-6","max_tokens":10,"messages":[{"role":"user","content":"x"}]}' > "$T3/full" 2>/dev/null
awk 'BEGIN{h=1} /^\r?$/ {h=0; next} h{print > "'"$T3"'/headers"; next} !h{print > "'"$T3"'/body"}' "$T3/full"
status=$(head -1 "$T3/full" | awk '{print $2}')
if [[ "$status" == "401" ]]; then
  pass "T3 auth: missing token → 401" "status=$status"
else
  fail "T3 auth missing token" "expected 401 got $status" "$(cat $T3/body | head -c 100)"
fi

# ────────────────────────────────────────────────────────────────
# T4 — Bad JSON body → 400
# ────────────────────────────────────────────────────────────────
T4=$WORK/t4; mkdir -p "$T4"
curl -sS -i -X POST "$GATEWAY/v1/messages" \
  -H "x-api-key: $TOKEN" -H "content-type: application/json" \
  --data 'this-is-not-json' > "$T4/full" 2>/dev/null
status=$(head -1 "$T4/full" | awk '{print $2}')
if [[ "$status" == "400" ]]; then
  pass "T4 invalid JSON → 400" "status=$status"
else
  fail "T4 invalid JSON" "expected 400 got $status"
fi

# ────────────────────────────────────────────────────────────────
# T5 — Missing required field (messages) → 400
# ────────────────────────────────────────────────────────────────
T5=$WORK/t5; mkdir -p "$T5"
curl -sS -i -X POST "$GATEWAY/v1/messages" \
  -H "x-api-key: $TOKEN" -H "content-type: application/json" \
  --data '{"model":"claude-sonnet-4-6","max_tokens":10}' > "$T5/full" 2>/dev/null
status=$(head -1 "$T5/full" | awk '{print $2}')
if [[ "$status" == "400" ]]; then
  pass "T5 missing messages[] → 400" "status=$status"
else
  fail "T5 missing messages" "expected 400 got $status"
fi

# ────────────────────────────────────────────────────────────────
# T6 — Default chain dispatches through the free chain
# ────────────────────────────────────────────────────────────────
T6=$WORK/t6; mkdir -p "$T6"
post_messages '{"model":"claude-sonnet-4-6","max_tokens":80,"messages":[{"role":"user","content":"Say hi."}],"metadata":{"fortune_no_cache":true}}' "$T6"
status=$(head -1 "$T6/full" | awk '{print $2}')
chain=$(hdr "$T6/headers" "x-fortune-llm-chain")
route=$(hdr "$T6/headers" "x-fortune-llm-route")
if [[ "$status" == "200" && -n "$chain" && -n "$route" ]]; then
  pass "T6 default chain dispatches" "status=$status chain=$chain route=$route"
else
  fail "T6 default chain" "status=$status no chain/route header" "$(cat $T6/body | head -c 200)"
fi

# ────────────────────────────────────────────────────────────────
# T7 — Cache: identical temp=0 prompt hits cache on 2nd call
# ────────────────────────────────────────────────────────────────
T7A=$WORK/t7a; T7B=$WORK/t7b; mkdir -p "$T7A" "$T7B"
payload="{\"model\":\"claude-sonnet-4-6\",\"max_tokens\":30,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"What is 11+12? Reply with just the number, then VERIFIED-$RUN_TAG-7.\"}]}"
post_messages "$payload" "$T7A"
cache_a=$(hdr "$T7A/headers" "x-fortune-llm-cache")
post_messages "$payload" "$T7B"
cache_b=$(hdr "$T7B/headers" "x-fortune-llm-cache")
if [[ "$cache_a" == "miss-stored" && "$cache_b" == "hit" ]]; then
  body_a=$(cat "$T7A/body" | head -c 200)
  pass "T7 cache: temp=0 same prompt twice → miss-stored then hit" "1st=$cache_a 2nd=$cache_b" "body=$body_a"
else
  fail "T7 cache temp=0" "expected miss-stored then hit, got 1st=$cache_a 2nd=$cache_b" "$(cat $T7A/headers | grep -i fortune-llm)"
fi

# ────────────────────────────────────────────────────────────────
# T8 — Cache opt-out: metadata.fortune_no_cache=true bypasses cache
# ────────────────────────────────────────────────────────────────
T8=$WORK/t8; mkdir -p "$T8"
payload="{\"model\":\"claude-sonnet-4-6\",\"max_tokens\":30,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"Say cache-bypass.\"}],\"metadata\":{\"fortune_no_cache\":true}}"
post_messages "$payload" "$T8"
cache_status=$(hdr "$T8/headers" "x-fortune-llm-cache")
if [[ -z "$cache_status" ]]; then
  pass "T8 fortune_no_cache → no cache header set" "(skipped path)"
else
  fail "T8 fortune_no_cache" "expected no cache header; got: $cache_status"
fi

# ────────────────────────────────────────────────────────────────
# T9 — Cache opt-in for non-zero temp: fortune_cache=true caches
# ────────────────────────────────────────────────────────────────
T9A=$WORK/t9a; T9B=$WORK/t9b; mkdir -p "$T9A" "$T9B"
payload="{\"model\":\"claude-sonnet-4-6\",\"max_tokens\":30,\"temperature\":0.7,\"messages\":[{\"role\":\"user\",\"content\":\"Say opt-in-cache-$RUN_TAG.\"}],\"metadata\":{\"fortune_cache\":true}}"
post_messages "$payload" "$T9A"
cache_a=$(hdr "$T9A/headers" "x-fortune-llm-cache")
post_messages "$payload" "$T9B"
cache_b=$(hdr "$T9B/headers" "x-fortune-llm-cache")
if [[ "$cache_a" == "miss-stored" && "$cache_b" == "hit" ]]; then
  pass "T9 fortune_cache=true caches even with temp>0" "1st=$cache_a 2nd=$cache_b"
else
  fail "T9 fortune_cache opt-in" "expected miss-stored then hit, got 1st=$cache_a 2nd=$cache_b"
fi

# ────────────────────────────────────────────────────────────────
# T10 — Force openrouter route works
# ────────────────────────────────────────────────────────────────
T10=$WORK/t10; mkdir -p "$T10"
payload='{"model":"claude-sonnet-4-6","max_tokens":30,"messages":[{"role":"user","content":"Say openrouter"}],"metadata":{"fortune_route":"openrouter","fortune_no_cache":true}}'
post_messages "$payload" "$T10"
route=$(hdr "$T10/headers" "x-fortune-llm-route")
if [[ "$route" == "openrouter" ]]; then
  pass "T10 force openrouter route" "route=$route"
else
  status=$(head -1 "$T10/full" | awk '{print $2}')
  fail "T10 force openrouter" "expected route=openrouter, got: $route (status $status)" "$(cat $T10/body | head -c 200)"
fi

# ────────────────────────────────────────────────────────────────
# T11 — Force anthropic route works
# ────────────────────────────────────────────────────────────────
T11=$WORK/t11; mkdir -p "$T11"
payload='{"model":"claude-sonnet-4-6","max_tokens":30,"messages":[{"role":"user","content":"Say anthropic"}],"metadata":{"fortune_route":"anthropic","fortune_no_cache":true}}'
post_messages "$payload" "$T11"
route=$(hdr "$T11/headers" "x-fortune-llm-route")
if [[ "$route" == "anthropic" ]]; then
  pass "T11 force anthropic route" "route=$route"
else
  fail "T11 force anthropic" "expected route=anthropic, got: $route"
fi

# ────────────────────────────────────────────────────────────────
# T12 — CORS preflight returns 204 with the right headers
# ────────────────────────────────────────────────────────────────
T12=$WORK/t12; mkdir -p "$T12"
curl -sS -i -X OPTIONS "$GATEWAY/v1/messages" -H "Origin: https://example.com" > "$T12/full" 2>/dev/null
status=$(head -1 "$T12/full" | awk '{print $2}')
cors=$(grep -i "^access-control-allow-origin:" "$T12/full" | head -1)
if [[ "$status" == "204" && -n "$cors" ]]; then
  pass "T12 CORS preflight" "status=$status $cors"
else
  fail "T12 CORS preflight" "expected 204+ACAO, got status=$status"
fi

# ────────────────────────────────────────────────────────────────
# T13 — 404 for unknown path
# ────────────────────────────────────────────────────────────────
T13=$WORK/t13; mkdir -p "$T13"
status=$(curl -sS -o /dev/null -w "%{http_code}" "$GATEWAY/nonexistent-path")
if [[ "$status" == "404" ]]; then
  pass "T13 unknown path → 404" "status=$status"
else
  fail "T13 unknown path" "expected 404 got $status"
fi

# ────────────────────────────────────────────────────────────────
# T14 — Stream from cache (the big new feature)
# ────────────────────────────────────────────────────────────────
T14A=$WORK/t14a; T14B=$WORK/t14b; mkdir -p "$T14A" "$T14B"
prompt="Reply in one sentence about photosynthesis, ending with VERIFIED-$RUN_TAG-14."
payload="{\"model\":\"claude-sonnet-4-6\",\"max_tokens\":80,\"temperature\":0,\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}]}"
post_messages "$payload" "$T14A"
cache_a=$(hdr "$T14A/headers" "x-fortune-llm-cache")
post_messages "$payload" "$T14B"
cache_b=$(hdr "$T14B/headers" "x-fortune-llm-cache")
# Both responses should be SSE (text/event-stream)
ct_a=$(hdr "$T14A/headers" "content-type")
ct_b=$(hdr "$T14B/headers" "content-type")
if [[ "$cache_a" == "miss-stored-stream" && "$cache_b" == "hit-stream" && "$ct_a" == *event-stream* && "$ct_b" == *event-stream* ]]; then
  # Verify SSE bodies contain the expected event names.
  has_message_start=$(grep -c "^event: message_start" "$T14B/body" 2>/dev/null || echo 0)
  has_content_block_delta=$(grep -c "^event: content_block_delta" "$T14B/body" 2>/dev/null || echo 0)
  if [[ "$has_message_start" -gt 0 && "$has_content_block_delta" -gt 0 ]]; then
    pass "T14 stream-from-cache" "1st=$cache_a 2nd=$cache_b 2nd-events: message_start=$has_message_start content_block_delta=$has_content_block_delta"
  else
    fail "T14 stream-from-cache (body)" "missing expected SSE events" "msg_start=$has_message_start delta=$has_content_block_delta"
  fi
else
  fail "T14 stream-from-cache" "expected miss-stored-stream then hit-stream, got 1st=$cache_a 2nd=$cache_b" "ct_a=$ct_a ct_b=$ct_b"
fi

# ────────────────────────────────────────────────────────────────
# T15 — Tool-using request emits a tool_use block
# ────────────────────────────────────────────────────────────────
T15=$WORK/t15; mkdir -p "$T15"
payload='{"model":"claude-sonnet-4-6","max_tokens":150,"messages":[{"role":"user","content":"Use the get_weather tool to check Paris."}],"tools":[{"name":"get_weather","description":"Get weather","input_schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}]}'
post_messages "$payload" "$T15"
status=$(head -1 "$T15/full" | awk '{print $2}')
has_tool=$(grep -c '"type":"tool_use"' "$T15/body" 2>/dev/null || echo 0)
if [[ "$status" == "200" && "$has_tool" -gt 0 ]]; then
  route=$(hdr "$T15/headers" "x-fortune-llm-route")
  pass "T15 tool-use round-trip" "status=$status route=$route tool_use_blocks=$has_tool"
else
  fail "T15 tool-use" "expected status 200 with tool_use, got status=$status tool_blocks=$has_tool" "$(cat $T15/body | head -c 200)"
fi

# ────────────────────────────────────────────────────────────────
# Summary
# ────────────────────────────────────────────────────────────────
echo "================================================================"
echo "PASSED: $PASS    FAILED: $FAIL"
if [[ $FAIL -gt 0 ]]; then
  echo ""
  echo "Failures:"
  for n in "${FAIL_NAMES[@]}"; do echo "  - $n"; done
  exit 1
fi
echo "All curl-matrix tests passed against $GATEWAY"
exit 0
