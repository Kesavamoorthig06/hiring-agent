"""Claude on Amazon Bedrock, with credentials held in memory only.

Credentials are pasted into the admin page and live in this process. They are
never written to disk, never logged and never returned by any endpoint. When
Bedrock rejects them (expired or invalid), the provider switches itself off and
the site offers Gemini only until fresh credentials are set.
"""

import json
import re
import threading
import time
from datetime import datetime, timezone

CONCURRENCY = 6
_slots = threading.BoundedSemaphore(CONCURRENCY)
_mu = threading.Lock()
_creds: dict = {}
_meta = {"set_at": None, "state": "unset", "reason": "", "changed_at": None}

AUTH_CODES = {
    "ExpiredTokenException",
    "ExpiredToken",
    "UnrecognizedClientException",
    "InvalidSignatureException",
    "InvalidClientTokenId",
    "SignatureDoesNotMatch",
    "AccessDeniedException",
    "AuthFailure",
    "RequestExpired",
    "NotAuthorizedException",
}


class ClaudeUnavailable(Exception):
    """Claude cannot be used right now. The message is safe to show."""


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _set_state(state, reason=""):
    with _mu:
        if _meta["state"] != state:
            _meta["changed_at"] = _now()
        _meta["state"] = state
        _meta["reason"] = reason


def available():
    with _mu:
        return bool(_creds) and _meta["state"] in ("untested", "ok")


def status():
    with _mu:
        return {
            "set": bool(_creds),
            "state": _meta["state"],
            "reason": _meta["reason"],
            "region": _creds.get("region"),
            "model_id": _creds.get("model_id"),
            "set_at": _meta["set_at"],
            "changed_at": _meta["changed_at"],
            "probe": list(_meta.get("probe", [])),
        }


def set_creds(access_key_id, secret_access_key, session_token, region, model_id):
    with _mu:
        _creds.clear()
        _creds.update(
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            session_token=session_token or None,
            region=region,
            model_id=model_id,
        )
        _meta["set_at"] = _now()
    _set_state("untested", "")


def clear():
    with _mu:
        _creds.clear()
        _meta["set_at"] = None
    _set_state("unset", "")


def validate(access_key_id, secret_access_key, session_token, region, model_id):
    if not re.fullmatch(r"[A-Z0-9]{16,40}", access_key_id or ""):
        return "Access key ID looks wrong."
    if not (20 <= len(secret_access_key or "") <= 100) or re.search(r"\s", secret_access_key):
        return "Secret access key looks wrong."
    if session_token and (len(session_token) > 4096 or re.search(r"\s", session_token)):
        return "Session token looks wrong."
    if not re.fullmatch(r"[a-z]{2}(-[a-z]+)+-\d", region or ""):
        return "Region looks wrong."
    if model_id and not re.fullmatch(r"[A-Za-z0-9._:/-]{5,200}", model_id):
        return "Model ID looks wrong."
    return ""


def _client():
    import boto3
    from botocore.config import Config

    with _mu:
        c = dict(_creds)
    if not c:
        raise ClaudeUnavailable("Claude is temporarily unavailable.")
    return boto3.client(
        "bedrock-runtime",
        region_name=c["region"],
        aws_access_key_id=c["access_key_id"],
        aws_secret_access_key=c["secret_access_key"],
        aws_session_token=c["session_token"],
        config=Config(
            connect_timeout=5,
            read_timeout=90,
            retries={"max_attempts": 1},
        ),
    ), c["model_id"]


def _convert(messages):
    system, turns = [], []
    for m in messages:
        role, text = m.get("role"), str(m.get("content", ""))
        if role == "system":
            system.append({"text": text})
            continue
        role = "assistant" if role == "assistant" else "user"
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"][0]["text"] += "\n\n" + text
        else:
            turns.append({"role": role, "content": [{"text": text}]})
    if not turns or turns[0]["role"] != "user":
        turns.insert(0, {"role": "user", "content": [{"text": "Begin."}]})
    return system, turns


_last = {"code": "", "msg": ""}


def _note(code, msg=""):
    import re
    _last["code"] = code
    _last["msg"] = re.sub(r"[A-Za-z0-9/+=_\-]{32,}", "[redacted]", msg or "")[:300]


def _invoke(messages, options=None, schema=None, max_tokens=4096):
    from botocore.exceptions import BotoCoreError, ClientError

    options = options or {}
    system, turns = _convert(messages)
    client, model_id = _client()
    args = {
        "modelId": model_id,
        "messages": turns,
        "inferenceConfig": {"maxTokens": max_tokens},
    }
    if "temperature" in options:
        args["inferenceConfig"]["temperature"] = float(options["temperature"])
    if system:
        args["system"] = system
    if schema:
        args["toolConfig"] = {
            "tools": [
                {
                    "toolSpec": {
                        "name": "respond",
                        "description": "Return the answer as structured JSON.",
                        "inputSchema": {"json": schema},
                    }
                }
            ],
            "toolChoice": {"tool": {"name": "respond"}},
        }
    with _slots:
        for attempt in range(3):
            t0 = time.monotonic()
            try:
                resp = client.converse(**args)
                break
            except ClientError as exc:
                code = exc.response.get("Error", {}).get("Code", "")
                print(f"[claude] error={code} after {time.monotonic() - t0:.1f}s", flush=True)
                _note(code, exc.response.get("Error", {}).get("Message", ""))
                if code in AUTH_CODES:
                    _set_state("invalid", code)
                    raise ClaudeUnavailable("Claude is temporarily unavailable.")
                if code in ("ThrottlingException", "ServiceUnavailableException", "ModelNotReadyException") and attempt < 2:
                    time.sleep(2 + attempt * 3)
                    continue
                raise ClaudeUnavailable("Claude could not answer just now. Please try again.")
            except BotoCoreError as exc:
                print(f"[claude] error={type(exc).__name__} after {time.monotonic() - t0:.1f}s", flush=True)
                _note(type(exc).__name__, str(exc))
                if attempt < 2:
                    time.sleep(2)
                    continue
                raise ClaudeUnavailable("Claude could not answer just now. Please try again.")
    _set_state("ok", "")
    blocks = resp.get("output", {}).get("message", {}).get("content", [])
    for b in blocks:
        if "toolUse" in b:
            return json.dumps(b["toolUse"].get("input", {}))
    return "".join(b.get("text", "") for b in blocks)


def chat(messages, options=None, schema=None):
    """Same return shape as the repo's provider: {"message": {"content": str}}."""
    return {"message": {"content": _invoke(messages, options, schema)}}


def _explain(code, msg):
    m = (msg or "").lower()
    if code in ("ExpiredTokenException", "ExpiredToken") or "expired" in m:
        return "Credentials expired. The lab session token is no longer valid; paste a fresh block."
    if code in ("UnrecognizedClientException", "InvalidSignatureException", "InvalidClientTokenId", "SignatureDoesNotMatch") or "security token" in m:
        return "Credentials rejected. The access key, secret or session token is wrong or does not belong together."
    if code == "AccessDeniedException":
        return "Access denied. These credentials are valid but not allowed to call this model (model access not enabled, or the inference profile is blocked)."
    if code == "ResourceNotFoundException" or "model identifier is invalid" in m or "provided model identifier" in m:
        return "Model not found. Check the Model ID, and that it exists in this region (use the us./eu. inference profile ID if needed)."
    if code == "ValidationException":
        return "Model ID or request rejected: " + (msg or "")[:200]
    if code in ("EndpointConnectionError", "ConnectTimeoutError", "ConnectionClosedError") or "could not connect" in m:
        return "Could not reach Bedrock in this region. Check the region value."
    if code in ("ThrottlingException", "ServiceUnavailableException", "ModelNotReadyException", "ModelTimeoutException"):
        return "Bedrock is throttled or busy right now (" + code + "). Credentials look fine; try again."
    return "Bedrock error " + (code or "unknown") + ((": " + msg[:160]) if msg else "") + "."


def test_invoke():
    """One tiny call to check the credentials. Returns (ok, message)."""
    _last["code"] = ""
    _last["msg"] = ""
    try:
        _invoke([{"role": "user", "content": "Reply with the single word ok."}], {"temperature": 0}, None, 8)
        return True, "Claude answered."
    except ClaudeUnavailable:
        return False, _explain(_last["code"], _last["msg"])
    except Exception as exc:
        return False, "Test failed (" + type(exc).__name__ + ")."


_FAMILY = {"sonnet": 3, "haiku": 2, "opus": 1}


def _rank(mid):
    """Higher is better: Sonnet-class first, then newer version, then newer date, then profile ids."""
    fam = next((v for k, v in _FAMILY.items() if k in mid), 0)
    m = re.search(r"claude-(?:(?:opus|sonnet|haiku)-)?(\d+)(?:-(\d+))?(?!\d)", mid)
    ver = (int(m.group(1)), int(m.group(2)) if m and m.group(2) and len(m.group(2)) < 3 else 0) if m else (0, 0)
    d = re.search(r"-(20\d{6})-", mid)
    date = int(d.group(1)) if d else 0
    is_profile = 1 if re.match(r"^(us|eu|apac|global|us-gov)\.", mid) else 0
    return (fam, ver, date, is_profile)


def discover():
    """List Claude models the saved credentials can see, try the best few, keep the first that answers.
    Returns (model_id or "", message)."""
    import boto3
    from botocore.config import Config
    from botocore.exceptions import BotoCoreError, ClientError

    with _mu:
        c = dict(_creds)
    if not c:
        return "", "No credentials saved."
    cl = boto3.client(
        "bedrock", region_name=c["region"], aws_access_key_id=c["access_key_id"],
        aws_secret_access_key=c["secret_access_key"], aws_session_token=c["session_token"],
        config=Config(connect_timeout=5, read_timeout=30, retries={"max_attempts": 1}),
    )
    cands = set()
    try:
        for prof in cl.list_inference_profiles(maxResults=100).get("inferenceProfileSummaries", []):
            pid = prof.get("inferenceProfileId", "")
            if "anthropic.claude" in pid and prof.get("status", "ACTIVE") == "ACTIVE":
                cands.add(pid)
        for mdl in cl.list_foundation_models(byProvider="Anthropic").get("modelSummaries", []):
            mid = mdl.get("modelId", "")
            if "ON_DEMAND" in (mdl.get("inferenceTypesSupported") or []) and "anthropic.claude" in mid:
                cands.add(mid)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        _note(code, exc.response.get("Error", {}).get("Message", ""))
        return "", _explain(code, _last["msg"])
    except BotoCoreError as exc:
        _note(type(exc).__name__, str(exc))
        return "", _explain(type(exc).__name__, str(exc))
    # skip embedding/instant/legacy text models that Converse cannot drive
    cands = [m for m in cands if not re.search(r"instant|v1(?!:)|embed|claude-v2", m)]
    ordered = sorted(cands, key=_rank, reverse=True)
    if not ordered:
        return "", "No Claude models visible to these credentials in " + c["region"] + "."
    # Claude Code's own default model ids first, then everything else by rank
    known = [
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "us.anthropic.claude-sonnet-4-20250514-v1:0",
        "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "global.anthropic.claude-haiku-4-5-20251001-v1:0",
        "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        "us.anthropic.claude-opus-4-1-20250805-v1:0",
    ]
    order = known + [m for m in ordered if m not in known]
    probe = []
    with _mu:
        _meta["probe"] = probe
    for mid in order[:60]:
        with _mu:
            _creds["model_id"] = mid
        _last["code"] = ""
        _last["msg"] = ""
        _set_state("untested", "")
        ok, msg = test_invoke()
        probe.append({"model": mid, "ok": ok, "code": _last["code"], "msg": _last["msg"][:160]})
        if ok:
            return mid, "Picked " + mid + " (tried " + str(len(probe)) + ", API: Converse)."
        if _last["code"] in ("ExpiredTokenException", "ExpiredToken", "UnrecognizedClientException", "InvalidSignatureException", "InvalidClientTokenId", "SignatureDoesNotMatch"):
            break
    with _mu:
        _creds["model_id"] = ""
    _set_state("untested", "")
    from collections import Counter
    cnt = Counter(p["code"] or "other" for p in probe)
    return "", "Tried " + str(len(probe)) + " Claude ids with Converse, none answered: " + ", ".join(k + " x" + str(v) for k, v in cnt.items()) + "."
