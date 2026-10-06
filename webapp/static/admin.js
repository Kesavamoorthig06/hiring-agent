(() => {
  const $ = (id) => document.getElementById(id);
  $("theme").addEventListener("click", () => {
    const t = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = t; try { localStorage.setItem("ha-theme", t); } catch (e) {}
  });
  const api = async (path, body) => {
    const r = await fetch("/admin/api/" + path, { method: body === undefined ? "GET" : "POST", credentials: "same-origin",
      headers: body === undefined ? {} : { "Content-Type": "application/json" }, body: body ? JSON.stringify(body) : undefined });
    const j = await r.json().catch(() => ({}));
    return { ok: r.ok, status: r.status, j };
  };
  const say = (t, kind) => { const m = $("msg"); m.textContent = t || ""; m.className = "msg " + (kind || ""); };
  const when = (iso) => iso ? new Date(iso).toLocaleString() : "";
  function paint(s) {
    const map = { unset: ["", "No credentials set"], untested: ["mid", "Saved, not tested yet"], ok: ["ok", "Working"], invalid: ["bad", "Rejected or expired"] };
    const [cls, txt] = map[s.state] || ["", s.state];
    $("dot").className = "dot " + cls; $("stTxt").textContent = txt + (s.reason ? " (" + s.reason + ")" : "");
    const meta = $("stMeta");
    if (s.set) { meta.hidden = false; meta.textContent = "Region " + s.region + " | set " + when(s.set_at); } else meta.hidden = true;
  }
  async function refresh() {
    const r = await api("status");
    if (r.status === 404) { $("panel").hidden = true; $("gate").hidden = false; return false; }
    $("gate").hidden = true; $("panel").hidden = false; paint(r.j); return true;
  }
  $("gateForm").addEventListener("submit", async (e) => {
    e.preventDefault(); $("gateMsg").textContent = "";
    const r = await api("login", { passcode: $("pass").value });
    $("pass").value = "";
    if (!r.ok) { $("gateMsg").textContent = r.j.detail || "Could not sign in."; $("gateMsg").className = "msg bad"; return; }
    refresh();
  });
  $("credForm").addEventListener("submit", async (e) => {
    e.preventDefault(); say("Saving");
    const r = await api("creds", { access_key_id: $("ak").value, secret_access_key: $("sk").value, session_token: $("tk").value, region: $("rg").value, model_id: $("md").value });
    if (!r.ok) return say(r.j.detail || "Could not save.", "bad");
    ["ak", "sk", "tk"].forEach((k) => ($(k).value = ""));
    paint(r.j); say("Saved. Use Test to check them.", "good");
  });
  $("test").addEventListener("click", async () => {
    say("Calling Claude once");
    const r = await api("test", {});
    if (r.status === 404) return refresh();
    paint(r.j); say(r.j.message || "", r.j.ok ? "good" : "bad");
  });
  $("clear").addEventListener("click", async () => {
    const r = await api("clear", {}); if (r.status === 404) return refresh(); paint(r.j); say("Cleared.", "good");
  });
  refresh();

  // Parse a pasted credentials block (env exports, credentials file, JSON, plain key=value) into the fields.
  const KEYS = {
    ak: /^(aws_?)?access_?key(_?id)?$/i,
    sk: /^(aws_?)?secret(_?access)?_?key$/i,
    tk: /^(aws_?)?session_?token$/i,
    rg: /^(aws_?)?(default_?)?region$/i,
    md: /^(bedrock_?|claude_?)?model(_?id)?$/i,
  };
  function parseBlock(text) {
    const out = {};
    const re = /(?:^|[\s,{;])(?:export\s+|set\s+|\$Env:)?["']?([A-Za-z_][A-Za-z0-9_]*)["']?\s*[=:]\s*["']?([^\s"',;]+)/gm;
    let m;
    while ((m = re.exec(text))) {
      for (const k in KEYS) if (KEYS[k].test(m[1]) && !(k in out)) out[k] = m[2];
    }
    return out;
  }
  $("blk").addEventListener("input", () => {
    const v = $("blk").value;
    if (!v.trim()) { $("blkMsg").textContent = ""; return; }
    const got = parseBlock(v);
    const names = { ak: "access key", sk: "secret", tk: "session token", rg: "region", md: "model" };
    const found = Object.keys(got);
    if (!found.length) { $("blkMsg").textContent = "Nothing recognised in that block."; return; }
    found.forEach((k) => { $(k).value = got[k]; });
    $("blk").value = "";
    $("blkMsg").textContent = "Filled: " + found.map((k) => names[k]).join(", ") + ". Check the fields, then save.";
  });
})();
