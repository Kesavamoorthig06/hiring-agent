(() => {
  const $ = (id) => document.getElementById(id);
  const show = (id) => ["upload", "progress", "results"].forEach((s) => ($(s).hidden = s !== id));
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  let state = null, history = [], busy = false;

  const drop = $("drop"), fileIn = $("file");
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileIn.click(); } });
  drop.addEventListener("click", (e) => { e.preventDefault(); fileIn.click(); });
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) start(f); });
  fileIn.addEventListener("change", () => { if (fileIn.files[0]) start(fileIn.files[0]); });
  $("reset").addEventListener("click", reset);

  function reset() {
    state = null; history = []; fileIn.value = ""; $("uploadErr").hidden = true;
    $("reset").hidden = true; $("log").innerHTML = ""; show("upload");
  }
  function fail(msg) { show("upload"); const e = $("uploadErr"); e.textContent = msg; e.hidden = false; fileIn.value = ""; }

  async function start(file) {
    $("uploadErr").hidden = true;
    if (!/\.pdf$/i.test(file.name) && file.type !== "application/pdf") return fail("Please upload a PDF file.");
    if (file.size > 5 * 1024 * 1024) return fail("That file is over 5 MB.");
    show("progress"); setStage("queued", "Uploading");
    const fd = new FormData(); fd.append("file", file);
    try {
      const r = await fetch("/api/analyze", { method: "POST", body: fd });
      if (!r.ok) return fail((await r.json().catch(() => ({}))).detail || "Upload failed.");
      poll((await r.json()).job_id);
    } catch { fail("Network error. Check your connection and try again."); }
  }

  const order = ["parse", "github", "score", "suggest"];
  function setStage(stage, label) {
    $("progLabel").textContent = label || "Working";
    const idx = order.indexOf(stage);
    document.querySelectorAll("#steps li").forEach((li) => {
      const i = order.indexOf(li.dataset.s);
      li.className = idx === -1 ? "" : i < idx ? "done" : i === idx ? "on" : "";
    });
  }
  async function poll(id) {
    for (let n = 0; n < 400; n++) {
      await new Promise((r) => setTimeout(r, 1500));
      let j;
      try { const r = await fetch("/api/jobs/" + id); if (!r.ok) return fail("This analysis expired. Please upload again."); j = await r.json(); }
      catch { continue; }
      if (j.status === "error") return fail(j.error || "Analysis failed.");
      if (j.status === "done") return render(j.result);
      setStage(j.stage, j.label);
    }
    fail("This is taking too long. Please try again.");
  }

  function render(res) {
    state = res; history = [];
    const ev = res.evaluation, sg = res.suggestions;
    $("candidate").textContent = res.candidate;
    $("headline").textContent = sg.headline;
    $("scoreNum").textContent = Math.round(ev.total);
    $("scoreMax").textContent = "of " + ev.max;
    $("ring").style.setProperty("--p", Math.min(100, (ev.total / ev.max) * 100));
    $("cats").innerHTML = ev.categories.map((c) =>
      `<div class="cat"><div class="cat-h"><span>${esc(c.label)}</span><span>${c.score}/${c.max}</span></div>
       <div class="bar"><i style="width:${(c.score / c.max) * 100}%"></i></div><p>${esc(c.evidence)}</p></div>`).join("");
    const bits = [];
    if (ev.bonus) bits.push("Bonus +" + ev.bonus);
    if (ev.deductions) bits.push("Deductions -" + ev.deductions + (ev.deduction_reasons ? " (" + ev.deduction_reasons + ")" : ""));
    $("bonusLine").textContent = bits.join(" · ");
    $("prios").innerHTML = sg.priorities.map((p) =>
      `<div class="prio"><div class="prio-h"><b>${esc(p.title)}</b><span class="tag ${esc(p.impact.toLowerCase())}">${esc(p.impact)}</span><span class="muted small">${esc(p.section)}</span></div>
       <p>${esc(p.why)}</p><p class="how">${esc(p.how)}</p></div>`).join("");
    $("rewriteCard").hidden = !sg.rewrites.length;
    $("rewrites").innerHTML = sg.rewrites.map((r) => `<div class="rw"><div class="b">${esc(r.before)}</div><div class="a">${esc(r.after)}</div></div>`).join("");
    $("strengths").innerHTML = ev.strengths.map((s) => `<li>${esc(s)}</li>`).join("");
    $("keywords").innerHTML = sg.missing_keywords.map((k) => `<span class="chip">${esc(k)}</span>`).join("") || '<span class="muted small">None flagged</span>';
    $("quick").innerHTML = ["What should I fix first?", "Rewrite my top project", "How do I get more open source signal?"]
      .map((q) => `<button type="button" class="chip">${q}</button>`).join("");
    $("quick").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => ask(b.textContent)));
    $("log").innerHTML = "";
    bot(`Hi ${res.candidate.split(" ")[0]}. Your score is ${Math.round(ev.total)} out of ${ev.max}. Ask me about any section, or tap a prompt below.`, false);
    $("reset").hidden = false; show("results"); window.scrollTo({ top: 0 });
  }

  function bot(text, record = true) {
    const d = document.createElement("div"); d.className = "m bot"; d.textContent = text; $("log").appendChild(d);
    $("log").scrollTop = $("log").scrollHeight; if (record) history.push({ role: "assistant", content: text }); return d;
  }
  async function ask(text) {
    if (busy || !state || !text.trim()) return;
    busy = true; $("send").disabled = true; $("quick").hidden = true;
    const me = document.createElement("div"); me.className = "m me"; me.textContent = text; $("log").appendChild(me);
    history.push({ role: "user", content: text });
    const t = bot("Thinking…", false); t.classList.add("typing");
    try {
      const r = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: history, resume_text: state.resume_text, evaluation: state.evaluation, suggestions: state.suggestions }) });
      const j = await r.json();
      t.remove();
      if (!r.ok) { history.pop(); bot(j.detail || "Something went wrong. Try again.", false); }
      else bot(j.reply);
    } catch { t.remove(); history.pop(); bot("Network error. Try again.", false); }
    busy = false; $("send").disabled = false; $("msg").focus();
  }
  $("chatForm").addEventListener("submit", (e) => { e.preventDefault(); const v = $("msg").value; $("msg").value = ""; ask(v); });
})();
