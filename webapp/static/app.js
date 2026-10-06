(() => {
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  let state = null, history = [], busy = false;

  /* ---------- theme ---------- */
  $("theme").addEventListener("click", () => {
    const t = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = t;
    try { localStorage.setItem("ha-theme", t); } catch (e) {}
  });

  /* ---------- views ---------- */
  const views = { home: $("home"), progress: $("progress"), results: $("results") };
  function show(name) {
    Object.entries(views).forEach(([k, el]) => (el.hidden = k !== name));
    $("links").hidden = name !== "home";
    $("reset").hidden = name !== "results";
    $("fab").hidden = name !== "results";
    window.scrollTo({ top: 0 });
    observeReveals();
  }

  /* ---------- reveal on scroll ---------- */
  const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } }), { threshold: 0.12 });
  function observeReveals() {
    document.querySelectorAll(".rv:not(.in)").forEach((el) => { if (!el.closest("[hidden]")) io.observe(el); });
  }
  observeReveals();

  /* ---------- hero tilt + tile spotlight ---------- */
  const scene = $("scene");
  if (scene && !reduce) {
    const host = scene.parentElement;
    host.addEventListener("pointermove", (e) => {
      const r = host.getBoundingClientRect();
      const x = (e.clientX - r.left) / r.width - 0.5, y = (e.clientY - r.top) / r.height - 0.5;
      scene.style.setProperty("--ry", (-14 + x * 22).toFixed(1) + "deg");
      scene.style.setProperty("--rx", (8 - y * 18).toFixed(1) + "deg");
    });
    host.addEventListener("pointerleave", () => { scene.style.removeProperty("--ry"); scene.style.removeProperty("--rx"); });
  }
  document.querySelectorAll(".spot").forEach((t) => t.addEventListener("pointermove", (e) => {
    const r = t.getBoundingClientRect();
    t.style.setProperty("--mx", e.clientX - r.left + "px"); t.style.setProperty("--my", e.clientY - r.top + "px");
  }));

  /* ---------- scroll-driven "how it works" ---------- */
  const stepEls = [...document.querySelectorAll(".step")];
  function onScroll() {
    if ($("home").hidden || !stepEls.length) return;
    const mid = innerHeight * 0.55;
    let act = -1;
    stepEls.forEach((s, i) => { if (s.getBoundingClientRect().top < mid) act = i; });
    stepEls.forEach((s, i) => s.classList.toggle("on", i <= act && i === act));
    const first = stepEls[0].getBoundingClientRect().top, last = stepEls[stepEls.length - 1].getBoundingClientRect().bottom;
    const p = Math.min(1, Math.max(0, (mid - first) / Math.max(1, last - first)));
    $("railFill").parentElement.style.setProperty("--p", p); $("railFill").style.setProperty("--p", p);
  }
  addEventListener("scroll", onScroll, { passive: true }); onScroll();

  /* ---------- model picker ---------- */
  let provider = "gemini", sessionKey = "";
  const segs = document.querySelectorAll(".seg-b");
  function pick(p) {
    provider = p;
    segs.forEach((b) => { const on = b.dataset.p === p; b.classList.toggle("on", on); b.setAttribute("aria-checked", on); });
    $("ownBox").hidden = p !== "gemini";
  }
  segs.forEach((b) => b.addEventListener("click", () => { if (!b.disabled) pick(b.dataset.p); }));
  async function loadProviders() {
    try {
      const r = await fetch("/api/providers"); const j = await r.json();
      const ok = !!(j.claude && j.claude.available);
      $("segClaude").disabled = !ok; $("claudeNote").hidden = ok;
      if (!ok && provider === "claude") pick("gemini");
    } catch (e) {}
  }
  loadProviders(); setInterval(loadProviders, 60000);

  /* ---------- upload ---------- */
  const drop = $("drop"), fileIn = $("file");
  const pickFile = () => fileIn.click();
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pickFile(); } });
  drop.addEventListener("click", (e) => { if (e.target === fileIn) return; pickFile(); });
  fileIn.addEventListener("click", (e) => e.stopPropagation());
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => { const f = e.dataTransfer && e.dataTransfer.files[0]; if (f) start(f); });
  ["dragover", "drop"].forEach((ev) => addEventListener(ev, (e) => { if (!drop.contains(e.target)) e.preventDefault(); }));
  fileIn.addEventListener("change", () => { if (fileIn.files[0]) start(fileIn.files[0]); });
  $("ctaBtn").addEventListener("click", () => { $("upload").scrollIntoView({ behavior: "smooth" }); setTimeout(pickFile, 500); });
  $("brand").addEventListener("click", (e) => { e.preventDefault(); if (!$("home").hidden) window.scrollTo({ top: 0, behavior: "smooth" }); else reset(); });
  $("reset").addEventListener("click", reset);
  $("fab").addEventListener("click", () => $("chatCol").scrollIntoView({ behavior: "smooth", block: "start" }));

  function reset() {
    state = null; history = []; sessionKey = ""; $("ownKey").value = ""; fileIn.value = ""; $("uploadErr").hidden = true; $("log").innerHTML = "";
    $("ringP").style.strokeDashoffset = 377; show("home");
  }
  function fail(msg) { show("home"); const e = $("uploadErr"); e.textContent = msg; e.hidden = false; fileIn.value = ""; }

  async function start(file) {
    $("uploadErr").hidden = true;
    if (!/\.pdf$/i.test(file.name) && file.type !== "application/pdf") return fail("Please upload a PDF file.");
    if (file.size > 5 * 1024 * 1024) return fail("That file is over 5 MB.");
    show("progress"); setStage("parse", "Reading your resume");
    const fd = new FormData(); fd.append("file", file); fd.append("provider", provider);
    sessionKey = provider === "gemini" ? $("ownKey").value.trim() : "";
    if (sessionKey) fd.append("own_key", sessionKey);
    $("ownKey").value = "";
    try {
      const r = await fetch("/api/analyze", { method: "POST", body: fd });
      if (!r.ok) { loadProviders(); return fail((await r.json().catch(() => ({}))).detail || "Upload failed."); }
      const first = await r.json();
      poll(first.job_id);
    } catch { fail("Network error. Check your connection and try again."); }
  }

  const order = ["parse", "github", "score", "suggest"];
  function setStage(stage, label, detail) {
    $("progLabel").textContent = label || "Working";
    const d = $("progDetail"); d.textContent = detail || ""; d.hidden = !detail;
    const idx = order.indexOf(stage);
    document.querySelectorAll("#steps li").forEach((li) => {
      const i = order.indexOf(li.dataset.s);
      li.className = idx === -1 ? "" : i < idx ? "done" : i === idx ? "on" : "";
    });
  }
  function fmtEta(s) {
    if (s < 45) return "under a minute";
    const m = Math.round(s / 60); return m < 60 ? "~" + m + " min" : "~" + Math.floor(m / 60) + " h " + (m % 60) + " min";
  }
  function setQueue(q) {
    const box = $("queueBox");
    if (!q || !q.position) { box.hidden = true; return; }
    box.hidden = false;
    $("qPos").textContent = "#" + q.position;
    $("qEta").textContent = fmtEta(q.eta_seconds);
    const n = Math.min(q.queue_size || q.position, 24);
    $("qTrack").innerHTML = Array.from({ length: n }, (_, i) => `<i class="${i === 0 ? "run" : ""} ${i + 1 === Math.min(q.position, n) ? "you" : ""}"></i>`).join("");
    $("progLabel").textContent = q.position === 1 ? "You are next" : (q.position - 1) + " ahead of you";
  }
  async function poll(id) {
    for (let n = 0; n < 1800; n++) {
      await new Promise((r) => setTimeout(r, 1200));
      let j;
      try { const r = await fetch("/api/jobs/" + id); if (!r.ok) return fail("This analysis expired. Please upload again."); j = await r.json(); }
      catch { continue; }
      if (j.status === "error") return fail(j.error || "Analysis failed.");
      if (j.status === "done") return render(j.result);
      setStage(j.stage, j.label, j.detail);
    }
    fail("This is taking too long. Please try again.");
  }

  /* ---------- results ---------- */
  const CAT_ICON = { open_source: "i-git", self_projects: "i-code", production: "i-box", technical_skills: "i-wrench", ai_fluency: "i-brain" };
  const icon = (id) => `<svg class="ic"><use href="#${id}"/></svg>`;
  function countUp(el, to) {
    if (reduce) { el.textContent = Math.round(to); return; }
    const t0 = performance.now();
    (function f(t) { const p = Math.min(1, (t - t0) / 1600); el.textContent = Math.round(to * (1 - Math.pow(1 - p, 3))); if (p < 1) requestAnimationFrame(f); })(t0);
  }
  function tierOf(p) { return p >= 0.8 ? "Standout" : p >= 0.6 ? "Strong" : p >= 0.4 ? "Solid base" : p >= 0.2 ? "Early stage" : "Just starting"; }

  function render(res) {
    state = res; history = [];
    const ev = res.evaluation, sg = res.suggestions;
    const pct = Math.min(1, ev.total / ev.max);
    $("candidate").textContent = res.candidate;
    $("headline").textContent = sg.headline;
    $("tier").textContent = tierOf(pct);
    $("scoreMax").textContent = "of " + ev.max;
    const stats = [];
    if (ev.bonus) stats.push(`<span class="stat g">Bonus +${esc(ev.bonus)}</span>`);
    if (ev.deductions) stats.push(`<span class="stat r" title="${esc(ev.deduction_reasons || "")}">Deductions -${esc(ev.deductions)}</span>`);
    $("stats").innerHTML = stats.join("");
    $("cats").innerHTML = ev.categories.map((c) =>
      `<div class="cat"><div class="cat-h"><span>${icon(CAT_ICON[c.key] || "i-gauge")}${esc(c.label)}</span><em>${esc(c.score)}/${esc(c.max)}</em></div>
       <div class="bar"><i data-w="${(c.score / c.max) * 100}"></i></div><p>${esc(c.evidence)}</p></div>`).join("");
    $("prios").innerHTML = sg.priorities.map((p, i) =>
      `<div class="prio"><span class="num">${String(i + 1).padStart(2, "0")}</span><div><div class="prio-h"><b>${esc(p.title)}</b><span class="tag ${esc(String(p.impact).toLowerCase())}">${esc(p.impact)}</span><span class="muted small">${esc(p.section)}</span></div>
       <p>${esc(p.why)}</p><p class="fix">${esc(p.how)}</p></div></div>`).join("");
    $("rewriteCard").hidden = !sg.rewrites.length;
    $("rewrites").innerHTML = sg.rewrites.map((r) => `<div class="rw"><div class="b">${esc(r.before)}</div><div class="a">${esc(r.after)}</div></div>`).join("");
    $("strengths").innerHTML = ev.strengths.map((s) => `<li>${esc(s)}</li>`).join("");
    $("keywords").innerHTML = sg.missing_keywords.map((k) => `<span class="chip">${esc(k)}</span>`).join("") || '<span class="muted small">None flagged</span>';
    $("quick").innerHTML = ["What should I fix first?", "Rewrite my top project", "How do I get more open source signal?"]
      .map((q) => `<button type="button" class="chip">${q}</button>`).join("");
    $("quick").hidden = false;
    $("quick").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => ask(b.textContent)));
    $("log").innerHTML = "";
    bot(`Hi ${res.candidate.split(" ")[0]}. Your score is ${Math.round(ev.total)} out of ${ev.max}. Ask me about any section, or tap a prompt below.`, false);
    show("results");
    setTimeout(() => {
      $("ringP").style.strokeDashoffset = 377 * (1 - pct);
      countUp($("scoreNum"), ev.total);
      document.querySelectorAll("#cats .bar i").forEach((b) => (b.style.width = b.dataset.w + "%"));
    }, 150);
    setTimeout(() => { $("scoreNum").textContent = Math.round(ev.total); }, 2200);
  }

  function bot(text, record = true) {
    const d = document.createElement("div"); d.className = "m bot"; d.textContent = text; $("log").appendChild(d);
    $("log").scrollTop = $("log").scrollHeight; if (record) history.push({ role: "assistant", content: text }); return d;
  }
  async function ask(text) {
    if (busy || !state || !text.trim()) return;
    busy = true; $("send").disabled = true; $("quick").hidden = true;
    const me = document.createElement("div"); me.className = "m me"; me.textContent = text; $("log").appendChild(me);
    $("log").scrollTop = $("log").scrollHeight;
    history.push({ role: "user", content: text });
    const t = bot("Thinking…", false); t.classList.add("typing");
    try {
      const r = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider, own_key: sessionKey || null, messages: history, resume_text: state.resume_text, evaluation: state.evaluation, suggestions: state.suggestions }) });
      const j = await r.json();
      t.remove();
      if (!r.ok) { history.pop(); bot(j.detail || "Something went wrong. Try again.", false); }
      else bot(j.reply);
    } catch { t.remove(); history.pop(); bot("Network error. Try again.", false); }
    busy = false; $("send").disabled = false; $("msg").focus({ preventScroll: true });
  }
  $("chatForm").addEventListener("submit", (e) => { e.preventDefault(); const v = $("msg").value; $("msg").value = ""; ask(v); });
})();
