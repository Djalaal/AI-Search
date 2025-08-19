/* search.js — combined core + results + thread, no chat logic */

// ---- Tiny DOM helpers ----
const $  = (sel, root=document) => root.querySelector(sel);
const $$ = (sel, root=document) => Array.from(root.querySelectorAll(sel));
const safe = (s="") => String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
const cutToWords = (text = "", maxWords = 100) => {
  const words = String(text).trim().split(/\s+/);
  return words.slice(0, maxWords).join(" ") + (words.length > maxWords ? "…" : "");
};
async function fetchJSON(url, options){
  const res = await fetch(url, options);
  if (!res.ok){
    const t = await res.text().catch(()=> "");
    throw new Error(t || `Request failed: ${res.status}`);
  }
  return res.json();
}
function setBlock(id, label, text) {
  const el = document.querySelector(id);
  if (!el) return;
  if (text) {
    el.classList.add("multiline");
    el.textContent = `${label}:\n${text}`;   // newline puts text on the next line
  } else {
    el.textContent = "";
  }
}

// ---- Routing / Views ----
const VIEWS = { HOME: "home", RESULTS: "results", THREAD: "thread" };
function show(view){
  $$(".view").forEach(v => v.classList.remove("active"));
  const el = $("#view-" + view);
  if (el) el.classList.add("active");
}
function gotoResults(q){
  const url = new URL(location.href);
  url.searchParams.set("mode", VIEWS.RESULTS);
  url.searchParams.set("q", q);
  history.pushState({}, "", url);
  showResults(q);
}
function gotoThread(id){
  const url = new URL(location.href);
  url.searchParams.set("mode", VIEWS.THREAD);
  url.searchParams.set("id", id);
  history.pushState({}, "", url);
  showThread(id);
}
async function routeFromURL(){
  const qs = new URLSearchParams(location.search);
  const mode = qs.get("mode");
  if (mode === VIEWS.RESULTS) {
    const q = qs.get("q") || "";
    await showResults(q);
  } else if (mode === VIEWS.THREAD) {
    const id = qs.get("id");
    await showThread(id);
  } else if ($("#view-home")) {
    show(VIEWS.HOME);
    const homeInput = $("#home-query");
    if (homeInput) homeInput.focus();
  }
}
window.addEventListener("popstate", routeFromURL);

// ---- Results view ----
async function showResults(query){
  if (!$("#view-results")) return;
  show(VIEWS.RESULTS);
  const qInput = $("#results-query");
  const summaryEl = $("#results-summary");
  const listEl = $("#results-list");
  if (qInput) qInput.value = query;
  if (summaryEl) summaryEl.textContent = "Searching…";
  if (listEl) listEl.innerHTML = "";

  try {
    const data = await fetchJSON("/lookup", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query })
    });
    const hits = Array.isArray(data.results) ? data.results : [];
    if (summaryEl) {
      summaryEl.textContent = hits.length
        ? `${hits.length} matching result${hits.length > 1 ? "s" : ""}`
        : "No matching results found.";
    }
    if (listEl){
      for (const r of hits) {
        const li = document.createElement("li");
        li.className = "result";
        
        if (r.thread_id) {
          const tid = r.thread_id;
          const subject = r.subject || `Thread ${tid}`;
          const topicLine = r.topic ? `${r.topic}` : "(No Topic)";
          const scorePill = (typeof r.score === "number")
            ? `<span class="pill">score: ${Math.round(Number(r.score))}%</span>`
            : "";
          const emailsCount = (typeof r.emails_count === "number")
            ? r.emails_count
            : (Array.isArray(r.emails) ? r.emails.length : undefined);
          const emailsPill = (typeof emailsCount === "number")
            ? `<span class="pill">${emailsCount} email${emailsCount === 1 ? "" : "s"}</span>`
            : "";

          li.innerHTML = `
            <a class="result-title"
              href="${location.origin}${location.pathname}?mode=thread&id=${encodeURIComponent(tid)}" target="_blank" rel="noopener">
              ${safe(subject)}
            </a>
            <div class="result-meta">
              <span class="pill">${safe(tid)}</span>
              ${scorePill}
              ${emailsPill}
            </div>
            <p class="result-snippet muted-text">${safe(cutToWords(topicLine, 60))}</p>
          `;
        } else if (r.doc_path) {
          const title = r.doc_name || (r.doc_path.split(/[\\/]/).pop());
          const scorePill = (typeof r.score === "number")
            ? `<span class="pill">score: ${Math.round(Number(r.score))}%</span>` : "";
          const pagePill = (r.page !== undefined && r.page !== null)
            ? `<span class="pill">page: ${String(r.page)}</span>` : "";

          // NEW: stash payload and link to /doc?id=...
          const docId = (crypto.randomUUID && crypto.randomUUID()) || Math.random().toString(36).slice(2);
          const payload = {
            doc_name: title,
            source: r.doc_path,
            page: r.page,
            paragraph_index: r.paragraph_index,
            paragraph: r.paragraph || ""
          };
          try { localStorage.setItem(`docview:${docId}`, JSON.stringify(payload)); } catch (_) {}

          li.innerHTML = `
            <a class="result-title"
              href="/doc?id=${encodeURIComponent(docId)}"
              target="_blank" rel="noopener">
              ${safe(title)}
            </a>
            <div class="result-meta">
              ${scorePill}
              ${pagePill}
            </div>
            <p class="result-snippet">${safe(cutToWords(r.paragraph || "", 80))}</p>
            <p class="muted-text" style="margin-top:4px">${safe(r.doc_path)}</p>
          `;
          listEl.appendChild(li);
        }
      }
    }
  } catch (err) {
    if (summaryEl) summaryEl.textContent = `Error: ${err.message || String(err)}`;
  }
}

// ---- Thread view ----
function pick(...vals){
  for (const v of vals) if (v !== undefined && v !== null && v !== "") return v;
  return undefined;
}
function formatRecipients(v){
  if (!v) return "";
  return Array.isArray(v) ? v.join(", ") : String(v);
}
function formatDateDisplay(s) {
  if (!s) return "No date";
  const d = new Date(s);
  if (Number.isNaN(d.getTime())) return s;
  return d.toLocaleString(undefined, {
    year: "numeric", month: "short", day: "2-digit",
    hour: "2-digit", minute: "2-digit"
  });
}
async function showThread(threadId){
  if (!$("#view-thread")) return;
  if (!threadId) {
    show(VIEWS.HOME);
    return;
  }
  show(VIEWS.THREAD);
  $("#emails-list").innerHTML = "";
  $("#thread-title").textContent = "Loading…";
  $("#thread-topic").textContent = "";
  $("#thread-detail1").textContent = "";
  $("#thread-detail2").textContent = "";

  let threadResp;
  try {
    threadResp = await fetchJSON(`/threads/${encodeURIComponent(threadId)}`);
  } catch {
    $("#thread-title").textContent = "Thread not found";
    return;
  }
  const { topic, detail1, detail2, email_ids = [], subject: subjectFromThread } = threadResp;
  const subject = subjectFromThread || topic || `Thread ${threadId}`;
  $("#thread-title").textContent = subject;
  setBlock("#thread-topic", "Topic", topic);
  setBlock("#thread-detail1", "Summary", detail1);
  //setBlock("#thread-detail2", "Details2", detail2);

  const emailsUl = $("#emails-list");
  for (const eid of email_ids) {
    const li = document.createElement("li");
    li.className = "result";
    li.innerHTML = `
      <details>
        <summary>
          <span class="email-header-line">Loading date…</span>
          <div class="email-meta"><span class="pill">${safe(eid)}</span></div>
        </summary>
        <div class="email-content">Loading…</div>
      </details>
    `;
    emailsUl.appendChild(li);
    try {
      const { data } = await fetchJSON(`/emails/${encodeURIComponent(eid)}`);
      const fromVal = pick(data.from, data.sender);
      const toVal   = pick(data.to, data.receiver);
      const ccVal   = pick(data.cc, data.ccs);
      const hdrs = [
        ["Subject", data.subject],
        ["From", fromVal],
        ["To", formatRecipients(toVal)],
        ["Cc", formatRecipients(ccVal)],
      ].filter(([k,v]) => v);
      const headerHTML = hdrs
        .map(([k,v]) => `<div class="kv"><span class="k">${k}</span><span class="v">${safe(String(v))}</span></div>`)
        .join("");
      const body = data.text_body || data.body || data.snippet || "";
      const attachments = Array.isArray(data.attachments) ? data.attachments : [];
      const attachHTML = attachments.length ? `
        <div class="attachments">
          <h4>Attachments (${attachments.length})</h4>
          <ul>${attachments.map(a => `<li>${safe(a.filename || a.name || "file")}</li>`).join("")}</ul>
        </div>` : "";
      const dateRaw = data.date || data.sent_at;
      const dateLabel = formatDateDisplay(dateRaw);
      const summaryEl = li.querySelector("summary");
      summaryEl.innerHTML = `
        <span class="email-header-line">${safe(dateLabel)}</span>
        <div class="email-meta"><span class="pill">${safe(eid)}</span></div>
      `;
      li.querySelector(".email-content").innerHTML = `
        <div class="email-headers">${headerHTML}</div>
        <pre class="email-body">${safe(String(body))}</pre>
        ${attachHTML}
      `;
    } catch {
      li.querySelector(".email-content").textContent = "Failed to load email.";
    }
  }
  const expandAll = document.getElementById("expand-all");
  const collapseAll = document.getElementById("collapse-all");
  if (expandAll) expandAll.onclick = () => $$("#emails-list details").forEach(d => d.open = true);
  if (collapseAll) collapseAll.onclick = () => $$("#emails-list details").forEach(d => d.open = false);
}

// ---- Bind forms ----
document.addEventListener("DOMContentLoaded", () => {
  const homeForm = document.getElementById("home-form");
  if (homeForm) {
    homeForm.addEventListener("submit", (e) => {
      e.preventDefault();
      const q = $("#home-query").value.trim();
      if (!q) return;
      gotoResults(q);
    });
  }
  const resultsForm = document.getElementById("results-form");
  if (resultsForm) {
    resultsForm.addEventListener("submit", (e) => {
      e.preventDefault();
      const q = $("#results-query").value.trim();
      if (!q) return;
      gotoResults(q);
    });
  }
  routeFromURL();
});
