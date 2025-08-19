const CONTEXT_MODE = "summary";  // Global context mode: "summary", "full", or "hybrid"
const $ = (sel, root=document) => root.querySelector(sel);
const safe = (s="") => String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");

function threadUrl(threadId){
  return `${location.origin}/search?mode=thread&id=${encodeURIComponent(threadId)}`;
}

function addUserMessage(text){
  const wrap = $("#chat-messages");
  const item = document.createElement("div");
  item.className = "msg user";
  item.innerHTML = `<div class="avatar"></div><div class="bubble">${safe(text)}</div>`;
  wrap.appendChild(item);
  wrap.scrollTop = wrap.scrollHeight;
}

function addAssistantMessage(html, thinking=false){
  const wrap = $("#chat-messages");
  const item = document.createElement("div");
  item.className = "msg assistant";
  item.innerHTML = `
    <div class="avatar"></div>
    <div class="bubble">
      <div class="meta-line muted-text">Vetus Assistant</div>
      <div class="content">${thinking ? `<span class="typing" aria-label="Thinking"></span>` : (html || "")}</div>
    </div>`;
  wrap.appendChild(item);
  wrap.scrollTop = wrap.scrollHeight;
  return item;
}

function updateAssistantMessage(container, {html}){
  const content = container.querySelector(".content");
  if (html !== undefined) content.innerHTML = html;
  const wrap = $("#chat-messages");
  wrap.scrollTop = wrap.scrollHeight;
}

async function fetchJSON(url, options){
  const res = await fetch(url, options);
  if (!res.ok){
    const t = await res.text().catch(()=> "");
    throw new Error(t || `Request failed: ${res.status}`);
  }
  return res.json();
}

function buildRewritePrompt(query){
  return [
    "USER QUESTION:",
    query,
    "",
    "INSTRUCTIONS:",
    "- Rewrite the USER QUESTION above into a single, concise embedding search query.",
    "- Drop filler and verbose statements. Output only the query, no quotes."
  ].join("\n");
}

function stripWrapperQuotes(s){
  // Remove simple surrounding quotes/backticks if the model includes them
  const m = s.match(/^['"`](.*)['"`]$/);
  return m ? m[1] : s;
}

function getSelectedContextMode(){
  // Prefer a <select id="context-mode"> if present (values: "summary" | "full" | "hybrid")
  const sel = document.getElementById("context-mode");
  if (sel && sel.value) return sel.value;
  // Fallback to a checkbox <input id="use-full-emails">
  const cb = document.getElementById("use-full-emails");
  if (cb) return cb.checked ? "full" : "summary";
  return "summary";
}

document.addEventListener("DOMContentLoaded", () => {
  const form = $("#chat-form");
  const input = $("#chat-input");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const val = input.value.trim();
    if (!val) return;
    input.value = "";

    addUserMessage(val);
    const thinkingEl = addAssistantMessage("", true);

     // Show immediate placeholder with the raw input
    const content0 = `<span class="typing" aria-label="Thinking"></span>`;
    thinkingEl.querySelector(".content").innerHTML = content0;

    try {
      // 1) Rewrite first via /ask so we can show the actual query immediately
      const rewritePrompt = buildRewritePrompt(val);
      let rewriteResp;
      try {
        rewriteResp = await fetchJSON("/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            user_input: rewritePrompt,
            model_name: "deepseek-r1-12k:8b",
            use_chat: false
          })
        });
      } catch (_e) {
        // Fallback to original input if rewrite fails
        rewriteResp = { answer: val };
      }
      const used = stripWrapperQuotes(String(rewriteResp.answer || "").trim()) || val;
      thinkingEl.querySelector(".meta-line").innerHTML =
        `Vetus Assistant • Query:<code>\n${safe(used)}</code>`;

      // 2) Now do retrieval+answer using the known query (no internal rewrite)
      // const context_mode = getSelectedContextMode();
      const context_mode = CONTEXT_MODE;  // Use global context mode
      const payload = { user_input: val, query: used, context_mode };
      const resp = await fetchJSON("/ask-with-rag", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      console.debug("🔎 Embedding search term used:", resp.used_query);

      const sources = Array.isArray(resp.sources) ? resp.sources : [];
      const answer = String(resp.answer || "").trim();

      const sourcesHTML = sources.length ? `
        <hr/>
        <div class="sources">
          <strong>Sources</strong>
          <div class="pill-list" style="display: flex; flex-direction: column; gap: 4px;">
            ${sources.map(s => {
              if (!s.thread_id && (s.doc_path || s.paragraph)) {
                // DOC: build client-side link
                const id = (crypto.randomUUID && crypto.randomUUID()) || Math.random().toString(36).slice(2);
                const payload = {
                  doc_name: s.doc_name || (s.doc_path ? s.doc_path.split(/[\\/]/).pop() : "Document"),
                  source: s.doc_path,
                  page: s.page,
                  paragraph_index: s.paragraph_index,
                  paragraph: s.paragraph || ""
                };
                try { localStorage.setItem(`docview:${id}`, JSON.stringify(payload)); } catch (_){}

                const label = payload.doc_name;
                const score = (typeof s.score === "number") ? ` (${Math.round(s.score)}%)` : "";
                const extra = (s.page !== undefined && s.page !== null) ? ` • page ${s.page}` : "";
                return `<a class="pill" href="/doc?id=${id}" target="_blank" rel="noopener">${safe(label)}${score}${extra}</a>`;
              } else {
                // THREAD (existing)
                return `
                  <a class="pill" href="${threadUrl(s.thread_id)}" target="_blank" rel="noopener">
                    ${safe(s.subject || s.topic || s.thread_id)}
                    ${typeof s.score === "number" ? ` (${Math.round(s.score)}%)` : ""}
                    ${typeof s.emails_count === "number" ? ` • ${s.emails_count} email${s.emails_count === 1 ? "" : "s"}` : ""}
                  </a>`;
              }
            }).join("")}
          </div>
        </div>` : "";

      const html = `<p>${safe(answer).replace(/\n{2,}/g,"</p><p>").replace(/\n/g,"<br/>")}</p>${sourcesHTML}`;
      thinkingEl.querySelector(".content").innerHTML = html;

    } catch (err) {
      updateAssistantMessage(thinkingEl, { html: `<p>Sorry, I hit an error: ${safe(err.message || String(err))}</p>` });
    }
  });

  // Optional welcome
  addAssistantMessage("<p>Hi! Ask a question and I’ll search your email threads for answers.</p>");
});
