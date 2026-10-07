
const BOT_ICON = `<svg viewBox="0 0 32 32" fill="none" class="bot-icon" xmlns="http://www.w3.org/2000/svg">
  <path d="M9 6V15C9 18.3 11.7 21 15 21C18.3 21 21 18.3 21 15V10" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/>
  <path d="M9 6C9 6 9 4 7 4C5 4 5 6 5 6" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/>
  <circle cx="24" cy="10" r="2.6" stroke="currentColor" stroke-width="2"/>
  <circle cx="15" cy="24" r="3" fill="currentColor"/>
</svg>`;

const chatWindow = document.getElementById("chatWindow");
const chatForm = document.getElementById("chatForm");
const userInput = document.getElementById("userInput");
const chipRow = document.getElementById("chipRow");
// message bubbles

function scrollToBottom() {
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function addUserMessage(text) {
  const msg = document.createElement("div");
  msg.className = "message user";
  msg.innerHTML = `<div class="bubble"></div>`;
  msg.querySelector(".bubble").textContent = text;
  chatWindow.appendChild(msg);
  scrollToBottom();
}

function addBotBubble(innerHTML, extraClass = "") {
  const msg = document.createElement("div");
  msg.className = `message bot ${extraClass}`.trim();
  msg.innerHTML = `<div class="avatar">${BOT_ICON}</div><div class="bubble">${innerHTML}</div>`;
  chatWindow.appendChild(msg);
  scrollToBottom();
  return msg;
}

function showTyping() {
  const msg = document.createElement("div");
  msg.className = "message bot typing";
  msg.innerHTML = `
    <div class="avatar">${BOT_ICON}</div>
    <div class="bubble">
      <span class="dot"></span><span class="dot"></span><span class="dot"></span>
    </div>`;
  chatWindow.appendChild(msg);
  scrollToBottom();
  return msg;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}


function renderResponse(data) {
  if (data.type === "emergency") {
    addBotBubble(`⚠️ <strong>${escapeHtml(data.message)}</strong>`, "emergency");
    return;
  }

  if (data.type === "clarify") {
    addBotBubble(escapeHtml(data.message));
    return;
  }

  if (data.type === "smalltalk") {
    addBotBubble(escapeHtml(data.message));
    return;
  }

  if (data.type === "result") {
    let html = "";

    if (data.detected_intent || (data.extracted_keywords && data.extracted_keywords.length)) {
      html += `<div class="pipeline-trace">`;
      if (data.detected_intent) {
        html += `<span class="trace-tag">Intent: ${escapeHtml(data.detected_intent)}</span>`;
      }
      if (data.extracted_keywords && data.extracted_keywords.length) {
        html += `<span class="trace-tag">Keywords: ${escapeHtml(data.extracted_keywords.join(", "))}</span>`;
      }
      html += `</div>`;
    }

    if (data.found_symptoms && data.found_symptoms.length) {
      html += `<p><span class="label">Symptoms noted:</span> ${escapeHtml(data.found_symptoms.join(", "))}</p>`;
    }

    data.matches.forEach((m) => {
      html += `
        <div class="result-card">
          <span class="score-tag">TF-IDF match: ${m.score}%</span>
          <h3>${escapeHtml(m.name)}</h3>
          <p>${escapeHtml(m.info)}</p>
          <p><span class="label">General medicine info:</span> ${escapeHtml(m.medicine_info)}</p>
          <p><span class="label">Suggested doctor:</span> ${escapeHtml(m.doctor)}</p>
          <p><span class="label">Precautions:</span></p>
          <ul>${m.precautions.map((p) => `<li>${escapeHtml(p)}</li>`).join("")}</ul>
          <p><span class="label">⚠ Seek care if you notice:</span></p>
          <ul>${m.warning_signs.map((w) => `<li>${escapeHtml(w)}</li>`).join("")}</ul>
        </div>`;
    });

    if (data.needs_more_info) {
      html += `<p style="margin-top:8px;">I don't have enough symptoms yet to narrow this down further — feel free to add more details.</p>`;
    }

    html += `<div class="disclaimer-box">${escapeHtml(data.disclaimer)}</div>`;

    addBotBubble(html);
    return;
  }

  // Fallback for anything unexpected
  addBotBubble("Sorry, something went wrong reading that response.");
}

// send message to backend 

async function sendMessage(text) {
  addUserMessage(text);
  const typingEl = showTyping();

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });
    const data = await res.json();
    typingEl.remove();
    renderResponse(data);
  } catch (err) {
    typingEl.remove();
    addBotBubble("I couldn't reach the server. Please make sure the Flask backend is running.");
    console.error(err);
  }
}

chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = userInput.value.trim();
  if (!text) return;
  userInput.value = "";
  sendMessage(text);
});

chipRow.addEventListener("click", (e) => {
  const chip = e.target.closest(".chip");
  if (!chip) return;
  sendMessage(chip.dataset.text);
});
