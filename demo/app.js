"use strict";

const byId = (id) => document.getElementById(id);
const form = byId("poem-form");
const keywordInput = byId("keywords");
const seedInput = byId("seed");
const ranking = byId("ranking");
const status = byId("status");
const produce = byId("produce");
const sampleList = byId("examples");
const samples = window.ORCHID_EXAMPLES;
let runtime = null;
let busy = false;

function setStatus(message, error = false) {
  status.textContent = message;
  status.classList.toggle("error", error);
}

function renderPoem(mode, item, words) {
  const poem = byId(`${mode}-poem`);
  const details = byId(`${mode}-details`);
  poem.replaceChildren();
  details.replaceChildren();
  const lines = item.generated.replace(/([，。！？；,.!?;])(?=[^\n])/gu, "$1\n").split("\n");
  lines.forEach((line, index) => {
    const end = item.distinct_rhyme && (index === 1 || index === 3)
      ? line.match(/^(.*)(\p{Script=Han})([，。！？；,.!?;]?)$/u) : null;
    if (index > 0) poem.append("\n");
    if (!end) {
      poem.append(line);
    } else {
      const character = document.createElement("span");
      character.className = "rhyme";
      character.textContent = end[2];
      poem.append(end[1], character, end[3]);
    }
  });
  const rhyme = item.distinct_rhyme
    ? `${item.rhyme_characters.join(" ／ ")} · ${item.ping_matches.join("、")}`
    : item.rhyme === "unknown" ? "韵部待判" : "未确认异字同韵";
  const dictionary = [
    ["句式", item.format_ok ? "四句五字" : "按原文显示"],
    ["偶句韵脚", rhyme],
    ["原词覆盖", `${item.keyword_hits.length} / ${words.length}${item.keyword_hits.length ? " · " + item.keyword_hits.join("、") : ""}`],
    ["生成耗时", `${item.seconds.toFixed(3)} 秒`],
  ];
  for (const [label, value] of dictionary) {
    const term = document.createElement("dt");
    const detail = document.createElement("dd");
    term.textContent = label;
    detail.textContent = value;
    details.append(term, detail);
  }
}

function show(pair, source) {
  renderPoem("raw", pair.raw, pair.keywords);
  renderPoem("rhymed", pair.rhymed, pair.keywords);
  byId("source").textContent = `${source} · 种子 ${pair.seed}`;
}

function showSample(sample) {
  keywordInput.value = sample.keywords.join(" ");
  seedInput.value = sample.seed;
  show(sample, "冻结测试样例 · CUDA");
  for (const [index, button] of [...sampleList.children].entries()) {
    button.setAttribute("aria-pressed", String(samples[index] === sample));
  }
}

function freezeForm(frozen) {
  busy = frozen;
  produce.disabled = frozen;
  keywordInput.disabled = frozen;
  seedInput.disabled = frozen;
  ranking.disabled = frozen || !runtime;
  for (const button of sampleList.children) button.disabled = frozen;
  byId("poems").setAttribute("aria-busy", String(frozen));
}

samples.forEach((sample) => {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = sample.keywords.join(" · ");
  button.addEventListener("click", () => {
    showSample(sample);
    setStatus(runtime ? "已载入题目，可生成新诗。" : "已载入封存样例。");
  });
  sampleList.append(button);
});
showSample(samples[0]);

async function connect() {
  if (location.protocol !== "file:") {
    try {
      const response = await fetch("/api/status");
      if (response.ok) runtime = await response.json();
    } catch {
      runtime = null;
    }
  }
  if (runtime && runtime.device) {
    produce.textContent = "生成两首";
    ranking.disabled = false;
    setStatus(`模型就绪 · ${runtime.device.toUpperCase()}`);
  } else {
    runtime = null;
    setStatus("离线样例 · 可点选诗题");
  }
}
connect();

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (busy) return;
  const keywords = keywordInput.value.trim().normalize("NFC").split(/[\s、，,]+/u).filter(Boolean);
  const seed = Number(seedInput.value);
  if (keywords.length < 1 || keywords.length > 3 || new Set(keywords).size !== keywords.length
      || keywords.some((word) => !/^\p{Script=Han}{1,2}$/u.test(word))) {
    setStatus("请输入一至三个不同的短词，每词一至两个汉字。", true);
    return;
  }
  if (!runtime) {
    const sample = samples.find((item) => item.seed === seed && item.keywords.join() === keywords.join());
    if (sample) {
      showSample(sample);
      setStatus("已载入封存样例。");
    } else {
      setStatus("离线模式可查看预设题。自选题需要启动本机服务。", true);
    }
    return;
  }
  freezeForm(true);
  setStatus("正在生成两首诗…");
  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({keywords, seed, ranking: ranking.value}),
    });
    const pair = await response.json();
    if (!response.ok) {
      setStatus(pair.error || "本机服务暂不可用。", true);
      return;
    }
    show(pair, `本机实跑 · ${runtime.device.toUpperCase()}`);
    for (const button of sampleList.children) button.setAttribute("aria-pressed", "false");
    setStatus("生成完成。");
  } catch {
    setStatus("连接本机服务失败，可先查看封存样例。", true);
  } finally {
    freezeForm(false);
  }
});
