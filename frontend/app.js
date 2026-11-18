const STAGES = [
  ["verde", "Verde"],
  ["amadurecendo", "Amadurecendo"],
  ["maduro", "Maduro"],
  ["passado", "Passado"],
  ["indefinido", "Indefinido"],
];
const SIZES = [
  ["pequeno", "Pequeno"],
  ["médio", "Médio"],
  ["grande", "Grande"],
];

const form = document.getElementById("form");
const fruitSelect = document.getElementById("fruit");
const filesInput = document.getElementById("files");
const dropzone = document.getElementById("dropzone");
const selected = document.getElementById("selected");
const tiledInput = document.getElementById("tiled");
const submit = document.getElementById("submit");
const results = document.getElementById("results");
const template = document.getElementById("result-template");

let files = [];

async function loadFruits() {
  const res = await fetch("/api/fruits");
  const fruits = await res.json();
  for (const f of fruits) {
    const opt = document.createElement("option");
    opt.value = f.key;
    opt.textContent = f.available ? f.label : `${f.label} (em breve)`;
    opt.disabled = !f.available;
    fruitSelect.appendChild(opt);
  }
}

function setFiles(list) {
  files = Array.from(list).filter((f) => f.type.startsWith("image/"));
  selected.textContent = files.length
    ? `${files.length} imagem(ns): ${files.map((f) => f.name).join(", ")}`
    : "";
  submit.disabled = files.length === 0;
}

filesInput.addEventListener("change", () => setFiles(filesInput.files));
["dragenter", "dragover"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.add("over");
  })
);
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.remove("over");
  })
);
dropzone.addEventListener("drop", (e) => setFiles(e.dataTransfer.files));

function chip(label, count, cls) {
  const li = document.createElement("li");
  li.className = `chip ${cls || ""}`;
  li.innerHTML = `<span>${label}</span><strong>${count}</strong>`;
  return li;
}

function renderResult(card, data) {
  card.querySelector(".r-title").textContent = data.filename || "Imagem";
  card.querySelector(".r-image").src = data.annotated_image;
  card.querySelector(".r-total").innerHTML =
    `<strong>${data.total}</strong> ${data.fruit_label.toLowerCase()}(s) encontrada(s)`;

  card.querySelector(".r-rip-title").textContent =
    data.ripeness_source === "modelo" ? "Maturação (modelo treinado)" : "Maturação (regras de cor)";
  const rip = card.querySelector(".r-ripeness");
  for (const [key, label] of STAGES) {
    if (key === "indefinido" && !data.by_ripeness[key]) continue;
    rip.appendChild(chip(label, data.by_ripeness[key] || 0, `st-${key}`));
  }
  const byModel = data.ripeness_source === "modelo";
  if (byModel) {
    card.querySelector(".r-rip-color-title").hidden = false;
    card.querySelector("th.col-color").hidden = false;
    const ripColor = card.querySelector(".r-ripeness-color");
    ripColor.hidden = false;
    for (const [key, label] of STAGES) {
      if (key === "indefinido" && !data.by_ripeness_color[key]) continue;
      ripColor.appendChild(chip(label, data.by_ripeness_color[key] || 0, `st-${key}`));
    }
  }
  const size = card.querySelector(".r-size");
  for (const [key, label] of SIZES) size.appendChild(chip(label, data.by_size[key] || 0));

  const tbody = card.querySelector("tbody");
  for (const d of data.detections) {
    const tr = document.createElement("tr");
    const colorCell = byModel
      ? `<td><span class="dot st-${d.ripeness_color}"></span>${d.ripeness_color}</td>`
      : "";
    tr.innerHTML = `<td>${d.id}</td><td><span class="dot st-${d.ripeness}"></span>${d.ripeness}</td>` +
      colorCell + `<td>${d.size}</td><td>${Math.round(d.confidence * 100)}%</td>`;
    tbody.appendChild(tr);
  }
  if (!data.detections.length) card.querySelector("details").remove();

  card.querySelector(".r-meta").textContent =
    `${data.image.width}×${data.image.height} px · ${(data.processing_ms / 1000).toFixed(1)} s` +
    (data.tiled ? " · modo detalhado" : "");
}

function renderError(card, name, message) {
  card.querySelector(".r-title").textContent = name;
  const p = document.createElement("p");
  p.className = "error";
  p.textContent = message;
  card.querySelector(".r-body").replaceChildren(p);
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!files.length) return;
  submit.disabled = true;
  results.innerHTML = "";

  // Envia uma imagem por vez: o servidor processa em fila.
  for (const [i, file] of files.entries()) {
    submit.textContent = `Analisando ${i + 1} de ${files.length}...`;
    const card = template.content.firstElementChild.cloneNode(true);
    card.classList.add("loading");
    card.querySelector(".r-title").textContent = `${file.name} (processando...)`;
    results.appendChild(card);

    const body = new FormData();
    body.append("file", file);
    body.append("fruit", fruitSelect.value);
    body.append("tiled", tiledInput.checked ? "true" : "false");
    try {
      const res = await fetch("/api/analyze", { method: "POST", body });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || `Erro ${res.status}`);
      renderResult(card, data);
    } catch (err) {
      renderError(card, file.name, err.message);
    } finally {
      card.classList.remove("loading");
    }
  }
  submit.textContent = "Analisar";
  submit.disabled = false;
});

loadFruits();
