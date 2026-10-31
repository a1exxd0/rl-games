import { installCardDefs } from "./cards.js";
import { GameController } from "./game.js";
import { renderModels } from "./models.js";
import { timing } from "./motion.js";
import { EngineClient } from "./protocol.js";
import { renderLog, statusText } from "./render.js";
import { Scene } from "./scene.js";

const HUMAN = "human";
const PREFERRED_BOTS = ["greedy", "random"];
const FOCUS_KEEPS_SPACE = "input, select, textarea, summary, #setup button, .tabs button";
const $ = (id) => document.getElementById(id);

const selected = new Set();
let botNames = [];
let ruleDefaults = {};
let modelReport = null;
let booted = false;
let resumeOnReturn = false;

installCardDefs();
const scene = new Scene($("table"), toggleCard);
const engine = new EngineClient(setStatus);
const game = new GameController(engine, render);

function setStatus(text, isError = false) {
  $("status").textContent = text;
  $("status").classList.toggle("error", isError);
}

const cardKey = (cards) => [...cards].sort().join(" ");

function selectionIsLegal() {
  const chosen = cardKey(selected);
  return game.snapshot.legal.some((play) => play.cards.length && cardKey(play.cards) === chosen);
}

const canPass = () => game.snapshot.legal.some((play) => !play.cards.length);

function toggleCard(card) {
  if (!selected.delete(card)) selected.add(card);
  render();
}

function submit(cards) {
  selected.clear();
  game.submit(cards);
}

function setVisible(id, visible) {
  $(id).hidden = !visible;
}

function renderControls() {
  const { mode, snapshot, playing } = game;
  const isHuman = mode === "human";
  const isReplay = mode === "replay";
  setVisible("auto-controls", !isHuman);
  setVisible("human-controls", isHuman);
  setVisible("replay-controls", isReplay);
  setVisible("save-replay", mode !== "replay" && snapshot.over);

  $("toggle").textContent = playing ? "Pause" : "Auto-play";
  $("toggle").disabled = !playing && !game.canStep;
  $("step").disabled = !game.canStep;
  $("play-selected").disabled = !selectionIsLegal();
  $("pass").disabled = !canPass();
  $("clear").disabled = !selected.size;

  if (isReplay) {
    $("back").disabled = !snapshot.cursor;
    $("cursor").max = snapshot.total;
    $("cursor").value = snapshot.cursor;
    $("cursor-label").textContent = `Move ${snapshot.cursor} of ${snapshot.total}`;
  }
}

function render(kind) {
  const { snapshot, error } = game;
  if (!snapshot) {
    if (error) setStatus(error, true);
    return;
  }
  const { flight } = timing(game.delay);
  if (kind) renderLog($("log"), snapshot, { kind, animated: flight > 0 });
  renderControls();
  setStatus(error ?? statusText(snapshot), Boolean(error));
  $("details").textContent = `Rules ${snapshot.rules_id}${snapshot.seed === null ? "" : ` · seed ${snapshot.seed}`}`;
  const shown = kind ? scene.show(snapshot, { kind, flight }) : undefined;
  scene.select({ selected, playable: new Set(snapshot.legal.flatMap((play) => play.cards)) });
  return shown;
}

function currentMode() {
  return document.querySelector('input[name="mode"]:checked').value;
}

const baseName = (bot) => bot.split("@")[0];

function defaultBot(seat) {
  const preferred = PREFERRED_BOTS.map((name) => botNames.find((bot) => baseName(bot) === name)).filter(Boolean);
  const pool = preferred.length ? preferred : botNames;
  return pool[seat % pool.length];
}

function renderSeatPickers() {
  const previous = [...$("seats").querySelectorAll("select")].map((select) => select.value);
  const count = Number($("players").value);
  const isHuman = currentMode() === HUMAN;
  const pickers = Array.from({ length: count }, (_, seat) => {
    const label = document.createElement("label");
    label.textContent = `Seat ${seat}`;
    if (isHuman && seat === 0) {
      label.append(Object.assign(document.createElement("span"), { textContent: "You" }));
      return label;
    }
    const select = document.createElement("select");
    select.append(...botNames.map((name) => new Option(name, name)));
    select.value = previous[seat] ?? defaultBot(seat);
    label.append(select);
    return label;
  });
  $("seats").replaceChildren(...pickers);
}

function renderRuleToggles() {
  const toggles = Object.entries(ruleDefaults).map(([name, checked]) => {
    const label = document.createElement("label");
    label.className = "check";
    const input = Object.assign(document.createElement("input"), { type: "checkbox", name, checked });
    const text = name.replaceAll("_", " ");
    label.append(input, text[0].toUpperCase() + text.slice(1));
    return label;
  });
  $("rules").replaceChildren(...toggles);
}

function readConfig() {
  const mode = currentMode();
  const rules = { n_players: Number($("players").value) };
  for (const input of $("rules").querySelectorAll("input")) rules[input.name] = input.checked;
  const bots = [...$("seats").querySelectorAll("select")].map((select) => select.value);
  const seed = $("seed").value.trim();
  return {
    mode: mode === HUMAN ? "human" : "watch",
    rules,
    seats: mode === HUMAN ? [HUMAN, ...bots] : bots,
    seed: seed === "" ? null : Number(seed),
  };
}

function startGame() {
  selected.clear();
  return game.newGame(readConfig());
}

async function saveReplay() {
  const replay = await game.export();
  const url = URL.createObjectURL(new Blob([JSON.stringify(replay)], { type: "application/json" }));
  const link = Object.assign(document.createElement("a"), { href: url, download: `tienlen-${replay.seed}.json` });
  link.click();
  URL.revokeObjectURL(url);
}

async function openReplay(file) {
  selected.clear();
  try {
    await game.loadReplay(JSON.parse(await file.text()));
  } catch (error) {
    game.fail(`Could not read replay: ${error.message}`);
  }
}

function showTab(tab) {
  const leavingGame = tab === "models" && !$("game-view").hidden;
  $("game-view").hidden = tab !== "game";
  $("models-view").hidden = tab !== "models";
  for (const button of document.querySelectorAll("[data-tab]")) {
    button.setAttribute("aria-selected", String(button.dataset.tab === tab));
  }
  history.replaceState(null, "", tab === "models" ? "#models" : "#game");
  if (leavingGame) {
    resumeOnReturn = game.playing;
    game.pause();
  } else if (tab === "game" && resumeOnReturn) {
    resumeOnReturn = false;
    game.toggle();
  }
}

function seatBot(ref, mode) {
  resumeOnReturn = false;
  showTab("game");
  document.querySelector(`input[name="mode"][value="${mode}"]`).checked = true;
  renderSeatPickers();
  const selects = [...$("seats").querySelectorAll("select")];
  selects.forEach((select, index) => {
    select.value = index === 0 ? ref : defaultBot(index);
  });
  return startGame();
}

function drawModels() {
  renderModels($("models-view"), modelReport, {
    ready: booted,
    onWatch: (ref) => seatBot(ref, "watch"),
    onPlay: (ref) => seatBot(ref, HUMAN),
  });
}

async function loadModels() {
  try {
    const response = await fetch("models.json");
    modelReport = response.ok ? await response.json() : null;
  } catch {
    modelReport = null;
  }
  drawModels();
}

function spaceIsForAutoPlay({ code, repeat, target }) {
  if (code !== "Space" || repeat || $("game-view").hidden || game.mode === "human") return false;
  return !target.closest(FOCUS_KEEPS_SPACE);
}

function handleKey(event) {
  if (!spaceIsForAutoPlay(event)) return;
  event.preventDefault();
  $("toggle").click();
}

function bindEvents() {
  document.addEventListener("keydown", handleKey);
  for (const button of document.querySelectorAll("[data-tab]")) {
    button.onclick = () => showTab(button.dataset.tab);
  }
  $("setup").addEventListener("submit", (event) => {
    event.preventDefault();
    startGame();
  });
  $("players").onchange = renderSeatPickers;
  for (const radio of document.querySelectorAll('input[name="mode"]')) radio.onchange = renderSeatPickers;
  $("replay-file").onchange = ({ target }) => {
    if (target.files[0]) openReplay(target.files[0]);
    target.value = "";
  };
  $("toggle").onclick = () => game.toggle();
  $("step").onclick = () => game.step();
  $("back").onclick = () => game.seek(game.snapshot.cursor - 1);
  $("cursor").oninput = ({ target }) => game.seek(Number(target.value));
  $("speed").onchange = ({ target }) => {
    game.delay = Number(target.value);
  };
  $("play-selected").onclick = () => submit([...selected]);
  $("pass").onclick = () => submit([]);
  $("clear").onclick = () => {
    selected.clear();
    render();
  };
  $("save-replay").onclick = saveReplay;
}

async function start() {
  bindEvents();
  loadModels();
  try {
    const { bots, rules } = await engine.call("boot");
    botNames = bots;
    ruleDefaults = rules;
    renderSeatPickers();
    renderRuleToggles();
    $("new-game").disabled = false;
    booted = true;
    drawModels();
    if (location.hash === "#models") showTab("models");
    await startGame();
  } catch (error) {
    setStatus(`Could not start: ${error.message}`, true);
  }
}

start();
