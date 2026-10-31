import { renderWinGrid } from "./grid.js";

const VERDICTS = {
  promoted: "Promoted",
  rejected: "Rejected",
  pending: "In progress",
  failed: "Failed",
  reference: "Reference",
};

const METERS = [
  ["vs_random", "vs random"],
  ["vs_greedy", "vs greedy"],
  ["h2h", "vs parent"],
];

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}

const percent = (value) => `${(value * 100).toFixed(1)}%`;

function meter(label, value) {
  const row = element("div", "meter");
  row.append(element("span", "meter-label", label));
  const track = element("span", "meter-track");
  const fill = element("span", "meter-fill");
  fill.style.transform = `scaleX(${value ?? 0})`;
  track.append(fill, element("span", "meter-even"));
  row.append(track, element("span", "meter-value", value === null ? "–" : percent(value)));
  return row;
}

function actions(ref, playable, ready, handlers) {
  const row = element("div", "model-actions");
  if (!playable) {
    row.append(element("span", "model-unplayable", ref ? "Can't run in the browser" : "No bot yet"));
    return row;
  }
  for (const [label, handler, className] of [["Watch", handlers.onWatch, "primary"], ["Play against", handlers.onPlay, ""]]) {
    const button = element("button", className, label);
    button.type = "button";
    button.disabled = !ready;
    button.onclick = () => handler(ref);
    row.append(button);
  }
  return row;
}

function labelled(label, text, className) {
  const block = element("div", className);
  block.append(element("h4", "", label), element("p", "", text));
  return block;
}

function metrics(model) {
  if (model.vs_random === null && model.vs_greedy === null) {
    return element("p", "model-waiting", "No results yet");
  }
  const block = element("div", "meters");
  const rows = model.parent_ref ? METERS : METERS.slice(0, 2);
  block.append(...rows.map(([key, label]) => meter(label, model[key])));
  return block;
}

function modelCard(model, isBest, ready, handlers) {
  const card = element("article", `model ${model.verdict}`);
  card.dataset.step = model.step;
  const node = element("div", "model-node", model.step);
  const body = element("div", "model-body");

  const header = element("header");
  const title = element("div", "model-title");
  title.append(element("h3", "", model.bot_ref ?? `Step ${model.step}`));
  title.append(element("span", `verdict ${model.verdict}`, VERDICTS[model.verdict] ?? model.verdict));
  if (isBest) title.append(element("span", "verdict best", "Current best"));
  header.append(title, element("p", "model-parent", model.parent_ref ? `Built on ${model.parent_ref}` : "Baseline"));

  body.append(header, labelled("Change", model.change, "model-change"));
  if (model.hypothesis) body.append(labelled("Hypothesis", model.hypothesis, "model-hypothesis"));
  body.append(metrics(model));
  if (model.notes) body.append(labelled("Notes", model.notes, "model-notes"));
  body.append(actions(model.bot_ref, model.playable, ready, handlers));
  card.append(node, body);
  return card;
}

function compactRow({ ref, detail, playable }, ready, handlers) {
  const row = element("li", "other-bot");
  const name = element("span", "other-name", ref);
  row.append(name, element("span", "other-detail", detail), actions(ref, playable, ready, handlers));
  return row;
}

function compactSection(rows, ready, handlers) {
  if (!rows.length) return [];
  const section = element("section", "other-bots");
  const list = element("ul");
  list.append(...rows.map((row) => compactRow(row, ready, handlers)));
  section.append(element("h2", "", "Other bots"), list);
  return [section];
}

function emptyState() {
  const empty = element("section", "models-empty");
  empty.append(
    element("h2", "", "No models yet"),
    element("p", "", "Each lab step adds a model here, with what changed and whether it helped."),
  );
  const steps = element("ol");
  for (const command of ['tienlen lab step lab_01_linear --change "…" --hypothesis "…"', "tienlen web build"]) {
    const item = element("li");
    item.append(element("code", "", command));
    steps.append(item);
  }
  empty.append(steps, element("p", "models-hint", "Then reload this page."));
  return empty;
}

function summary(steps, best) {
  const count = (verdict) => steps.filter((model) => model.verdict === verdict).length;
  const parts = [
    `${steps.length} ${steps.length === 1 ? "step" : "steps"}`,
    `${count("promoted")} promoted`,
    `${count("rejected")} rejected`,
  ];
  if (count("pending")) parts.push(`${count("pending")} in progress`);
  const line = element("div", "models-summary");
  line.append(element("h2", "", "Model timeline"), element("p", "", parts.join(" · ")));
  if (best) line.append(element("p", "models-best", `Current best: ${best.bot_ref}`));
  return line;
}

const referenceRow = (model) => ({
  ref: model.bot_ref,
  detail: [model.vs_random, model.vs_greedy].every((rate) => rate !== null)
    ? `${percent(model.vs_random)} vs random · ${percent(model.vs_greedy)} vs greedy`
    : "Reference",
  playable: model.playable,
});

export function renderModels(root, report, { ready, ...handlers }) {
  const models = report?.models ?? [];
  const steps = models.filter((model) => model.verdict !== "reference");
  const references = models.filter((model) => model.verdict === "reference").map(referenceRow);
  const others = (report?.others ?? []).map(({ ref, kind, playable }) => ({ ref, detail: kind, playable }));
  const rest = compactSection([...references, ...others], ready, handlers);
  const grid = renderWinGrid(report?.win_rates);
  if (!steps.length) {
    root.replaceChildren(emptyState(), ...grid, ...rest);
    return;
  }
  const best = steps.filter((model) => model.verdict === "promoted").at(-1);
  const timeline = element("div", "timeline");
  timeline.append(...steps.map((model) => modelCard(model, model === best, ready, handlers)));
  root.replaceChildren(summary(steps, best), ...grid, timeline, ...rest);
}
