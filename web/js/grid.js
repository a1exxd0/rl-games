const EVEN = 0.5;
const STRONG_FROM = 0.7;

const percent = (value) => `${(value * 100).toFixed(1)}%`;

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function indexRates(rows) {
  const rates = new Map();
  for (const row of rows) {
    if (!rates.has(row.first)) rates.set(row.first, new Map());
    rates.get(row.first).set(row.second, row);
  }
  return rates;
}

function meanRate(opponents) {
  const values = [...opponents.values()].map((row) => row.win_rate);
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

const rankedBots = (rates) =>
  [...rates.keys()].sort((a, b) => meanRate(rates.get(b)) - meanRate(rates.get(a)));

function fillCell(cell, row) {
  const strength = Math.abs(row.win_rate - EVEN) * 2;
  const pole = row.win_rate >= EVEN ? "--grid-win" : "--grid-loss";
  cell.style.background = `color-mix(in oklab, var(${pole}) ${strength * 100}%, var(--grid-mid))`;
  cell.classList.toggle("strong", strength >= STRONG_FROM);
  cell.textContent = percent(row.win_rate);
  cell.title = `${row.first} beat ${row.second} in ${percent(row.win_rate)} of ${row.games.toLocaleString()} games (95% CI ${percent(row.ci_low)}–${percent(row.ci_high)})`;
}

function bodyCell(rates, bot, opponent) {
  if (bot === opponent) return element("td", "grid-self");
  const row = rates.get(bot).get(opponent);
  const cell = element("td", row ? "grid-cell" : "grid-missing");
  if (row) fillCell(cell, row);
  return cell;
}

function headerRow(bots) {
  const row = element("tr");
  row.append(element("th", "grid-corner"));
  for (const bot of bots) {
    const heading = element("th", "grid-column");
    heading.scope = "col";
    heading.append(element("span", "", bot));
    row.append(heading);
  }
  return row;
}

function bodyRow(rates, bots, bot) {
  const row = element("tr");
  const heading = element("th", "grid-row", bot);
  heading.scope = "row";
  row.append(heading, ...bots.map((opponent) => bodyCell(rates, bot, opponent)));
  return row;
}

function legend() {
  const block = element("div", "grid-legend");
  block.append(element("span", "", "0%"), element("span", "grid-scale"), element("span", "", "100%"));
  return block;
}

export function renderWinGrid(rows) {
  if (!rows?.length) return [];
  const rates = indexRates(rows);
  const bots = rankedBots(rates);
  const table = element("table", "win-grid");
  const head = element("thead");
  const body = element("tbody");
  head.append(headerRow(bots));
  body.append(...bots.map((bot) => bodyRow(rates, bots, bot)));
  table.append(head, body);

  const section = element("section", "win-rates");
  const header = element("header");
  const title = element("div");
  title.append(
    element("h2", "", "Win rates"),
    element("p", "", "Each cell is the row bot's win rate against the column bot, 2 players. Strongest bots first."),
  );
  header.append(title, legend());
  const scroller = element("div", "grid-scroller");
  scroller.tabIndex = 0;
  scroller.append(table);
  section.append(header, scroller);
  return [section];
}
