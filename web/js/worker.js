import { PYODIDE_URL } from "./protocol.js";

const HUMAN = "human";
const BOTS_DIR = "/bots";

let pyodide;
let app;
let manifest;
let queue = Promise.resolve();
const installedBots = new Set();

const siteUrl = (path) => new URL(`../${path}`, import.meta.url).href;

async function fetchFile(path) {
  const response = await fetch(siteUrl(path));
  if (!response.ok) throw new Error(`Could not fetch ${path}: ${response.status}`);
  return new Uint8Array(await response.arrayBuffer());
}

function forward(type, payload = {}) {
  const reply = JSON.parse(app.handle(JSON.stringify({ type, payload })));
  if (!reply.ok) throw new Error(reply.error);
  return reply.result;
}

async function boot() {
  postMessage({ progress: "Loading Python runtime" });
  const { loadPyodide } = await import(`${PYODIDE_URL}pyodide.mjs`);
  pyodide = await loadPyodide({ indexURL: PYODIDE_URL });
  manifest = JSON.parse(new TextDecoder().decode(await fetchFile("bots.json")));

  postMessage({ progress: "Installing the engine" });
  await pyodide.loadPackage("micropip");
  const micropip = pyodide.pyimport("micropip");
  await micropip.install.callKwargs(siteUrl(manifest.wheel), { deps: false });

  pyodide.FS.writeFile("session.py", await fetchFile("py/session.py"));
  pyodide.FS.mkdirTree(BOTS_DIR);
  app = pyodide.pyimport("session").App();
  return { bots: manifest.bots.map((bot) => bot.name), rules: forward("schema") };
}

async function installBot(name) {
  const bot = manifest.bots.find((entry) => entry.name === name);
  if (!bot) throw new Error(`Unknown bot: ${name}`);
  if (bot.packages.length) await pyodide.loadPackage(bot.packages);
  pyodide.FS.mkdirTree(`${BOTS_DIR}/${name}`);
  for (const file of bot.files) {
    const bytes = await fetchFile(`bots/${name}/${file}`);
    pyodide.FS.writeFile(`${BOTS_DIR}/${name}/${file}`, bytes);
  }
  installedBots.add(name);
}

async function newGame(payload) {
  const missing = payload.seats.filter((seat) => seat !== HUMAN && !installedBots.has(seat));
  for (const name of new Set(missing)) await installBot(name);
  return forward("new_game", payload);
}

const commands = { boot, new_game: newGame };

async function handle({ id, type, payload }) {
  try {
    const command = commands[type] ?? ((args) => forward(type, args));
    postMessage({ id, ok: true, result: await command(payload) });
  } catch (error) {
    postMessage({ id, ok: false, error: error.message });
  }
}

self.onmessage = ({ data }) => {
  queue = queue.then(() => handle(data));
};
