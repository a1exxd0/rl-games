import { timing } from "./motion.js";

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const KINDS = { new_game: "deal", load_replay: "jump", step: "move", play: "move" };

export class GameController {
  #engine;
  #onChange;
  #game = 0;
  #loop = 0;

  mode = null;
  snapshot = null;
  playing = false;
  delay = 700;
  error = null;

  constructor(engine, onChange) {
    this.#engine = engine;
    this.#onChange = onChange;
  }

  get #hasNext() {
    if (!this.snapshot) return false;
    if (this.mode === "replay") return this.snapshot.cursor < this.snapshot.total;
    if (this.mode === "human") return !this.snapshot.over && !this.snapshot.legal.length;
    return !this.snapshot.over;
  }

  get canStep() {
    return this.mode !== "human" && this.#hasNext;
  }

  #kindOf(type, payload) {
    if (type !== "seek") return KINDS[type];
    return payload.cursor === this.snapshot.cursor + 1 ? "move" : "jump";
  }

  async #request(game, type, payload, mode = this.mode) {
    try {
      const snapshot = await this.#engine.call(type, payload);
      if (game !== this.#game) return false;
      const kind = this.#kindOf(type, payload);
      this.mode = mode;
      this.snapshot = snapshot;
      this.error = null;
      await this.#onChange(kind);
      return game === this.#game;
    } catch (error) {
      if (game === this.#game) this.fail(error.message);
      return false;
    }
  }

  async #open(mode, type, payload) {
    const game = ++this.#game;
    this.pause();
    const autoPlays = mode !== "replay";
    const loop = this.#loop;
    this.playing = autoPlays;
    const opened = await this.#request(game, type, payload, mode);
    if (opened && autoPlays && loop === this.#loop) this.#run();
  }

  #advance() {
    if (this.mode === "replay") {
      return this.#request(this.#game, "seek", { cursor: this.snapshot.cursor + 1 });
    }
    return this.#request(this.#game, "step");
  }

  async #run() {
    const loop = ++this.#loop;
    this.playing = true;
    this.#onChange();
    while (loop === this.#loop && this.#hasNext) {
      await sleep(timing(this.delay).rest);
      if (loop === this.#loop) await this.#advance();
    }
    if (loop === this.#loop) this.pause();
  }

  newGame({ mode, ...payload }) {
    return this.#open(mode, "new_game", payload);
  }

  loadReplay(replay) {
    return this.#open("replay", "load_replay", { replay });
  }

  pause() {
    this.#loop++;
    this.playing = false;
    this.#onChange();
  }

  fail(message) {
    this.error = message;
    this.pause();
  }

  toggle() {
    if (this.playing) this.pause();
    else this.#run();
  }

  step() {
    this.pause();
    if (this.canStep) return this.#advance();
  }

  seek(cursor) {
    this.pause();
    return this.#request(this.#game, "seek", { cursor });
  }

  async submit(cards) {
    if (await this.#request(this.#game, "play", { cards })) this.#run();
  }

  export() {
    return this.#engine.call("export");
  }
}
