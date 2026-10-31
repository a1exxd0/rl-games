export const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/";

export class EngineClient {
  #worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  #pending = new Map();
  #nextId = 1;

  constructor(onProgress) {
    this.#worker.onmessage = ({ data }) => {
      if ("progress" in data) {
        onProgress(data.progress);
        return;
      }
      const { resolve, reject } = this.#pending.get(data.id);
      this.#pending.delete(data.id);
      if (data.ok) resolve(data.result);
      else reject(new Error(data.error));
    };
    this.#worker.onerror = (event) => {
      for (const { reject } of this.#pending.values()) reject(new Error(event.message));
      this.#pending.clear();
    };
  }

  call(type, payload = {}) {
    const id = this.#nextId++;
    return new Promise((resolve, reject) => {
      this.#pending.set(id, { resolve, reject });
      this.#worker.postMessage({ id, type, payload });
    });
  }
}
