import { CARD_BACK, cardFront, cardName } from "./cards.js";
import { CARD_HEIGHT, CARD_WIDTH, layoutTable } from "./layout.js";
import { motionAllowed, pop, transformOf, travel } from "./motion.js";
import { ORDINALS, seatLabel, seatName } from "./render.js";

const FACE_DOWN = "rotateY(180deg)";
const FACE_UP = "rotateY(0deg)";
const FLYING_Z = 1000;
const HUMAN_FLIGHT = 240;
const DEAL_SPREAD = 2;
const SHUFFLE_OUT = 160;

const poseTransform = (pose) => transformOf(pose, CARD_WIDTH, CARD_HEIGHT);

function createCardElement({ card }) {
  const element = document.createElement("button");
  element.type = "button";
  element.className = "card";
  element.disabled = true;
  element.innerHTML = `<span class="card-lift"><span class="card-flip">${card ? cardFront(card) : ""}${CARD_BACK}</span></span>`;
  if (card) {
    element.dataset.card = card;
    element.setAttribute("aria-label", cardName(card));
  }
  return element;
}

const flipOf = (element) => element.querySelector(".card-flip");

function createPlate(seat) {
  const plate = document.createElement("div");
  plate.className = "plate";
  plate.dataset.seat = seat;
  plate.innerHTML = `<span class="seat-dot"></span>
    <span class="plate-text"><span class="plate-name"></span><span class="plate-sub"><span class="plate-count"></span><span class="plate-unit"></span></span></span>
    <span class="plate-badge" hidden></span>`;
  return plate;
}

function badgeFor(view, seat) {
  const place = view.placements.indexOf(seat);
  if (place >= 0) return ORDINALS[place];
  return view.passed.includes(seat) ? "Passed" : "";
}

function updatePlate(plate, snapshot) {
  const { view } = snapshot;
  const seat = Number(plate.dataset.seat);
  const count = view.hand_counts[seat];
  const badge = badgeFor(view, seat);
  plate.dataset.count = count;
  plate.classList.toggle("active", view.to_act === seat);
  plate.classList.toggle("passed", badge === "Passed");
  plate.classList.toggle("finished", view.placements.includes(seat));
  plate.querySelector(".plate-name").textContent = seatName(snapshot, seat);
  plate.querySelector(".plate-count").textContent = count;
  plate.querySelector(".plate-unit").textContent = count === 1 ? " card" : " cards";
  const label = plate.querySelector(".plate-badge");
  label.hidden = !badge;
  label.textContent = badge;
}

function pileCaption(snapshot) {
  const { current, last_player: lastPlayer } = snapshot.view;
  if (current) return `${seatName(snapshot, lastPlayer)} · ${current.type.replaceAll("_", " ")}`;
  return snapshot.over ? "" : "Open lead";
}

export class Scene {
  #stage;
  #cards = new Map();
  #plates = new Map();
  #note = document.createElement("p");
  #finale = document.createElement("div");
  #running = new Set();
  #leaving = new Set();
  #rest = null;
  #token = 0;
  #snapshot = null;

  constructor(stage, onCard) {
    this.#stage = stage;
    this.#note.className = "pile-note";
    this.#finale.className = "finale";
    this.#finale.hidden = true;
    const mark = document.createElement("div");
    mark.className = "felt-mark";
    mark.textContent = "Tiến Lên";
    stage.append(mark, this.#note, this.#finale);
    stage.addEventListener("click", (event) => {
      const card = event.target.closest(".card:not(:disabled)");
      if (card) onCard(card.dataset.card);
    });
    new ResizeObserver(() => this.#resnap()).observe(stage);
  }

  get #size() {
    return { width: this.#stage.clientWidth, height: this.#stage.clientHeight };
  }

  #resnap() {
    if (!this.#snapshot) return;
    this.#settle();
    this.#snap(layoutTable(this.#snapshot, this.#size));
    this.#showPlates(this.#snapshot);
  }

  #settle() {
    this.#token++;
    for (const animation of this.#running) animation.cancel();
    this.#running.clear();
    for (const element of this.#leaving) element.remove();
    this.#leaving.clear();
    this.#rest?.();
    this.#rest = null;
  }

  #setPose(entry, pose) {
    const { element } = entry;
    entry.pose = pose;
    element.style.transform = poseTransform(pose);
    element.style.zIndex = pose.z;
    element.dataset.zone = pose.zone;
    element.classList.toggle("stale", Boolean(pose.stale));
    flipOf(element).style.transform = pose.face ? FACE_UP : FACE_DOWN;
  }

  #create(pose) {
    const entry = { element: createCardElement(pose), pose: null };
    this.#cards.set(pose.key, entry);
    this.#stage.append(entry.element);
    return entry;
  }

  #placePlates(layout) {
    const live = new Set(layout.plates.map((plate) => plate.seat));
    for (const [seat, element] of this.#plates) {
      if (live.has(seat)) continue;
      element.remove();
      this.#plates.delete(seat);
    }
    for (const { seat, position, x, y } of layout.plates) {
      if (!this.#plates.has(seat)) {
        this.#plates.set(seat, createPlate(seat));
        this.#stage.append(this.#plates.get(seat));
      }
      const element = this.#plates.get(seat);
      element.dataset.position = position;
      element.style.width = `${layout.plateWidth}px`;
      element.style.transform = `translate(${x}px, ${y}px) translate(-50%, -50%)`;
    }
    this.#note.style.transform = `translate(${layout.pile.x}px, ${layout.pile.y}px) translate(-50%, ${layout.pile.card * 0.7 + 8}px)`;
  }

  #snap(layout) {
    const wanted = new Map(layout.cards.map((pose) => [pose.key, pose]));
    for (const [key, entry] of this.#cards) {
      if (wanted.has(key)) continue;
      entry.element.remove();
      this.#cards.delete(key);
    }
    for (const pose of layout.cards) this.#setPose(this.#cards.get(pose.key) ?? this.#create(pose), pose);
    this.#placePlates(layout);
  }

  #showPlates(snapshot) {
    for (const plate of this.#plates.values()) updatePlate(plate, snapshot);
    this.#note.textContent = pileCaption(snapshot);
  }

  #showFinale(snapshot) {
    this.#finale.hidden = !snapshot.over;
    if (!snapshot.over) return;
    const rows = snapshot.view.placements.map((seat, place) => {
      const row = document.createElement("li");
      row.style.setProperty("--order", place);
      const medal = document.createElement("span");
      medal.className = `medal place-${place}`;
      medal.textContent = ORDINALS[place];
      const name = document.createElement("span");
      name.textContent = seatLabel(snapshot, seat);
      row.append(medal, name);
      return row;
    });
    const title = document.createElement("h2");
    title.textContent = snapshot.view.placements[0] === snapshot.human ? "You win" : "Game over";
    const list = document.createElement("ol");
    list.append(...rows);
    this.#finale.replaceChildren(title, list);
  }

  #clearFinale() {
    this.#finale.hidden = true;
  }

  #fly(entry, pose, duration, delay = 0, lifted = false) {
    const from = entry.pose;
    const frames = [{ transform: poseTransform(from) }];
    if (lifted) {
      const mid = { x: (from.x + pose.x) / 2, y: (from.y + pose.y) / 2, rot: (from.rot + pose.rot) / 2, scale: Math.max(from.scale, pose.scale) * 1.14 };
      frames.push({ transform: poseTransform(mid) });
    }
    frames.push({ transform: poseTransform(pose) });
    this.#running.add(travel(entry.element, frames, { duration, delay }));
    if (from.face !== pose.face) {
      const turn = [{ transform: from.face ? FACE_UP : FACE_DOWN }, { transform: pose.face ? FACE_UP : FACE_DOWN }];
      this.#running.add(travel(flipOf(entry.element), turn, { duration, delay }));
    }
    entry.element.style.zIndex = FLYING_Z + pose.z;
    entry.element.dataset.zone = pose.zone;
    entry.element.classList.toggle("stale", Boolean(pose.stale));
    entry.pose = pose;
    entry.element.style.transform = poseTransform(pose);
    flipOf(entry.element).style.transform = pose.face ? FACE_UP : FACE_DOWN;
  }

  #sweep(entry, target, duration, index) {
    const { element, pose } = entry;
    const end = { ...target, x: target.x + index * 3 };
    const frames = [
      { transform: poseTransform(pose), opacity: 1 },
      { transform: poseTransform(end), opacity: 0.9, offset: 0.8 },
      { transform: poseTransform(end), opacity: 0 },
    ];
    this.#running.add(travel(element, frames, { duration, delay: index * 12, sweep: true }));
    element.style.opacity = "0";
    this.#leaving.add(element);
  }

  #speech(seat, duration, text) {
    const plate = this.#plates.get(seat);
    const bubble = document.createElement("span");
    bubble.className = "bubble";
    bubble.textContent = text;
    plate.append(bubble);
    this.#running.add(pop(bubble, duration));
    this.#leaving.add(bubble);
  }

  #removeGone(wanted, layout, duration, moved) {
    let origin = null;
    let swept = 0;
    for (const [key, entry] of this.#cards) {
      if (wanted.has(key)) continue;
      this.#cards.delete(key);
      if (entry.pose.zone === "hand" && !entry.pose.face) {
        if (entry.pose.key.startsWith(`hidden:${moved.seat}:`)) origin = entry.pose;
        entry.element.remove();
      } else {
        this.#sweep(entry, layout.sweep, duration, swept++);
      }
    }
    return origin;
  }

  #move(layout, moved, duration) {
    const wanted = new Map(layout.cards.map((pose) => [pose.key, pose]));
    const origin = this.#removeGone(wanted, layout, duration, moved);
    const plate = layout.plates.find(({ seat }) => seat === moved.seat);
    const start = origin ?? { x: plate.x, y: plate.y, rot: 0, scale: layout.pile.card / CARD_WIDTH, z: 0, zone: "hand", face: false };
    if (!moved.play.cards.length) this.#speech(moved.seat, Math.max(duration, 320), "Pass");
    for (const pose of layout.cards) {
      const existing = this.#cards.get(pose.key);
      const entry = existing ?? this.#create(pose);
      if (!existing) this.#setPose(entry, { ...start, key: pose.key });
      if (poseTransform(entry.pose) === poseTransform(pose) && entry.pose.face === pose.face) {
        this.#setPose(entry, pose);
        continue;
      }
      const leavingHand = entry.pose.zone === "hand" && pose.zone === "pile";
      this.#fly(entry, pose, leavingHand ? duration : duration * 0.8, 0, leavingHand);
    }
  }

  #deal(layout, flight) {
    for (const entry of this.#cards.values()) {
      this.#running.add(travel(entry.element, [{ opacity: 1 }, { opacity: 0 }], { duration: SHUFFLE_OUT }));
      entry.element.style.opacity = "0";
      this.#leaving.add(entry.element);
    }
    this.#cards.clear();
    const deck = { ...layout.pile, rot: 0, scale: layout.pile.card / CARD_WIDTH, z: 0, zone: "deck", face: false };
    const total = layout.cards.length;
    const seats = layout.plates.length;
    const order = (i) => (i % (total / seats)) * seats + Math.floor(i / (total / seats));
    const step = total > 1 ? (flight * DEAL_SPREAD) / total : 0;
    layout.cards.forEach((pose, i) => {
      const entry = this.#create(pose);
      this.#setPose(entry, { ...deck, key: pose.key });
      this.#fly(entry, pose, flight * 0.8, order(i) * step);
    });
    this.#placePlates(layout);
  }

  #restingState(layout, snapshot) {
    const rests = layout.cards.map((pose) => [this.#cards.get(pose.key), pose]);
    return () => {
      for (const [entry, pose] of rests) if (entry) entry.element.style.zIndex = pose.z;
      this.#showPlates(snapshot);
      this.#showFinale(snapshot);
    };
  }

  show(snapshot, { kind, flight }) {
    this.#settle();
    this.#snapshot = snapshot;
    this.#clearFinale();
    const layout = layoutTable(snapshot, this.#size);
    const moved = snapshot.view.moves.at(-1);
    const animate = flight > 0 && motionAllowed() && (kind === "deal" || (kind === "move" && moved));
    if (!animate) {
      this.#snap(layout);
      this.#showPlates(snapshot);
      this.#showFinale(snapshot);
      return Promise.resolve();
    }

    this.#placePlates(layout);
    this.#note.textContent = "";
    if (kind === "deal") {
      this.#showPlates(snapshot);
      this.#note.textContent = "";
      this.#deal(layout, flight);
    } else {
      this.#move(layout, moved, moved.seat === snapshot.human ? Math.min(flight, HUMAN_FLIGHT) : flight);
    }
    this.#rest = this.#restingState(layout, snapshot);
    const token = this.#token;
    return Promise.allSettled([...this.#running].map((animation) => animation.finished)).then(() => {
      if (token === this.#token) this.#settle();
    });
  }

  select({ selected, playable }) {
    for (const [key, { element }] of this.#cards) {
      const enabled = playable.has(key);
      element.disabled = !enabled;
      element.classList.toggle("selected", selected.has(key));
      element.classList.toggle("dimmed", playable.size > 0 && !enabled && element.dataset.zone === "hand");
    }
  }
}
