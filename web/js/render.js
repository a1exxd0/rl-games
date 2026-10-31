import { suitIcon } from "./cards.js";

export const ORDINALS = ["1st", "2nd", "3rd", "4th"];

export function seatName(snapshot, seat) {
  return seat === snapshot.human ? "You" : snapshot.seats[seat];
}

export function seatLabel(snapshot, seat) {
  if (seat === snapshot.human) return "You";
  return `${snapshot.seats[seat]} (seat ${seat})`;
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function cardChip(card) {
  const chip = element("span", "log-card", card.slice(0, -1));
  chip.insertAdjacentHTML("beforeend", suitIcon(card.slice(-1)));
  return chip;
}

function logItem(snapshot, { seat, play }) {
  const item = element("li");
  item.dataset.seat = seat;
  const who = element("span", "who");
  who.append(element("span", "seat-dot"), element("span", "who-name", seatLabel(snapshot, seat)));
  const what = element("span", "what");
  if (play.cards.length) what.append(...play.cards.map(cardChip));
  else what.append(element("span", "log-pass", "pass"));
  item.append(who, what);
  return item;
}

export function renderLog(root, snapshot, { kind, animated }) {
  const { moves, round_start: roundStart } = snapshot.view;
  const appending = kind === "move" && root.children.length === moves.length - 1;
  if (appending) {
    const item = logItem(snapshot, moves.at(-1));
    if (animated) item.classList.add("arriving");
    root.append(item);
  } else {
    root.replaceChildren(...moves.map((move) => logItem(snapshot, move)));
  }
  [...root.children].forEach((item, index) => {
    item.classList.toggle("current-round", index >= roundStart);
  });
  root.scrollTo({ top: root.scrollHeight, behavior: appending && animated ? "smooth" : "auto" });
}

export function statusText(snapshot) {
  const { placements, to_act: toAct } = snapshot.view;
  if (snapshot.over) {
    const order = placements.map((seat, place) => `${ORDINALS[place]} ${seatLabel(snapshot, seat)}`);
    return `Game over: ${order.join(", ")}`;
  }
  if (toAct === snapshot.human) return "Your turn";
  return `${seatLabel(snapshot, toAct)} to act`;
}
