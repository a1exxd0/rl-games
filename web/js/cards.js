const WIDTH = 250;
const HEIGHT = 350;
const LEFT = 78;
const MIDDLE = 125;
const RIGHT = 172;
const PIP = 58;

const SUIT_NAMES = { S: "spades", C: "clubs", D: "diamonds", H: "hearts" };

const SUIT_SHAPES = {
  S: '<path d="M50 3C62 24 93 40 93 63c0 13-10 22-21 22-7 0-13-3-17-9 1 9 5 15 12 21H33c7-6 11-12 12-21-4 6-10 9-17 9C17 85 7 76 7 63 7 40 38 24 50 3Z"/>',
  H: '<path d="M50 92C20 64 5 48 5 29 5 14 17 5 30 5c9 0 16 5 20 14C54 10 61 5 70 5c13 0 25 9 25 24 0 19-15 35-45 63Z"/>',
  D: '<path d="M50 2c9 18 22 33 38 48-16 15-29 30-38 48-9-18-22-33-38-48C28 35 41 20 50 2Z"/>',
  C: '<circle cx="50" cy="27" r="22"/><circle cx="26" cy="58" r="22"/><circle cx="74" cy="58" r="22"/><path d="M45 50h10c1 20 5 34 13 46H32c8-12 12-26 13-46Z"/>',
};

const PIP_LAYOUTS = {
  2: [[MIDDLE, 0.2], [MIDDLE, 0.8]],
  3: [[MIDDLE, 0.2], [MIDDLE, 0.5], [MIDDLE, 0.8]],
  4: [[LEFT, 0.2], [RIGHT, 0.2], [LEFT, 0.8], [RIGHT, 0.8]],
  5: [[LEFT, 0.2], [RIGHT, 0.2], [MIDDLE, 0.5], [LEFT, 0.8], [RIGHT, 0.8]],
  6: [[LEFT, 0.2], [RIGHT, 0.2], [LEFT, 0.5], [RIGHT, 0.5], [LEFT, 0.8], [RIGHT, 0.8]],
  7: [[LEFT, 0.2], [RIGHT, 0.2], [MIDDLE, 0.35], [LEFT, 0.5], [RIGHT, 0.5], [LEFT, 0.8], [RIGHT, 0.8]],
  8: [[LEFT, 0.2], [RIGHT, 0.2], [MIDDLE, 0.35], [LEFT, 0.5], [RIGHT, 0.5], [MIDDLE, 0.65], [LEFT, 0.8], [RIGHT, 0.8]],
  9: [[LEFT, 0.2], [RIGHT, 0.2], [LEFT, 0.4], [RIGHT, 0.4], [MIDDLE, 0.5], [LEFT, 0.6], [RIGHT, 0.6], [LEFT, 0.8], [RIGHT, 0.8]],
  10: [[LEFT, 0.2], [RIGHT, 0.2], [MIDDLE, 0.3], [LEFT, 0.4], [RIGHT, 0.4], [LEFT, 0.6], [RIGHT, 0.6], [MIDDLE, 0.7], [LEFT, 0.8], [RIGHT, 0.8]],
};

const DEFS = `
<svg class="card-defs" width="0" height="0" aria-hidden="true">
  <defs>
    ${Object.entries(SUIT_SHAPES)
      .map(([suit, shape]) => `<symbol id="suit-${suit}" viewBox="0 0 100 100">${shape}</symbol>`)
      .join("")}
    <pattern id="back-lattice" width="26" height="26" patternUnits="userSpaceOnUse" patternTransform="rotate(45 0 0)">
      <rect width="26" height="26" fill="#1d3557"/>
      <path d="M0 0h26M0 0v26" stroke="#e8c877" stroke-width="2" opacity="0.55"/>
      <circle cx="13" cy="13" r="3" fill="#e8c877" opacity="0.5"/>
    </pattern>
    <linearGradient id="card-paper" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#fffefa"/>
      <stop offset="1" stop-color="#f3eee1"/>
    </linearGradient>
  </defs>
</svg>`;

const svg = (body, className) =>
  `<svg class="${className}" viewBox="0 0 ${WIDTH} ${HEIGHT}" aria-hidden="true">${body}</svg>`;

function pip(suit, x, y, size) {
  const flip = y > HEIGHT / 2 ? ` transform="rotate(180 ${x} ${y})"` : "";
  const half = size / 2;
  return `<use href="#suit-${suit}" x="${x - half}" y="${y - half}" width="${size}" height="${size}"${flip}/>`;
}

function index(rank, suit) {
  const squeeze = rank === "10" ? ' textLength="48" lengthAdjust="spacingAndGlyphs"' : "";
  return `<g>
    <text class="card-index" x="36" y="70" text-anchor="middle"${squeeze}>${rank}</text>
    ${pip(suit, 36, 98, 40)}
  </g>`;
}

const rotatedIndex = (rank, suit) =>
  `<g transform="rotate(180 ${WIDTH / 2} ${HEIGHT / 2})">${index(rank, suit)}</g>`;

function aceCentre(suit) {
  return `<circle cx="${MIDDLE}" cy="175" r="78" fill="none" stroke="currentColor" stroke-width="2" opacity="0.28"/>
    ${pip(suit, MIDDLE, 175, 112)}`;
}

function courtCentre(rank, suit) {
  return `<rect x="66" y="42" width="118" height="266" rx="10" fill="currentColor" opacity="0.07"/>
    <rect x="66" y="42" width="118" height="266" rx="10" fill="none" stroke="currentColor" stroke-width="3" opacity="0.55"/>
    ${pip(suit, 94, 78, 40)}
    <g transform="rotate(180 ${WIDTH / 2} ${HEIGHT / 2})">${pip(suit, 94, 78, 40)}</g>
    <text class="card-court" x="${MIDDLE}" y="222" text-anchor="middle">${rank}</text>`;
}

function centre(rank, suit) {
  if (rank === "A") return aceCentre(suit);
  if ("JQK".includes(rank)) return courtCentre(rank, suit);
  return PIP_LAYOUTS[rank].map(([x, y]) => pip(suit, x, y * HEIGHT, PIP)).join("");
}

const paper = `<rect width="${WIDTH}" height="${HEIGHT}" rx="20" fill="url(#card-paper)"/>`;

const faces = new Map();

export function cardFront(card) {
  if (!faces.has(card)) {
    const rank = card.slice(0, -1);
    const suit = card.slice(-1);
    const colour = "DH".includes(suit) ? "card-red" : "card-black";
    const body = `${paper}<g class="${colour}">${index(rank, suit)}${centre(rank, suit)}${rotatedIndex(rank, suit)}</g>`;
    faces.set(card, svg(body, "card-face front"));
  }
  return faces.get(card);
}

export const CARD_BACK = svg(
  `<rect width="${WIDTH}" height="${HEIGHT}" rx="20" fill="#f4efe2"/>
   <rect x="14" y="14" width="${WIDTH - 28}" height="${HEIGHT - 28}" rx="12" fill="url(#back-lattice)" stroke="#e8c877" stroke-width="3"/>
   <path d="M125 118 182 175 125 232 68 175Z" fill="#1d3557" stroke="#e8c877" stroke-width="4"/>
   <g fill="#e8c877"><use href="#suit-S" x="101" y="151" width="48" height="48"/></g>`,
  "card-face back",
);

export const cardName = (card) => `${card.slice(0, -1)} of ${SUIT_NAMES[card.slice(-1)]}`;

export function suitIcon(suit) {
  const colour = "DH".includes(suit) ? "card-red" : "card-black";
  return `<svg class="suit-icon ${colour}" viewBox="0 0 100 100" aria-hidden="true"><use href="#suit-${suit}"/></svg>`;
}

export const installCardDefs = () => document.body.insertAdjacentHTML("afterbegin", DEFS);
