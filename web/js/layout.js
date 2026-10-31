export const CARD_WIDTH = 100;
export const CARD_HEIGHT = 140;

const RATIO = CARD_HEIGHT / CARD_WIDTH;
const PLATE_HEIGHT = 42;
const GAP = 8;
const ROW_GAP = 6;
const ARC_DEGREES = 5;
const COMPACT_WIDTH = 560;
const SWEEP_SHRINK = 0.7;

const POSITIONS = {
  2: ["bottom", "top"],
  3: ["bottom", "right", "left"],
  4: ["bottom", "right", "top", "left"],
};

const hashOf = (text) => [...text].reduce((hash, char) => (hash * 31 + char.charCodeAt(0)) % 9973, 7);

function metrics({ width, height }, count) {
  const compact = width < COMPACT_WIDTH;
  return {
    width,
    height,
    compact,
    pad: compact ? 12 : 22,
    plateWidth: compact ? 108 : 156,
    humanCard: compact ? 48 : Math.min(84, width * 0.1, height * 0.14),
    openCard: compact ? (count === 4 ? 36 : 42) : Math.min(count === 4 ? 54 : 70, height * 0.105),
    hiddenCard: compact ? 30 : 44,
    pileCard: compact ? 46 : Math.min(74, width * 0.085, height * 0.125),
  };
}

function handCards(snapshot, seat) {
  const hand = snapshot.hands[seat];
  if (hand !== null) return hand.map((card) => ({ key: card, card, face: true }));
  const count = snapshot.view.hand_counts[seat];
  return Array.from({ length: count }, (_, i) => ({ key: `hidden:${seat}:${i}`, card: null, face: false }));
}

function describeHand(snapshot, seat, position, m) {
  const cards = handCards(snapshot, seat);
  const human = seat === snapshot.human;
  const hidden = snapshot.hands[seat] === null;
  const width = human ? m.humanCard : hidden ? m.hiddenCard : m.openCard;
  const rows = human && m.compact ? 2 : 1;
  return { seat, position, cards, human, hidden, width, rows, height: width * RATIO };
}

const bandOf = (hand, m) => {
  if (!hand) return m.pad;
  const handHeight = hand.height * hand.rows + ROW_GAP * (hand.rows - 1);
  return m.pad + handHeight + GAP + PLATE_HEIGHT + GAP;
};

function rowPoses(cards, hand, m, y, direction) {
  const count = cards.length;
  const middle = (count - 1) / 2;
  const reach = m.width - 2 * m.pad - hand.width;
  const preferred = hand.width * (hand.hidden ? 0.24 : 0.74);
  const spacing = count > 1 ? Math.min(preferred, reach / (count - 1)) : 0;
  const arc = hand.rows > 1 ? 0 : Math.min(count * 0.6, ARC_DEGREES);
  return cards.map((card, i) => {
    const t = count > 1 ? (i - middle) / middle : 0;
    return {
      ...card,
      x: m.width / 2 + (i - middle) * spacing,
      y: y + direction * t * t * hand.height * (arc ? 0.05 : 0),
      rot: direction * t * arc,
      scale: hand.width / CARD_WIDTH,
      z: i,
      zone: "hand",
    };
  });
}

function bottomPoses(hand, m) {
  const lowest = m.height - m.pad - hand.height / 2;
  if (hand.rows === 1) return rowPoses(hand.cards, hand, m, lowest, 1);
  const split = Math.ceil(hand.cards.length / 2);
  const upper = rowPoses(hand.cards.slice(0, split), hand, m, lowest - hand.height - ROW_GAP, 1);
  const lower = rowPoses(hand.cards.slice(split), hand, m, lowest, 1);
  return [...upper, ...lower];
}

function columnPoses(hand, m, top, bottom) {
  const count = hand.cards.length;
  const x = hand.position === "left" ? m.pad + hand.width / 2 : m.width - m.pad - hand.width / 2;
  const middle = (count - 1) / 2;
  const preferred = hand.height * (hand.hidden ? 0.17 : 0.42);
  const spacing = count > 1 ? Math.min(preferred, (bottom - top - hand.height) / (count - 1)) : 0;
  const centre = (top + bottom) / 2;
  return hand.cards.map((card, i) => ({
    ...card,
    x,
    y: centre + (i - middle) * spacing,
    rot: 0,
    scale: hand.width / CARD_WIDTH,
    z: i,
    zone: "hand",
  }));
}

function posesFor(hand, m, top, bottom) {
  if (hand.position === "bottom") return bottomPoses(hand, m);
  if (hand.position === "top") return rowPoses(hand.cards, hand, m, m.pad + hand.height / 2, -1);
  return columnPoses(hand, m, top, bottom);
}

function plateFor(hand, m, top, bottom) {
  const { position, seat, width } = hand;
  const handReach = hand.height * hand.rows + ROW_GAP * (hand.rows - 1);
  if (position === "bottom") {
    return { seat, position, x: m.width / 2, y: m.height - m.pad - handReach - GAP - PLATE_HEIGHT / 2 };
  }
  if (position === "top") {
    return { seat, position, x: m.width / 2, y: m.pad + handReach + GAP + PLATE_HEIGHT / 2 };
  }
  const inset = m.pad + width + GAP + m.plateWidth / 2;
  const x = position === "left" ? inset : m.width - inset;
  return { seat, position, x, y: m.compact ? top + PLATE_HEIGHT / 2 : (top + bottom) / 2 };
}

function playPoses(move, index, last, m, pile) {
  const { cards } = move.play;
  const hash = hashOf(cards[0]);
  const newest = index === last;
  const width = newest ? m.pileCard : m.pileCard * 0.92;
  const spread = m.pileCard * 0.3;
  const offsetX = newest ? 0 : ((hash % 21) / 10 - 1) * spread;
  const offsetY = newest ? 0 : ((hash % 13) / 6 - 1) * spread;
  const tilt = newest ? (hash % 5) - 2 : (hash % 17) - 8;
  const reach = pile.reach - width;
  const spacing = cards.length > 1 ? Math.min(width * 0.62, reach / (cards.length - 1)) : 0;
  return cards.map((card, i) => ({
    key: card,
    card,
    face: true,
    x: pile.x + offsetX + (i - (cards.length - 1) / 2) * spacing,
    y: pile.y + offsetY,
    rot: tilt,
    scale: width / CARD_WIDTH,
    z: 100 + index * 20 + i,
    zone: "pile",
    stale: !newest,
  }));
}

function pilePoses(view, m, pile) {
  const plays = view.moves.slice(view.round_start).filter((move) => move.play.cards.length);
  return plays.flatMap((move, index) => playPoses(move, index, plays.length - 1, m, pile));
}

export function layoutTable(snapshot, size) {
  const { view } = snapshot;
  const count = view.rules.n_players;
  const m = metrics(size, count);
  const hands = POSITIONS[count].map((position, offset) =>
    describeHand(snapshot, (view.seat + offset) % count, position, m),
  );
  const handAt = (position) => hands.find((hand) => hand.position === position);
  const top = bandOf(handAt("top"), m);
  const bottom = m.height - bandOf(handAt("bottom"), m);
  const side = handAt("left") ?? handAt("right");
  const sideBand = side ? m.pad + side.width + GAP + (m.compact ? 0 : m.plateWidth) : m.pad;
  const pile = {
    x: m.width / 2,
    y: (top + bottom) / 2,
    reach: m.width - 2 * sideBand,
    card: m.pileCard,
  };

  return {
    cards: [...hands.flatMap((hand) => posesFor(hand, m, top, bottom)), ...pilePoses(view, m, pile)],
    plates: hands.map((hand) => plateFor(hand, m, top, bottom)),
    plateWidth: m.plateWidth,
    pile,
    sweep: { x: m.width * 0.9, y: pile.y - pile.card, rot: 28, scale: (m.pileCard / CARD_WIDTH) * SWEEP_SHRINK },
  };
}
