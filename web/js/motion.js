const MAX_FLIGHT = 700;
const FLIGHT_SHARE = 0.7;
const FLIGHT_EASING = "cubic-bezier(0.2, 0.75, 0.25, 1)";
const SWEEP_EASING = "cubic-bezier(0.55, 0, 0.75, 0.3)";

const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");

export const motionAllowed = () => !reducedMotion.matches;

export function timing(delay) {
  if (!delay || reducedMotion.matches) return { flight: 0, rest: delay };
  const flight = Math.min(delay * FLIGHT_SHARE, MAX_FLIGHT);
  return { flight, rest: delay - flight };
}

export const transformOf = ({ x, y, rot = 0, scale = 1 }, width, height) =>
  `translate(${x - width / 2}px, ${y - height / 2}px) rotate(${rot}deg) scale(${scale})`;

export function travel(element, frames, { duration, delay = 0, sweep = false }) {
  return element.animate(frames, {
    duration,
    delay,
    easing: sweep ? SWEEP_EASING : FLIGHT_EASING,
    fill: "backwards",
  });
}

export const pop = (element, duration) =>
  element.animate(
    [
      { transform: "translate(-50%, 6px) scale(0.6)", opacity: 0 },
      { transform: "translate(-50%, -6px) scale(1.05)", opacity: 1, offset: 0.25 },
      { transform: "translate(-50%, -8px) scale(1)", opacity: 1, offset: 0.75 },
      { transform: "translate(-50%, -14px) scale(1)", opacity: 0 },
    ],
    { duration, easing: "ease-out" },
  );
