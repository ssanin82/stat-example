import type { Config } from "tailwindcss";

const config: Config = {
  // Dark mode is the default (set via `<html class="dark">` in
  // app/layout.tsx). The 'class' strategy lets us toggle later if
  // we ever want a light option, without a full rebuild.
  darkMode: "class",
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Subtle grayscale palette tuned for an always-on trading
        // panel. Avoid pure black/white -- both fatigue the eyes
        // when staring at numbers for an hour.
        bg: {
          DEFAULT: "#0b0d10",
          card: "#15181d",
          border: "#22272e",
        },
        accent: {
          green: "#3ddc97",
          red: "#ff6b6b",
          amber: "#f59e0b",
          // Phase 4G (v1.4.214) — CALM mode + forward-signal CALM
          // classification colour. Distinct from green (NORMAL) so
          // the operator can tell "aggressive upshift" from "steady
          // state" at a glance. Cool teal-blue plays nice with the
          // dark Spaceship theme.
          blue: "#4cc9f0",
          // v1.5.33 — TP (take-profit) marker colour. Distinct from
          // the amber SF accent and the teal-blue CALM accent so
          // operators can tell SF / TP / regime markers apart at a
          // glance on the PnL sub-bands and the Stats/History rows.
          // Saturated violet keeps strong contrast on the dark bg.
          violet: "#a855f7",
        },
      },
      fontFamily: {
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "Consolas", "monospace"],
      },
    },
  },
  plugins: [],
};

export default config;
