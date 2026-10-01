/**
 * Generate the Death Star wireframe SVG for the Bot Stats tab
 * background. Static asset — committed to ``public/`` so the
 * dashboard uses it as a CSS ``background-image`` (no React,
 * no runtime cost).
 *
 * Re-run with::
 *
 *     cd frontend && npm run build:death-star
 *
 * Operator-supplied SVG design (ChatGPT 2026-05-09, JSX form);
 * this generator inlines the values (latitudes, longitudes,
 * trench, greebles, ticks) so they ship as a single static file.
 */
import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const OUT_PATH = join(__dirname, "..", "public", "death-star-wireframe.svg");

const W = 800;
const H = 800;
const CX = 400;
const CY = 400;
const R = 310;

// Latitude arcs: y-offset → projected ellipse rx (full projection
// of the sphere's intersection with the latitude plane). ry is
// flattened by 0.18 to give the equator the "trench tilt" look.
const LATITUDES = [-230, -180, -130, -85, -42, 0, 44, 90, 138, 190, 240];
const latArcs = LATITUDES.map((y) => {
  const rx = Math.sqrt(Math.max(0, R * R - y * y));
  const ry = rx * 0.18;
  const opacity = y === 0 ? 0.42 : 0.22;
  const sw = y === 0 ? 1.2 : 0.7;
  return `    <ellipse cx="${CX}" cy="${CY + y}" rx="${rx.toFixed(2)}" ry="${ry.toFixed(2)}" opacity="${opacity}" stroke-width="${sw}" />`;
}).join("\n");

// Longitude arcs: degrees east-west → flattened ellipse rx.
const LONGITUDES = [-70, -50, -32, -16, 0, 16, 32, 50, 70];
const lonArcs = LONGITUDES.map((deg) => {
  const rx = R * Math.cos((deg * Math.PI) / 180);
  return `    <ellipse cx="${CX}" cy="${CY}" rx="${rx.toFixed(2)}" ry="${R}" opacity="0.18" stroke-width="0.7" />`;
}).join("\n");

// Equatorial trench blocks — 34 small castellation paths along
// the equator band, varying heights for visual texture.
const trenchBlocks = Array.from({ length: 34 }, (_, i) => {
  const x = 92 + i * 18;
  const h = i % 3 === 0 ? 18 : i % 3 === 1 ? 12 : 8;
  const yOff = 386 + (i % 4);
  return `    <path d="M${x} ${yOff} L${x + 9} ${yOff} L${x + 9} ${386 + h}" opacity="0.35" stroke-width="0.65" />`;
}).join("\n");

// Greeble distribution — 120 short hash-marks scattered with the
// golden-angle pattern so they read as urban / mechanical detail
// without obvious rings.
const greebles = Array.from({ length: 120 }, (_, i) => {
  const angle = (i * 137.5 * Math.PI) / 180;
  const r = 35 + ((i * 23) % 265);
  const x = CX + Math.cos(angle) * r;
  const y = CY + Math.sin(angle) * r;
  const len = 5 + (i % 7);
  return `    <path d="M${x.toFixed(2)} ${y.toFixed(2)} h${len} M${(x + len + 3).toFixed(2)} ${y.toFixed(2)} h${(len / 2).toFixed(2)}" />`;
}).join("\n");

// Superlaser dish — 16 radial spokes from the central pinhole
// out to the dish rim.
const dishRays = Array.from({ length: 16 }, (_, i) => {
  const a = (i / 16) * Math.PI * 2;
  const x1 = (Math.cos(a) * 15).toFixed(2);
  const y1 = (Math.sin(a) * 15).toFixed(2);
  const x2 = (Math.cos(a) * 86).toFixed(2);
  const y2 = (Math.sin(a) * 86).toFixed(2);
  return `      <line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" opacity="0.35" stroke-width="0.7" />`;
}).join("\n");

// Outer HUD ticks — 48 ticks around the sphere; every 4th tick
// is longer + brighter to mark a "cardinal".
const hudTicks = Array.from({ length: 48 }, (_, i) => {
  const a = (i / 48) * Math.PI * 2;
  const r1 = 325;
  const r2 = i % 4 === 0 ? 350 : 338;
  const x1 = (CX + Math.cos(a) * r1).toFixed(2);
  const y1 = (CY + Math.sin(a) * r1).toFixed(2);
  const x2 = (CX + Math.cos(a) * r2).toFixed(2);
  const y2 = (CY + Math.sin(a) * r2).toFixed(2);
  const op = i % 4 === 0 ? 0.7 : 0.35;
  const sw = i % 4 === 0 ? 1.2 : 0.7;
  return `    <line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" opacity="${op}" stroke-width="${sw}" />`;
}).join("\n");

// Surface panel networks — long polyline arcs across the sphere
// suggesting equatorial / tropical tech corridors.
const panelNetworks = `
    <path d="M175 205 L250 185 L315 210 L380 188 L475 214 L560 190 L640 230" />
    <path d="M145 285 L230 255 L320 270 L390 245 L505 272 L610 260 L690 300" />
    <path d="M135 505 L220 485 L315 510 L405 492 L500 515 L615 500 L690 535" />
    <path d="M180 610 L265 570 L350 590 L445 565 L535 600 L615 575" />
    <path d="M230 160 L240 245 L210 305 L250 365" />
    <path d="M335 130 L320 225 L348 310 L325 370" />
    <path d="M465 120 L480 220 L455 300 L490 365" />
    <path d="M585 170 L555 250 L590 330 L565 380" />
    <path d="M210 455 L250 520 L235 595" />
    <path d="M330 438 L355 520 L340 650" />
    <path d="M475 435 L450 535 L480 675" />
    <path d="M610 440 L570 525 L595 610" />
`.trim();

// Rectangular armor panels — 12 rounded rects scattered in the
// non-trench zones to break up the surface visually.
const armorPanels = [
  [230, 235, 42, 20],
  [300, 250, 55, 24],
  [470, 245, 62, 22],
  [560, 300, 52, 18],
  [180, 330, 45, 16],
  [280, 520, 64, 20],
  [410, 540, 70, 22],
  [535, 565, 50, 18],
  [215, 620, 48, 18],
  [390, 170, 58, 18],
  [520, 180, 44, 18],
  [150, 455, 55, 20],
]
  .map(([x, y, w, h]) => `      <rect x="${x}" y="${y}" width="${w}" height="${h}" rx="2" />`)
  .join("\n");

const generatedAt = new Date().toISOString();

const svg = `<?xml version="1.0" encoding="UTF-8"?>
<!--
  Death Star wireframe. Auto-generated by
  frontend/scripts/build-death-star.mjs from the operator-supplied
  ChatGPT design (2026-05-09). Static decorative asset for the
  Bot Stats tab background.

  Generated:    ${generatedAt}
  ViewBox:      ${W}×${H}

  DO NOT hand-edit — the next "npm run build:death-star" overwrites.
-->
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}"
     preserveAspectRatio="xMidYMid meet" aria-hidden="true"
     opacity="0.98">
  <defs>
    <filter id="dsGlow">
      <feGaussianBlur stdDeviation="2.2" result="blur" />
      <feMerge>
        <feMergeNode in="blur" />
        <feMergeNode in="SourceGraphic" />
      </feMerge>
    </filter>
    <radialGradient id="dsCore" cx="42%" cy="35%" r="65%">
      <stop offset="0%" stop-color="rgba(6,182,212,0.16)" />
      <stop offset="55%" stop-color="rgba(6,182,212,0.06)" />
      <stop offset="100%" stop-color="rgba(6,182,212,0.015)" />
    </radialGradient>
    <clipPath id="sphereClip">
      <circle cx="${CX}" cy="${CY}" r="${R}" />
    </clipPath>
  </defs>

  <g filter="url(#dsGlow)" fill="none" stroke="rgba(6,182,212,0.82)"
     stroke-linecap="round" stroke-linejoin="round">

    <!-- Outer sphere -->
    <circle cx="${CX}" cy="${CY}" r="${R}"
            fill="url(#dsCore)" stroke="rgba(6,182,212,0.95)" stroke-width="2" />

    <g clip-path="url(#sphereClip)">

      <!-- Latitude arcs -->
${latArcs}

      <!-- Longitude arcs -->
${lonArcs}

      <!-- Equatorial trench (twin rims + dashed centre) -->
      <path d="M80 382 C205 365 310 362 400 368 C510 375 610 388 720 374"
            stroke="rgba(6,182,212,0.95)" stroke-width="2" />
      <path d="M78 421 C205 407 315 404 400 410 C510 417 612 430 722 416"
            stroke="rgba(6,182,212,0.95)" stroke-width="2" />
      <path d="M95 398 C220 386 315 386 398 391 C505 397 612 410 705 396"
            stroke="rgba(6,182,212,0.28)" stroke-dasharray="6 9" />

      <!-- Trench castellation blocks -->
${trenchBlocks}

      <!-- Surface panel networks -->
      <g opacity="0.46" stroke-width="0.75">
${panelNetworks}
      </g>

      <!-- Rectangular armor panels -->
      <g opacity="0.42" stroke-width="0.65">
${armorPanels}
      </g>

      <!-- City-like greebles -->
      <g opacity="0.36" stroke-width="0.55">
${greebles}
      </g>

      <!-- Superlaser dish -->
      <g transform="translate(270 275)">
        <circle cx="0" cy="0" r="88"
                fill="rgba(6,182,212,0.045)"
                stroke="rgba(6,182,212,0.95)" stroke-width="1.8" />
        <circle cx="0" cy="0" r="64" opacity="0.55" />
        <circle cx="0" cy="0" r="38" opacity="0.42" />
        <circle cx="0" cy="0" r="14"
                fill="rgba(245,158,11,0.65)"
                stroke="rgba(245,158,11,0.9)" />
${dishRays}
        <path d="M-70 -20 Q0 -70 70 -20" opacity="0.35" />
        <path d="M-70 20 Q0 70 70 20" opacity="0.35" />
        <path d="M-20 -70 Q-70 0 -20 70" opacity="0.3" />
        <path d="M20 -70 Q70 0 20 70" opacity="0.3" />
      </g>
    </g>

    <!-- Limb darkening / far-side curvature -->
    <path d="M610 150 C720 275 728 510 600 650"
          stroke="rgba(6,182,212,0.25)" stroke-width="7" opacity="0.35" />

    <!-- Outer HUD ticks -->
${hudTicks}

    <!-- Faint orbit rings -->
    <ellipse cx="${CX}" cy="${CY}" rx="370" ry="95" opacity="0.18" />
    <ellipse cx="${CX}" cy="${CY}" rx="390" ry="120" opacity="0.1"
             transform="rotate(-18 ${CX} ${CY})" />
  </g>
</svg>
`;

writeFileSync(OUT_PATH, svg, "utf8");
console.log(
  `wrote ${OUT_PATH} (${(svg.length / 1024).toFixed(1)} KB)`,
);
