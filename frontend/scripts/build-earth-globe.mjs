/**
 * Generate an Earth wireframe globe SVG for the Bot Stats tab
 * background. Static asset — committed to ``public/`` so the
 * dashboard uses it as a CSS ``background-image``.
 *
 * Pipeline: Natural Earth countries-50m TopoJSON  →  d3-geo
 * orthographic projection rotated so Dubai is at the apex  →  the
 * full visible hemisphere rendered (countries + graticule + sphere
 * outline + Dubai crosshair) in the same cyan wireframe palette as
 * the regional maps.
 *
 * Re-run with::
 *
 *     cd frontend && npm run build:earth-globe
 */
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { feature as topoFeature } from "topojson-client";
import {
  geoOrthographic,
  geoPath,
  geoGraticule10,
  geoCircle,
} from "d3-geo";

const __dirname = dirname(fileURLToPath(import.meta.url));
const FRONTEND = join(__dirname, "..");
const TOPO_PATH = join(
  FRONTEND,
  "node_modules",
  "world-atlas",
  "countries-50m.json",
);
const OUT_PATH = join(FRONTEND, "public", "earth-globe-wireframe.svg");

// ---------------------------------------------------------------------------
// Knobs
// ---------------------------------------------------------------------------

const W = 800;
const H = 800;
const RADIUS = 360;
const CX = W / 2;
const CY = H / 2;

// Dubai apex. ``rotate`` takes [lambda, phi] in degrees and rotates
// the globe by ``-lon, -lat`` so the named point ends up at the
// projection centre (the apex facing the viewer).
const DUBAI_LONLAT = [55.27, 25.2];

// Cities visible on the Dubai-facing hemisphere. Anything more than
// ~90° great-circle distance from Dubai will be on the back side
// and the orthographic projection auto-clips it (returns null).
const CITIES = [
  { name: "DUBAI", lonlat: DUBAI_LONLAT, emphasis: true },
  // Middle East
  { name: "ABU DHABI", lonlat: [54.37, 24.45] },
  { name: "RIYADH", lonlat: [46.72, 24.63] },
  { name: "DOHA", lonlat: [51.53, 25.29] },
  { name: "TEHRAN", lonlat: [51.39, 35.69] },
  { name: "BAGHDAD", lonlat: [44.36, 33.31] },
  { name: "JERUSALEM", lonlat: [35.21, 31.78] },
  { name: "ISTANBUL", lonlat: [28.98, 41.01] },
  // South Asia
  { name: "MUMBAI", lonlat: [72.88, 19.08] },
  { name: "DELHI", lonlat: [77.21, 28.61] },
  { name: "KARACHI", lonlat: [67.01, 24.86] },
  { name: "DHAKA", lonlat: [90.41, 23.81] },
  // East / SE Asia (some on the visible-edge limb)
  { name: "BANGKOK", lonlat: [100.5, 13.75] },
  { name: "SINGAPORE", lonlat: [103.85, 1.35] },
  { name: "HONG KONG", lonlat: [114.17, 22.32] },
  { name: "BEIJING", lonlat: [116.41, 39.9] },
  { name: "SHANGHAI", lonlat: [121.47, 31.23] },
  // Africa
  { name: "CAIRO", lonlat: [31.23, 30.04] },
  { name: "LAGOS", lonlat: [3.39, 6.52] },
  { name: "NAIROBI", lonlat: [36.82, -1.29] },
  { name: "JOHANNESBURG", lonlat: [28.05, -26.2] },
  { name: "ADDIS ABABA", lonlat: [38.74, 9.03] },
  // Europe
  { name: "LONDON", lonlat: [-0.13, 51.51] },
  { name: "PARIS", lonlat: [2.35, 48.86] },
  { name: "ROME", lonlat: [12.5, 41.9] },
  { name: "MADRID", lonlat: [-3.7, 40.42] },
  { name: "BERLIN", lonlat: [13.4, 52.52] },
  { name: "MOSCOW", lonlat: [37.62, 55.75] },
  // Russia / Central Asia
  { name: "ASTANA", lonlat: [71.43, 51.13] },
];

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

const world = JSON.parse(readFileSync(TOPO_PATH, "utf8"));
const countries = topoFeature(world, world.objects.countries);

const projection = geoOrthographic()
  .scale(RADIUS)
  .translate([CX, CY])
  .rotate([-DUBAI_LONLAT[0], -DUBAI_LONLAT[1]])
  .clipAngle(90); // auto-clip the back hemisphere

const pathGen = geoPath(projection);

// 1. Sphere outline — d3-geo gives us this as a sphere feature.
const sphereD = pathGen({ type: "Sphere" }) || "";

// 2. Graticule (10° spacing).
const graticuleD = pathGen(geoGraticule10()) || "";

// 3. Countries — render every land feature; back side is auto-clipped.
const countryPaths = countries.features
  .map((f) => {
    const d = pathGen(f);
    if (!d) return "";
    return `    <path d="${d}" fill="rgba(6,182,212,0.04)" stroke="rgba(6,182,212,0.55)" stroke-width="0.6" stroke-linejoin="round" stroke-linecap="round" />`;
  })
  .filter(Boolean)
  .join("\n");

// 4. Dubai-centred concentric circles (visual emphasis on the apex).
const dubaiCircleSmall = pathGen(geoCircle().center(DUBAI_LONLAT).radius(2)()) || "";
const dubaiCircleMedium = pathGen(geoCircle().center(DUBAI_LONLAT).radius(8)()) || "";
const dubaiCircleLarge = pathGen(geoCircle().center(DUBAI_LONLAT).radius(20)()) || "";

// 5. City markers — only those on the visible hemisphere render.
const cityMarks = CITIES.map((c) => {
  const px = projection(c.lonlat);
  if (!px) return ""; // back side or invalid
  const [x, y] = px;
  if (!Number.isFinite(x) || !Number.isFinite(y)) return "";
  const r = c.emphasis ? 5 : 3;
  const fontSize = c.emphasis ? 14 : 11;
  const opacity = c.emphasis ? 0.95 : 0.7;
  const labelColor = c.emphasis
    ? "rgba(0,229,255,0.95)"
    : "rgba(6,182,212,0.85)";
  const fillColor = c.emphasis
    ? "rgba(245,158,11,1)"
    : "rgba(245,158,11,0.85)";
  return `  <g opacity="${opacity}">
    <circle cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="${r}" fill="${fillColor}" />
    <text x="${(x + 8).toFixed(2)}" y="${(y + 4).toFixed(2)}" font-size="${fontSize}" font-family="ui-monospace, monospace" letter-spacing="2" fill="${labelColor}">${c.name}</text>
  </g>`;
}).filter(Boolean).join("\n");

const generatedAt = new Date().toISOString();

const svg = `<?xml version="1.0" encoding="UTF-8"?>
<!--
  Earth wireframe globe centred on Dubai. Auto-generated by
  frontend/scripts/build-earth-globe.mjs from Natural Earth
  countries-50m TopoJSON via d3-geo orthographic projection.

  Generated:    ${generatedAt}
  Apex:         Dubai (${DUBAI_LONLAT.join(", ")})
  ViewBox:      ${W}×${H}

  DO NOT hand-edit — the next "npm run build:earth-globe" overwrites.
-->
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}"
     preserveAspectRatio="xMidYMid meet" aria-hidden="true"
     opacity="0.98">
  <defs>
    <radialGradient id="globeCore" cx="50%" cy="50%" r="55%">
      <stop offset="0%" stop-color="rgba(6,182,212,0.10)" />
      <stop offset="80%" stop-color="rgba(6,182,212,0.03)" />
      <stop offset="100%" stop-color="rgba(6,182,212,0)" />
    </radialGradient>
    <radialGradient id="globeLimb" cx="50%" cy="50%" r="50%">
      <stop offset="80%" stop-color="rgba(0,0,0,0)" />
      <stop offset="100%" stop-color="rgba(6,182,212,0.18)" />
    </radialGradient>
  </defs>

  <g fill="none" stroke="rgba(6,182,212,0.82)"
     stroke-linecap="round" stroke-linejoin="round">

    <!-- Sphere fill (very faint cyan core) -->
    <path d="${sphereD}" fill="url(#globeCore)" stroke="none" />

    <!-- Graticule: 10° lat/lon grid -->
    <path d="${graticuleD}" stroke="rgba(6,182,212,0.22)" stroke-width="0.5" />

    <!-- Country borders / land masses -->
${countryPaths}

    <!-- Dubai apex emphasis: three concentric great-circles -->
    <path d="${dubaiCircleLarge}"
          stroke="rgba(0,229,255,0.45)" stroke-width="1.0"
          stroke-dasharray="3 4" />
    <path d="${dubaiCircleMedium}"
          stroke="rgba(0,229,255,0.65)" stroke-width="1.2" />
    <path d="${dubaiCircleSmall}"
          stroke="rgba(0,229,255,0.95)" stroke-width="1.4"
          fill="rgba(0,229,255,0.18)" />

    <!-- Sphere outline (the globe's silhouette) -->
    <path d="${sphereD}"
          stroke="rgba(0,229,255,0.9)" stroke-width="2" />

    <!-- Limb-darkening overlay -->
    <path d="${sphereD}" fill="url(#globeLimb)" stroke="none" />

  </g>

  <!-- City markers + labels (above the globe wireframe) -->
${cityMarks}

  <!-- Outer HUD ring (matches the spaceship-globe component aesthetic) -->
  <g fill="none" stroke="rgba(6,182,212,0.4)" stroke-width="0.7">
    <circle cx="${CX}" cy="${CY}" r="${RADIUS + 18}" stroke-dasharray="0.5 1.5" />
  </g>

  <!-- Outer tick marks (cardinal-emphasised) -->
  <g stroke="rgba(6,182,212,0.55)">
${Array.from({ length: 24 }, (_, i) => {
  const a = (i / 24) * Math.PI * 2 - Math.PI / 2;
  const r1 = RADIUS + 28;
  const r2 = i % 6 === 0 ? RADIUS + 50 : RADIUS + 38;
  const x1 = (CX + Math.cos(a) * r1).toFixed(2);
  const y1 = (CY + Math.sin(a) * r1).toFixed(2);
  const x2 = (CX + Math.cos(a) * r2).toFixed(2);
  const y2 = (CY + Math.sin(a) * r2).toFixed(2);
  const sw = i % 6 === 0 ? 1.4 : 0.7;
  const op = i % 6 === 0 ? 0.85 : 0.5;
  return `    <line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke-width="${sw}" opacity="${op}" />`;
}).join("\n")}
  </g>
</svg>
`;

writeFileSync(OUT_PATH, svg, "utf8");
console.log(
  `wrote ${OUT_PATH} (${(svg.length / 1024).toFixed(1)} KB) — orthographic globe centred on Dubai, ${cityMarks ? CITIES.length : 0} cities listed`,
);
