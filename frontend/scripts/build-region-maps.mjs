/**
 * Build real-coordinate wireframe SVG backgrounds for the Stats panel
 * tabs. Each region in the ``REGIONS`` array below produces one
 * static SVG file checked into ``frontend/public/`` so the dashboard
 * uses them as cheap CSS ``background-image``s (no React component,
 * no per-render cost).
 *
 * Pipeline per region:  Natural Earth countries-50m TopoJSON → d3-geo
 * Mercator projection (per-region centre + scale) → bounding-box
 * filter → styled SVG paths → file.
 *
 * Re-run with::
 *
 *     cd frontend && npm run build:map
 *
 * Add a region by appending a new entry to ``REGIONS``; the script
 * will emit one SVG per entry.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { feature as topoFeature } from "topojson-client";
import { geoMercator, geoPath, geoBounds } from "d3-geo";

const __dirname = dirname(fileURLToPath(import.meta.url));
const FRONTEND = join(__dirname, "..");
const TOPO_PATH = join(
  FRONTEND,
  "node_modules",
  "world-atlas",
  "countries-50m.json",
);
const PUBLIC_DIR = join(FRONTEND, "public");

// ---------------------------------------------------------------------------
// Shared knobs
// ---------------------------------------------------------------------------

// ViewBox aspect tuned to the Stats panel's tab body, which is
// ~3:1 wide-to-tall at the default row count. With CSS
// ``background-size: cover`` and a matching SVG aspect, the image
// fills the tab edge-to-edge with minimal cropping.
const W = 1800;
const H = 600;

const DEFAULT_COUNTRY_STYLE = {
  fill: "rgba(6,182,212,0.035)",
  stroke: "rgba(6,182,212,0.55)",
  strokeWidth: 0.7,
};

// ---------------------------------------------------------------------------
// Per-region config
// ---------------------------------------------------------------------------

const REGIONS = [
  // -------------------------------------------------------------------------
  // Persian Gulf — Config tab background. UAE-centred.
  // -------------------------------------------------------------------------
  {
    id: "persian-gulf",
    center: [54.3, 25.3],
    scale: 2400,
    bounds: { west: 25, east: 85, south: 5, north: 45 },
    highlight: {
      "United Arab Emirates": {
        fill: "rgba(0,229,255,0.18)",
        stroke: "rgba(0,229,255,1)",
        strokeWidth: 1.6,
      },
      Oman: {
        fill: "rgba(245,158,11,0.07)",
        stroke: "rgba(245,158,11,0.7)",
        strokeWidth: 1.0,
      },
    },
    crosshairLatLon: [54.5, 24.7],
    cities: [
      { name: "ABU DHABI", lonlat: [54.37, 24.45], emphasis: true },
      { name: "DUBAI", lonlat: [55.27, 25.2], emphasis: true },
      { name: "DOHA", lonlat: [51.53, 25.29] },
      { name: "MANAMA", lonlat: [50.58, 26.23] },
      { name: "RIYADH", lonlat: [46.72, 24.63] },
      { name: "KUWAIT CITY", lonlat: [47.97, 29.38] },
      { name: "TEHRAN", lonlat: [51.39, 35.69] },
      { name: "MUSCAT", lonlat: [58.59, 23.61] },
      { name: "BAGHDAD", lonlat: [44.36, 33.31] },
      { name: "CAIRO", lonlat: [31.23, 30.04] },
      { name: "AMMAN", lonlat: [35.93, 31.95] },
      { name: "DAMASCUS", lonlat: [36.3, 33.51] },
      { name: "BEIRUT", lonlat: [35.51, 33.89] },
      { name: "ISTANBUL", lonlat: [28.98, 41.01] },
      { name: "ANKARA", lonlat: [32.85, 39.93] },
      { name: "JERUSALEM", lonlat: [35.21, 31.78] },
      { name: "KARACHI", lonlat: [67.01, 24.86] },
      { name: "ISLAMABAD", lonlat: [73.05, 33.68] },
      { name: "MECCA", lonlat: [39.83, 21.42] },
      { name: "JEDDAH", lonlat: [39.2, 21.49] },
      { name: "SANAA", lonlat: [44.21, 15.35] },
      { name: "ADEN", lonlat: [45.04, 12.78] },
      { name: "ISFAHAN", lonlat: [51.67, 32.65] },
      { name: "SHIRAZ", lonlat: [52.53, 29.59] },
      { name: "ASTANA", lonlat: [71.43, 51.13] },
    ],
  },

  // -------------------------------------------------------------------------
  // Singapore — Open Orders tab background. SE-Asia city-state centre.
  // Scale chosen to keep Malaysia, Indonesia, Thailand, Vietnam, and
  // the Philippines visible while Singapore stays the focal point.
  // Singapore itself is geographically tiny — the crosshair marker is
  // the operator's main visual cue for "where the country is".
  // -------------------------------------------------------------------------
  {
    id: "singapore",
    center: [103.85, 1.35],
    scale: 3000,
    bounds: { west: 75, east: 135, south: -15, north: 25 },
    highlight: {
      Singapore: {
        fill: "rgba(0,229,255,0.32)",
        stroke: "rgba(0,229,255,1)",
        strokeWidth: 2.0,
      },
      Malaysia: {
        fill: "rgba(245,158,11,0.07)",
        stroke: "rgba(245,158,11,0.7)",
        strokeWidth: 1.0,
      },
    },
    crosshairLatLon: [103.85, 1.35],
    cities: [
      { name: "SINGAPORE", lonlat: [103.85, 1.35], emphasis: true },
      { name: "KUALA LUMPUR", lonlat: [101.7, 3.14], emphasis: true },
      { name: "JAKARTA", lonlat: [106.85, -6.21] },
      { name: "BANGKOK", lonlat: [100.5, 13.75] },
      { name: "MANILA", lonlat: [120.98, 14.6] },
      { name: "HO CHI MINH", lonlat: [106.66, 10.76] },
      { name: "HONG KONG", lonlat: [114.17, 22.32] },
      { name: "PHNOM PENH", lonlat: [104.92, 11.55] },
      { name: "VIENTIANE", lonlat: [102.6, 17.97] },
      { name: "HANOI", lonlat: [105.85, 21.03] },
      { name: "YANGON", lonlat: [96.16, 16.87] },
      { name: "NAYPYIDAW", lonlat: [96.1, 19.74] },
      { name: "SURABAYA", lonlat: [112.75, -7.25] },
      { name: "MEDAN", lonlat: [98.68, 3.59] },
      { name: "PADANG", lonlat: [100.36, -0.95] },
      { name: "BANDAR SERI BEGAWAN", lonlat: [114.94, 4.89] },
      { name: "CEBU", lonlat: [123.89, 10.32] },
      { name: "DAVAO", lonlat: [125.61, 7.07] },
      { name: "TAIPEI", lonlat: [121.57, 25.04] },
      { name: "DENPASAR", lonlat: [115.21, -8.67] },
      { name: "DA NANG", lonlat: [108.2, 16.05] },
    ],
  },

  // -------------------------------------------------------------------------
  // Jakarta — Order History tab background. Indonesia centre.
  // Wider lon/lat window than Singapore to surface the Indonesian
  // archipelago (Java, Sumatra, Borneo, Sulawesi) plus neighbours
  // (Malaysia, Singapore, Philippines south, Australia north).
  // Indonesia spans ~5000 km E-W so the scale is moderate.
  // -------------------------------------------------------------------------
  {
    id: "jakarta",
    center: [106.85, -6.21],
    scale: 2000,
    bounds: { west: 80, east: 145, south: -25, north: 15 },
    highlight: {
      Indonesia: {
        fill: "rgba(0,229,255,0.22)",
        stroke: "rgba(0,229,255,1)",
        strokeWidth: 1.8,
      },
      Malaysia: {
        fill: "rgba(245,158,11,0.07)",
        stroke: "rgba(245,158,11,0.7)",
        strokeWidth: 1.0,
      },
    },
    crosshairLatLon: [106.85, -6.21],
    cities: [
      { name: "JAKARTA", lonlat: [106.85, -6.21], emphasis: true },
      { name: "SURABAYA", lonlat: [112.75, -7.25], emphasis: true },
      { name: "BANDUNG", lonlat: [107.6, -6.91] },
      { name: "MEDAN", lonlat: [98.68, 3.59] },
      { name: "SINGAPORE", lonlat: [103.85, 1.35] },
      { name: "KUALA LUMPUR", lonlat: [101.7, 3.14] },
      { name: "MANILA", lonlat: [120.98, 14.6] },
      { name: "PERTH", lonlat: [115.86, -31.95] },
      { name: "DENPASAR", lonlat: [115.21, -8.67] },
      { name: "MAKASSAR", lonlat: [119.43, -5.13] },
      { name: "YOGYAKARTA", lonlat: [110.36, -7.8] },
      { name: "SEMARANG", lonlat: [110.42, -6.99] },
      { name: "PADANG", lonlat: [100.36, -0.95] },
      { name: "KUCHING", lonlat: [110.34, 1.55] },
      { name: "MANADO", lonlat: [124.83, 1.49] },
      { name: "BANDAR SERI BEGAWAN", lonlat: [114.94, 4.89] },
      { name: "PHNOM PENH", lonlat: [104.92, 11.55] },
      { name: "HO CHI MINH", lonlat: [106.66, 10.76] },
      { name: "BANGKOK", lonlat: [100.5, 13.75] },
      { name: "DARWIN", lonlat: [130.84, -12.46] },
      { name: "DILI", lonlat: [125.58, -8.55] },
      { name: "PORT MORESBY", lonlat: [147.18, -9.45] },
      { name: "CEBU", lonlat: [123.89, 10.32] },
    ],
  },

  // -------------------------------------------------------------------------
  // Bangkok — Fill History tab background. Mainland Southeast Asia
  // centre. Highlights Thailand; neighbours (Cambodia, Laos, Vietnam,
  // Myanmar, Malaysia north) all visible at this scale.
  // -------------------------------------------------------------------------
  {
    id: "bangkok",
    center: [100.5, 13.75],
    scale: 2400,
    bounds: { west: 75, east: 130, south: -5, north: 35 },
    highlight: {
      Thailand: {
        fill: "rgba(0,229,255,0.22)",
        stroke: "rgba(0,229,255,1)",
        strokeWidth: 1.8,
      },
      Cambodia: {
        fill: "rgba(245,158,11,0.07)",
        stroke: "rgba(245,158,11,0.7)",
        strokeWidth: 1.0,
      },
      Laos: {
        fill: "rgba(245,158,11,0.07)",
        stroke: "rgba(245,158,11,0.7)",
        strokeWidth: 1.0,
      },
    },
    crosshairLatLon: [100.5, 13.75],
    cities: [
      { name: "BANGKOK", lonlat: [100.5, 13.75], emphasis: true },
      { name: "HANOI", lonlat: [105.85, 21.03], emphasis: true },
      { name: "HO CHI MINH", lonlat: [106.66, 10.76] },
      { name: "PHNOM PENH", lonlat: [104.92, 11.55] },
      { name: "VIENTIANE", lonlat: [102.6, 17.97] },
      { name: "YANGON", lonlat: [96.16, 16.87] },
      { name: "KUALA LUMPUR", lonlat: [101.7, 3.14] },
      { name: "SINGAPORE", lonlat: [103.85, 1.35] },
      { name: "CHIANG MAI", lonlat: [98.99, 18.79] },
      { name: "MANDALAY", lonlat: [96.08, 21.97] },
      { name: "NAYPYIDAW", lonlat: [96.1, 19.74] },
      { name: "DA NANG", lonlat: [108.2, 16.05] },
      { name: "HAIPHONG", lonlat: [106.68, 20.86] },
      { name: "KUNMING", lonlat: [102.83, 25.04] },
      { name: "NANNING", lonlat: [108.37, 22.82] },
      { name: "GUANGZHOU", lonlat: [113.27, 23.13] },
      { name: "HONG KONG", lonlat: [114.17, 22.32] },
      { name: "MACAU", lonlat: [113.55, 22.2] },
      { name: "HAIKOU", lonlat: [110.33, 20.04] },
      { name: "PHUKET", lonlat: [98.4, 7.88] },
      { name: "PENANG", lonlat: [100.33, 5.42] },
      { name: "DHAKA", lonlat: [90.41, 23.81] },
      { name: "KOLKATA", lonlat: [88.36, 22.57] },
      { name: "MANILA", lonlat: [120.98, 14.6] },
    ],
  },

  // -------------------------------------------------------------------------
  // Hong Kong — Market tab background. East-Asia / South-China-Sea
  // window centred on HK. Scale chosen to keep mainland China south
  // coast, Taiwan, the Philippines, and parts of Vietnam visible
  // around the focal point. Hong Kong itself is geographically tiny;
  // Natural Earth carries it as ``Hong Kong S.A.R.`` (similar for
  // Macau as ``Macao S.A.R.``) — both are highlighted so the
  // viewer's eye anchors on the SAR cluster on the Pearl River
  // delta. Crosshair sits on HK proper.
  // -------------------------------------------------------------------------
  {
    id: "hong-kong",
    center: [114.17, 22.32],
    scale: 2500,
    bounds: { west: 80, east: 150, south: 5, north: 40 },
    highlight: {
      "Hong Kong S.A.R.": {
        fill: "rgba(0,229,255,0.32)",
        stroke: "rgba(0,229,255,1)",
        strokeWidth: 2.0,
      },
      "Macao S.A.R": {
        fill: "rgba(0,229,255,0.20)",
        stroke: "rgba(0,229,255,0.85)",
        strokeWidth: 1.5,
      },
      Taiwan: {
        fill: "rgba(245,158,11,0.07)",
        stroke: "rgba(245,158,11,0.7)",
        strokeWidth: 1.0,
      },
    },
    crosshairLatLon: [114.17, 22.32],
    cities: [
      { name: "HONG KONG", lonlat: [114.17, 22.32], emphasis: true },
      { name: "SHENZHEN", lonlat: [114.06, 22.55], emphasis: true },
      { name: "MACAU", lonlat: [113.55, 22.2] },
      { name: "GUANGZHOU", lonlat: [113.27, 23.13] },
      { name: "TAIPEI", lonlat: [121.57, 25.04] },
      { name: "MANILA", lonlat: [120.98, 14.6] },
      { name: "HANOI", lonlat: [105.85, 21.03] },
      { name: "SHANGHAI", lonlat: [121.47, 31.23] },
      { name: "WUHAN", lonlat: [114.31, 30.59] },
      { name: "CHONGQING", lonlat: [106.55, 29.56] },
      { name: "CHENGDU", lonlat: [104.07, 30.67] },
      { name: "NANJING", lonlat: [118.78, 32.06] },
      { name: "HANGZHOU", lonlat: [120.16, 30.27] },
      { name: "XIAMEN", lonlat: [118.1, 24.46] },
      { name: "FUZHOU", lonlat: [119.3, 26.07] },
      { name: "KAOHSIUNG", lonlat: [120.31, 22.63] },
      { name: "HAIKOU", lonlat: [110.33, 20.04] },
      { name: "SANYA", lonlat: [109.5, 18.25] },
      { name: "DA NANG", lonlat: [108.2, 16.05] },
      { name: "HO CHI MINH", lonlat: [106.66, 10.76] },
      { name: "VIENTIANE", lonlat: [102.6, 17.97] },
      { name: "CEBU", lonlat: [123.89, 10.32] },
      { name: "QUEZON CITY", lonlat: [121.04, 14.68] },
      { name: "NAHA", lonlat: [127.69, 26.21] },
      { name: "BANGKOK", lonlat: [100.5, 13.75] },
    ],
  },
];

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

const world = JSON.parse(readFileSync(TOPO_PATH, "utf8"));
const countries = topoFeature(world, world.objects.countries);

function buildRegion(region) {
  const projection = geoMercator()
    .center(region.center)
    .scale(region.scale)
    .translate([W / 2, H / 2]);
  const pathGen = geoPath(projection);

  const intersectsRegion = (feat) => {
    const [[w, s], [e, n]] = geoBounds(feat);
    return (
      e >= region.bounds.west &&
      w <= region.bounds.east &&
      n >= region.bounds.south &&
      s <= region.bounds.north
    );
  };

  const visible = countries.features.filter(intersectsRegion);
  const baseFeatures = [];
  const highlightFeatures = [];
  for (const f of visible) {
    if (region.highlight[f.properties.name]) highlightFeatures.push(f);
    else baseFeatures.push(f);
  }

  const pathTag = (feat, style) => {
    const d = pathGen(feat);
    if (!d) return "";
    return `  <path d="${d}" fill="${style.fill}" stroke="${style.stroke}" stroke-width="${style.strokeWidth}" stroke-linejoin="round" stroke-linecap="round" />`;
  };

  const countryPaths = [
    ...baseFeatures.map((f) => pathTag(f, DEFAULT_COUNTRY_STYLE)),
    ...highlightFeatures.map((f) =>
      pathTag(f, region.highlight[f.properties.name]),
    ),
  ].filter(Boolean);

  const cityMarks = region.cities
    .map((c) => {
      const px = projection(c.lonlat);
      if (!px) return "";
      const [x, y] = px;
      if (x < 0 || x > W || y < 0 || y > H) return "";
      const r = c.emphasis ? 4 : 3;
      const fontSize = c.emphasis ? 13 : 11;
      const opacity = c.emphasis ? 0.95 : 0.7;
      const labelColor = c.emphasis
        ? "rgba(0,229,255,0.95)"
        : "rgba(6,182,212,0.85)";
      return `  <g opacity="${opacity}">
    <circle cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="${r}" fill="rgba(245,158,11,0.9)" />
    <text x="${(x + 8).toFixed(2)}" y="${(y + 4).toFixed(2)}" font-size="${fontSize}" font-family="ui-monospace, monospace" letter-spacing="2" fill="${labelColor}">${c.name}</text>
  </g>`;
    })
    .filter(Boolean);

  const anchor = projection(region.crosshairLatLon);
  const crosshair = anchor
    ? `  <g transform="translate(${anchor[0].toFixed(2)} ${anchor[1].toFixed(2)})" stroke="rgba(0,229,255,0.85)" fill="none" stroke-width="1.2">
    <circle r="34" />
    <circle r="18" stroke="rgba(0,229,255,0.5)" />
    <circle r="4" fill="rgba(245,158,11,1)" stroke="none" />
    <line x1="-48" y1="0" x2="-24" y2="0" />
    <line x1="24" y1="0" x2="48" y2="0" />
    <line x1="0" y1="-48" x2="0" y2="-24" />
    <line x1="0" y1="24" x2="0" y2="48" />
  </g>`
    : "";

  const generatedAt = new Date().toISOString();

  return {
    path: join(PUBLIC_DIR, `${region.id}-wireframe.svg`),
    body: `<?xml version="1.0" encoding="UTF-8"?>
<!--
  Region map (${region.id}). Auto-generated by
  frontend/scripts/build-region-maps.mjs from Natural Earth
  countries-50m TopoJSON via d3-geo Mercator.

  Generated:    ${generatedAt}
  Projection:   geoMercator center=${JSON.stringify(region.center)} scale=${region.scale}
  ViewBox:      ${W}×${H}
  Countries:    ${visible.length} (filtered to ${region.bounds.west}–${region.bounds.east}E, ${region.bounds.south}–${region.bounds.north}N)

  DO NOT hand-edit — the next "npm run build:map" will overwrite.
-->
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}"
     preserveAspectRatio="xMidYMid meet" aria-hidden="true"
     opacity="0.98">
  <defs>
    <radialGradient id="gulfGlow" cx="50%" cy="50%" r="55%">
      <stop offset="0%" stop-color="rgba(6,182,212,0.10)" />
      <stop offset="100%" stop-color="rgba(6,182,212,0)" />
    </radialGradient>
    <pattern id="grid" width="55" height="50" patternUnits="userSpaceOnUse">
      <path d="M 55 0 L 0 0 0 50" fill="none"
            stroke="rgba(6,182,212,0.07)" stroke-width="1" />
    </pattern>
  </defs>

  <rect width="${W}" height="${H}" fill="url(#grid)" />
  <rect width="${W}" height="${H}" fill="url(#gulfGlow)" />

${countryPaths.join("\n")}

${cityMarks.join("\n")}

${crosshair}
</svg>
`,
    summary: `${region.id}: ${visible.length} countries, ${cityMarks.length} cities`,
  };
}

for (const region of REGIONS) {
  const out = buildRegion(region);
  writeFileSync(out.path, out.body, "utf8");
  console.log(
    `wrote ${out.path} (${(out.body.length / 1024).toFixed(1)} KB) — ${out.summary}`,
  );
}
