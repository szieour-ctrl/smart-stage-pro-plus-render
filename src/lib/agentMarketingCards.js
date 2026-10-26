// agentMarketingCards.js — Agent Marketing (Local Market Insider) opening and closing cards.
//
// Oct 26, 2026. Part of the "editorial compositing" split: GPT Image 2 makes a TEXT-FREE
// background plate; everything with words, numbers, names or compliance wording is drawn here
// by code as a transparent PNG layer (sharp + SVG, same engine as reelOverlay.js) and placed by
// ffmpeg. Nothing in this file is wired into a render route yet.
//
// Two cards, two canvases each:
//   opening  — market name + "Market Update", the hook headline, eyebrow with the report period,
//              agent lockup (headshot circle, name, brokerage) and the MetroList notice bar.
//   closing  — CTA headline, supporting line, tagline, agent lockup, MetroList notice bar.
//              variant "headshot" (default): circular headshot lockup.
//              variant "portrait": the background plate already contains the agent (Appearance
//              Sheet, later); no circle is drawn, the name lockup moves beside the portrait zone.
//   canvas "portrait" = 1080x1920 (9:16), "landscape" = 1920x1080 (16:9).
//
// The MetroList notice and the period are built HERE from the report's date window (never typed
// by the planner or the model). Caption band: captions go in CAPTION_BAND[canvas], above the
// notice bar; no card puts anything inside it.
const sharp = require("sharp");

const CANVAS = {
  portrait: {
    w: 1080, h: 1920, marginX: 105, maxW: 870,
    notice: { h: 150, size: 28, maxW: 960 },
    captionBand: { y: 1585, h: 180 },
    lockupCy: 1395, hs: 240,
    bottomLimit: 1235,
    title: 176, kind: 118, sub: 74, subMaxW: 700, eyebrow: 26, headline: 108, support: 66,
    top: 150,
    portraitZone: { x: 0, y: 960, w: 800, h: 780 },
  },
  landscape: {
    w: 1920, h: 1080, marginX: 130, maxW: 1000,
    notice: { h: 104, size: 26, maxW: 1700 },
    captionBand: { y: 860, h: 100 },
    lockupCy: 740, hs: 150,
    bottomLimit: 640,
    title: 150, kind: 100, sub: 52, subMaxW: 860, eyebrow: 22, headline: 84, support: 50,
    top: 96,
    portraitZone: { x: 1060, y: 120, w: 760, h: 820 },
  },
};
const CAPTION_BAND = { portrait: CANVAS.portrait.captionBand, landscape: CANVAS.landscape.captionBand };
const PORTRAIT_ZONE = { portrait: CANVAS.portrait.portraitZone, landscape: CANVAS.landscape.portraitZone };

const DISPLAY = "Playfair Display";
const BODY = "Montserrat";

const typo = (s) => String(s ?? "").replace(/'/g, "\u2019");
const esc = (s) => typo(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const hex = (v, fb) => (/^#[0-9a-fA-F]{6}$/.test(String(v || "")) ? v : fb);
const lum = (h) => { const n = parseInt(h.slice(1), 16); return (0.2126 * ((n >> 16) & 255) + 0.7152 * ((n >> 8) & 255) + 0.0722 * (n & 255)) / 255; };

// ---------- MetroList notice + period (code-built; never typed) ----------
const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
const parseDay = (s) => { const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(String(s || "")); return m ? { y: +m[1], m: +m[2] - 1, d: +m[3] } : null; };
const longDate = (p) => `${MONTHS[p.m]} ${p.d}, ${p.y}`;
const lastDayOf = (y, m) => new Date(Date.UTC(y, m + 1, 0)).getUTCDate();

// "Based on information from MetroList Services, Inc. for the period September 1, 2026 through September 30, 2026."
function metroListNotice(period) {
  const a = parseDay(period && period.start), b = parseDay(period && period.end);
  if (!a || !b) return null;
  return `Based on information from MetroList Services, Inc. for the period ${longDate(a)} through ${longDate(b)}.`;
}
// "September 2026" for a whole calendar month; otherwise "Aug 2 – Oct 26, 2026".
function periodLabel(period) {
  const a = parseDay(period && period.start), b = parseDay(period && period.end);
  if (!a || !b) return "";
  if (a.y === b.y && a.m === b.m && a.d === 1 && b.d === lastDayOf(b.y, b.m)) return `${MONTHS[a.m]} ${a.y}`;
  const s = (p) => `${MONTHS[p.m].slice(0, 3)} ${p.d}`;
  return a.y === b.y ? `${s(a)} – ${s(b)}, ${b.y}` : `${s(a)}, ${a.y} – ${s(b)}, ${b.y}`;
}

// The notice is required whenever the piece uses MetroList-derived figures. A missing or invalid period
// must stop the render, never ship a card without it. Web-only pieces pass noNotice: true on purpose.
function needNotice(p) {
  if (p.noNotice === true) return null;
  const n = p.noticeText || metroListNotice(p.period);
  if (!n) throw new Error("MetroList notice required: report period is missing or invalid (pass period {start,end} as YYYY-MM-DD, or noNotice:true for a web-only piece)");
  return n;
}

// ---------- text measuring (pango, same engine librsvg uses) ----------
const measureCache = new Map();
async function measure(text, { family, weight, size, spacing = 0 }) {
  const key = [text, family, weight, size, spacing].join("|");
  if (measureCache.has(key)) return measureCache.get(key);
  const markup = `<span font_family="${esc(family)}" font_weight="${weight}" font_features="lnum" letter_spacing="${Math.round(spacing * 1024)}" size="${Math.round(size * 1024)}">${esc(text)}</span>`;
  const { info } = await sharp({ text: { text: markup, rgba: true, dpi: 72 } }).png().toBuffer({ resolveWithObject: true });
  measureCache.set(key, info.width);
  return info.width;
}
// Greedy wrap into lines no wider than maxW, then narrowed so the lines are balanced
// (same line count, no dangling one-word last line).
async function greedy(words, maxW, font) {
  const lines = []; let cur = "";
  for (const w of words) {
    const t = cur ? `${cur} ${w}` : w;
    if (!cur || (await measure(t, font)) <= maxW) cur = t; else { lines.push(cur); cur = w; }
  }
  if (cur) lines.push(cur);
  return lines;
}
async function wrap(text, maxW, font) {
  const words = String(text || "").trim().split(/\s+/).filter(Boolean);
  let best = await greedy(words, maxW, font);
  if (best.length < 2) return best;
  for (let w = maxW - 20; w > maxW * 0.5; w -= 20) {
    const l = await greedy(words, w, font);
    if (l.length !== best.length) break;
    best = l;
  }
  return best;
}
// Largest size <= start that fits one line in maxW (never below min).
async function fitSize(text, maxW, start, min, font) {
  let size = start;
  while (size > min && (await measure(text, { ...font, size })) > maxW) size -= 2;
  return size;
}

async function usableImage(buf) {
  if (!buf) return null;
  try { await sharp(buf).metadata(); return buf; } catch { return null; }
}
async function circleHeadshot(buf, d) {
  return sharp(buf).rotate().resize(d, d, { fit: "cover", position: "attention" }).png().toBuffer();
}

function palette(brand = {}) {
  const gold = hex(brand.primary, "#B08D3F");
  const sec = hex(brand.secondary, "#1D2023");
  const ink = lum(sec) < 0.35 ? sec : "#1D2023";
  return { gold, ink, ring: hex(brand.border, gold) };
}

const T = (x, y, text, { size, weight = 400, family = DISPLAY, fill, anchor = "start", spacing = 0, opacity = 1 }) =>
  `<text x="${x}" y="${Math.round(y)}" text-anchor="${anchor}" font-family="${family}" font-weight="${weight}" font-size="${size}"${spacing ? ` letter-spacing="${spacing}"` : ""} fill="${fill}"${opacity < 1 ? ` fill-opacity="${opacity}"` : ""}>${esc(text)}</text>`;
const dataUri = (b) => `data:image/png;base64,${b.toString("base64")}`;

// Shared pieces ---------------------------------------------------------------
async function noticeBar(cv, pal, noticeText) {
  const n = cv.notice, top = cv.h - n.h;
  if (!noticeText) return { svg: "", top: cv.h };
  const font = { family: BODY, weight: 500, size: n.size };
  const lines = await wrap(noticeText, n.maxW, font);
  const lh = Math.round(n.size * 1.38), block = lines.length * lh;
  const y0 = top + Math.round((n.h - block) / 2) + Math.round(n.size * 1.02);
  const svg = `<rect x="0" y="${top}" width="${cv.w}" height="${n.h}" fill="${pal.ink}"/>
  <rect x="0" y="${top}" width="${cv.w}" height="3" fill="${pal.gold}"/>
  ${lines.map((l, i) => T(cv.w / 2, y0 + i * lh, l, { size: n.size, weight: 500, family: BODY, fill: "#FFFFFF", anchor: "middle", opacity: 0.95 })).join("\n  ")}`;
  return { svg, top };
}

async function lockup(cv, pal, agent, hsBuf, { cx, cy, align = "left", circle = true }) {
  const name = agent.name || "";
  const brokerage = String(agent.brokerage || "").toUpperCase();
  const r = cv.hs / 2;
  const textMaxW = align === "right" ? 560 : cv.maxW - (circle ? cv.hs + 40 : 0);
  const nameSize = await fitSize(name, textMaxW, cv === CANVAS.portrait ? 64 : 52, 34, { family: DISPLAY, weight: 600 });
  const brkSize = cv === CANVAS.portrait ? 24 : 20, brkSpacing = 5;
  const brkFont = { family: BODY, weight: 500, size: brkSize, spacing: brkSpacing };
  const brkLines = brokerage ? (await wrap(brokerage, textMaxW, brkFont)).slice(0, 2) : [];
  const nameW = await measure(name, { family: DISPLAY, weight: 600, size: nameSize });
  let brkW = 0;
  for (const l of brkLines) brkW = Math.max(brkW, await measure(l, brkFont));
  const textW = Math.max(nameW, brkW);
  let out = "", x;
  if (circle && hsBuf) {
    const hx = cx, hy = cy - r;
    out += `<clipPath id="hs"><circle cx="${hx + r}" cy="${cy}" r="${r}"/></clipPath>
  <image href="${dataUri(hsBuf)}" x="${hx}" y="${hy}" width="${cv.hs}" height="${cv.hs}" clip-path="url(#hs)" preserveAspectRatio="xMidYMid slice"/>
  <circle cx="${hx + r}" cy="${cy}" r="${r}" fill="none" stroke="${pal.gold}" stroke-width="5"/>`;
    x = hx + cv.hs + 40;
  } else x = align === "right" ? cx - textW - 30 : cx;
  const brkLH = brkSize + 14;
  const barH = nameSize + (brkLines.length ? 22 + brkLines.length * brkLH : 0) + 8, barTop = cy - barH / 2;
  out += `<rect x="${x - 26}" y="${Math.round(barTop)}" width="4" height="${Math.round(barH)}" fill="${pal.gold}"/>`;
  out += T(x, barTop + nameSize * 0.86, name, { size: nameSize, weight: 600, fill: "#FFFFFF" });
  brkLines.forEach((l, i) => { out += T(x, barTop + nameSize + brkSize + 22 + i * brkLH, l, { size: brkSize, weight: 500, family: BODY, fill: "#FFFFFF", spacing: brkSpacing, opacity: 0.95 }); });
  return out;
}

const scrims = (cv, lockTop, which) => `<defs>
    <linearGradient id="lt" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#fff" stop-opacity="0.5"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></linearGradient>
    <linearGradient id="lb" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#000" stop-opacity="0"/><stop offset="1" stop-color="#000" stop-opacity="0.6"/></linearGradient>
    <linearGradient id="ll" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#fff" stop-opacity="0.5"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></linearGradient>
  </defs>
  ${cv === CANVAS.portrait || which === "top" ? `<rect x="0" y="0" width="${cv.w}" height="${Math.round(cv.h * 0.58)}" fill="url(#lt)"/>` : `<rect x="0" y="0" width="${Math.round(cv.w * 0.62)}" height="${cv.h}" fill="url(#ll)"/>`}
  <rect x="0" y="${lockTop - 160}" width="${cv.w}" height="${cv.h - lockTop + 160}" fill="url(#lb)"/>`;

// Opening ---------------------------------------------------------------------
async function openingSvg(p, cv) {
  const pal = palette(p.brand);
  const agent = p.agent || {};
  const hs = (await usableImage(p.headshotBuf)) ? await circleHeadshot(p.headshotBuf, cv.hs) : null;
  const notice = needNotice(p);
  const nb = await noticeBar(cv, pal, notice);
  const X = cv.marginX;
  const market = p.market || "", kind = p.kind || "Market Update", sub = p.headline || "";
  const eyebrow = [p.eyebrow, periodLabel(p.period)].filter(Boolean).join("  ·  ").toUpperCase();

  // Fit the whole text stack above cv.bottomLimit by shrinking a scale factor.
  let scale = 1, layout;
  for (; scale >= 0.62; scale -= 0.04) {
    // One line if it fits at >= 62% of the full size, otherwise up to two balanced lines at a smaller size.
    let ts = Math.round(cv.title * scale), marketLines = [market];
    if ((await measure(market, { family: DISPLAY, weight: 800, size: ts })) > cv.maxW) {
      ts = await fitSize(market, cv.maxW, ts, Math.round(cv.title * 0.62), { family: DISPLAY, weight: 800 });
      if ((await measure(market, { family: DISPLAY, weight: 800, size: ts })) > cv.maxW) {
        ts = Math.round(cv.title * 0.7 * scale);
        marketLines = await wrap(market, cv.maxW, { family: DISPLAY, weight: 800, size: ts });
        while (marketLines.length > 2 && ts > 56) { ts -= 4; marketLines = await wrap(market, cv.maxW, { family: DISPLAY, weight: 800, size: ts }); }
      }
    }
    const ks = await fitSize(kind, cv.maxW, Math.min(Math.round(cv.kind * scale), Math.round(ts * 0.68)), 40, { family: DISPLAY, weight: 600 });
    const ss = Math.round(cv.sub * scale);
    const subLines = await wrap(sub, cv.subMaxW, { family: DISPLAY, weight: 500, size: ss });
    const es = Math.round(cv.eyebrow * scale);
    const eyeLines = eyebrow ? await wrap(eyebrow, cv.maxW, { family: BODY, weight: 500, size: es, spacing: 5 }) : [];
    let y = cv.top + 4;
    const rule1 = y; y += 40;
    const titleBase = y + ts * 0.8; y = titleBase + (marketLines.length - 1) * ts * 1.04 + 10;
    const kindBase = y + ks * 0.82; y = kindBase + 34;
    const rule2 = y; y += 52;
    const subBase = y + ss * 0.8; y += subLines.length * ss * 1.14 + 18;
    const eyeBase = y + es; y = eyeBase + (Math.max(1, eyeLines.length) - 1) * es * 1.5 + 30;
    const rule3 = y; y += 6;
    layout = { eyeLines, rule1, ts, marketLines, titleBase, ks, kindBase, rule2, ss, subLines, subBase, es, eyeBase, rule3, bottom: y };
    if (y <= cv.bottomLimit) break;
  }
  const L = layout;
  const lockTop = nb.top - (cv === CANVAS.portrait ? 0 : 0);
  const lockCy = cv.lockupCy;
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${cv.w}" height="${cv.h}">
  ${scrims(cv, lockCy - cv.hs / 2 - 20, "both")}
  <rect x="${X}" y="${L.rule1}" width="145" height="4" fill="${pal.gold}"/>
  ${L.marketLines.map((l, i) => T(X - 4, L.titleBase + i * L.ts * 1.04, l, { size: L.ts, weight: 800, fill: pal.ink })).join("\n  ")}
  ${T(X - 2, L.kindBase, kind, { size: L.ks, weight: 600, fill: pal.gold })}
  <rect x="${X}" y="${L.rule2}" width="110" height="3" fill="${pal.ink}"/>
  ${L.subLines.map((l, i) => T(X, L.subBase + i * L.ss * 1.14, l, { size: L.ss, weight: 500, fill: pal.ink })).join("\n  ")}
  ${L.eyeLines.map((l, i) => T(X, L.eyeBase + i * L.es * 1.5, l, { size: L.es, weight: 500, family: BODY, fill: pal.ink, spacing: 5 })).join("\n  ")}
  <rect x="${X}" y="${L.rule3}" width="90" height="3" fill="${pal.gold}"/>
  ${agent.name ? await lockup(cv, pal, agent, hs, { cx: X, cy: lockCy, circle: true }) : ""}
  ${nb.svg}
</svg>`;
  return { svg, warnings: notice ? [] : ["no MetroList notice (noNotice was set)"], textBottom: L.bottom, limit: cv.bottomLimit, fits: L.bottom <= cv.bottomLimit };
}

// Closing ---------------------------------------------------------------------
async function closingSvg(p, cv) {
  const pal = palette(p.brand);
  const agent = p.agent || {};
  const variant = p.variant === "portrait" ? "portrait" : "headshot";
  const hs = variant === "headshot" && (await usableImage(p.headshotBuf)) ? await circleHeadshot(p.headshotBuf, cv.hs) : null;
  const notice = needNotice(p);
  const nb = await noticeBar(cv, pal, notice);
  const X = cv.marginX;
  const dark = p.headline && p.headline[0] ? p.headline[0] : "", accent = p.headline && p.headline[1] ? p.headline[1] : "";
  const support = p.support || "", tagline = String(p.tagline || "").toUpperCase();

  let scale = 1, layout;
  for (; scale >= 0.62; scale -= 0.04) {
    const hsz = Math.round(cv.headline * scale);
    const dl = await wrap(dark, cv.maxW, { family: DISPLAY, weight: 800, size: hsz });
    const al = await wrap(accent, cv.maxW, { family: DISPLAY, weight: 800, size: hsz });
    const ss = Math.round(cv.support * scale), es = Math.round(cv.eyebrow * scale);
    const sl = await wrap(support, cv.subMaxW, { family: DISPLAY, weight: 500, size: ss });
    const tagLines = tagline ? await wrap(tagline, cv.maxW, { family: BODY, weight: 500, size: es, spacing: 5 }) : [];
    let y = cv.top + 4;
    const rule1 = y; y += 34;
    const hBase = y + hsz * 0.82; y = hBase + (dl.length + al.length - 1) * hsz * 1.14 + 30;
    const rule2 = y; y += 48;
    const sBase = y + ss * 0.8; y += sl.length * ss * 1.14 + 18;
    const eBase = tagline ? y + es : 0; if (tagline) y = eBase + (tagLines.length - 1) * es * 1.5 + 30;
    const rule3 = y; y += 6;
    layout = { tagLines, rule1, hsz, dl, al, hBase, rule2, ss, sl, sBase, es, eBase, rule3, bottom: y };
    if (y <= cv.bottomLimit) break;
  }
  const L = layout;
  const headSvg = [...L.dl.map((l) => ({ l, c: pal.ink })), ...L.al.map((l) => ({ l, c: pal.gold }))]
    .map((r, i) => T(X - 4, L.hBase + i * L.hsz * 1.14, r.l, { size: L.hsz, weight: 800, fill: r.c })).join("\n  ");
  // Lockup: headshot variant sits bottom-left like the opening; portrait variant leaves the portrait zone clear
  // and puts the name block bottom-right.
  let lock = "";
  if (agent.name) {
    // 9:16: portrait zone is bottom-left, so the name block goes bottom-right. 16:9: portrait zone is the
    // right half, so the name block stays bottom-left (no circle).
    if (variant !== "portrait") lock = await lockup(cv, pal, agent, hs, { cx: X, cy: cv.lockupCy, circle: true });
    else if (cv === CANVAS.portrait) lock = await lockup(cv, pal, agent, null, { cx: cv.w - cv.marginX, cy: 1480, align: "right", circle: false });
    else lock = await lockup(cv, pal, agent, null, { cx: X + 26, cy: cv.lockupCy, circle: false });
  }
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${cv.w}" height="${cv.h}">
  ${scrims(cv, cv.lockupCy - cv.hs / 2 - 20, "both")}
  <rect x="${X}" y="${L.rule1}" width="145" height="4" fill="${pal.gold}"/>
  ${headSvg}
  <rect x="${X}" y="${L.rule2}" width="110" height="3" fill="${pal.gold}"/>
  ${L.sl.map((l, i) => T(X, L.sBase + i * L.ss * 1.14, l, { size: L.ss, weight: 500, fill: pal.ink })).join("\n  ")}
  ${L.tagLines.map((l, i) => T(X, L.eBase + i * L.es * 1.5, l, { size: L.es, weight: 500, family: BODY, fill: pal.ink, spacing: 5 })).join("\n  ")}
  <rect x="${X}" y="${L.rule3}" width="90" height="3" fill="${pal.gold}"/>
  ${lock}
  ${nb.svg}
</svg>`;
  return { svg, warnings: notice ? [] : ["no MetroList notice (noNotice was set)"], textBottom: L.bottom, limit: cv.bottomLimit, fits: L.bottom <= cv.bottomLimit };
}

// Playfair's default figures are old-style (a "4" drops below the line). librsvg can't switch the font
// feature on from SVG, so big figures are drawn through pango markup (font_features="lnum") into a
// trimmed PNG and embedded. Returns the ink-only image plus its size; its bottom edge is the baseline.
async function liningFigure(text, size, color, weight = 800) {
  const parts = Array.isArray(text) ? text : [{ t: text }];
  const spans = parts.map((q) => `<span font_family="${DISPLAY}" font_weight="${weight}" font_features="lnum" foreground="${color}" size="${Math.round(size * (q.s || 1) * 1024)}">${esc(q.t)}</span>`).join("");
  const { data, info } = await sharp({ text: { text: spans, rgba: true, dpi: 72 } }).trim().png().toBuffer({ resolveWithObject: true });
  return { buf: data, w: info.width, h: info.height };
}

// Big-stat card ---------------------------------------------------------------
// Middle scene: lead-in line, one huge figure, trailing line, optional icon, and a short code-built
// source line ("METROLIST · SEPTEMBER 2026") at the bottom. Subtle branding: palette and type only
// (no headshot, no notice bar; the full notice lives on the opening and closing cards).
// params: { lead, value, unit?, trail?, icon?: "calendar"|"none", sourceLabel?, period, brand }
// value MUST come from the verified data, never typed by the planner; lead/trail are script text.
function calendarIcon(cx, cy, r, pal) {
  const a = r * 0.46; // half-size of the calendar glyph
  return `<circle cx="${cx}" cy="${cy}" r="${r * 0.78}" fill="#F0E9D8"/>
  <path d="M ${cx - r * 0.1} ${cy - r * 1.08} A ${r * 1.1} ${r * 1.1} 0 1 1 ${cx + r * 0.95} ${cy + r * 0.55}" fill="none" stroke="${pal.gold}" stroke-width="${Math.max(6, r * 0.1)}" stroke-linecap="round"/>
  <g fill="none" stroke="${pal.gold}" stroke-width="${Math.max(4, r * 0.07)}" stroke-linejoin="round" stroke-linecap="round">
    <rect x="${cx - a}" y="${cy - a * 0.8}" width="${a * 2}" height="${a * 1.7}" rx="${a * 0.18}"/>
    <path d="M ${cx - a} ${cy - a * 0.2} H ${cx + a} M ${cx - a * 0.5} ${cy - a * 1.15} V ${cy - a * 0.55} M ${cx + a * 0.5} ${cy - a * 1.15} V ${cy - a * 0.55}"/>
  </g>
  <g fill="${pal.gold}">${[-0.55, 0, 0.55].flatMap((dx) => [0.25, 0.7].map((dy) => `<circle cx="${cx + a * dx}" cy="${cy + a * dy}" r="${Math.max(3, r * 0.05)}"/>`)).join("")}</g>`;
}

async function statSvg(p, cv) {
  const pal = palette(p.brand);
  const portrait = cv === CANVAS.portrait;
  const X = cv.marginX;
  const text = [p.value, p.unit].filter(Boolean).join(" ").toUpperCase();
  const label = (p.sourceLabel || "MetroList") + (periodLabel(p.period) ? `  ·  ${periodLabel(p.period)}` : "");
  const colW = portrait ? cv.maxW : 1000;
  const iconR = p.icon === "none" ? 0 : (portrait ? 70 : 64);
  let scale = 1, L;
  for (; scale >= 0.62; scale -= 0.04) {
    const ls = Math.round(cv.sub * scale);
    const leadLines = await wrap(p.lead || "", colW, { family: DISPLAY, weight: 500, size: ls });
    const vs = await fitSize(text, colW, Math.round((portrait ? 270 : 250) * scale), 90, { family: DISPLAY, weight: 800 });
    const trailMax = iconR ? colW - iconR * 2 - 50 : colW;
    const trailLines = await wrap(p.trail || "", trailMax, { family: DISPLAY, weight: 500, size: ls });
    let y = cv.top + 4;
    const rule1 = y; y += 44;
    const leadBase = y + ls * 0.8; y += leadLines.length * ls * 1.14 + 10;
    const valBase = y + vs * 0.82; y = valBase + 40;
    const trailBase = y + ls * 0.8; y += trailLines.length * ls * 1.14 + 14;
    const rule2 = y; y += 6;
    L = { ls, leadLines, vs, trailLines, rule1, leadBase, valBase, trailBase, rule2, bottom: y, trailTop: trailBase - ls * 0.8 };
    if (y <= cv.bottomLimit + (portrait ? 250 : 120)) break;
  }
  const iconCx = portrait ? cv.w - cv.marginX - iconR - 10 : X + colW + iconR + 70;
  const iconCy = portrait ? L.trailTop + (L.trailLines.length * L.ls * 1.14) / 2 : L.valBase - L.vs * 0.36;
  const footY = portrait ? cv.h - 64 : cv.h - 46;
  const fig = await liningFigure(text, L.vs, pal.gold);
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${cv.w}" height="${cv.h}">
  <defs>
    <linearGradient id="lt" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#fff" stop-opacity="0.5"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></linearGradient>
    <linearGradient id="ll" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#fff" stop-opacity="0.5"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></linearGradient>
    <linearGradient id="lb" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#000" stop-opacity="0"/><stop offset="1" stop-color="#000" stop-opacity="0.45"/></linearGradient>
  </defs>
  ${portrait ? `<rect x="0" y="0" width="${cv.w}" height="${Math.round(cv.h * 0.6)}" fill="url(#lt)"/>` : `<rect x="0" y="0" width="${Math.round(cv.w * 0.66)}" height="${cv.h}" fill="url(#ll)"/>`}
  <rect x="0" y="${cv.h - 260}" width="${cv.w}" height="260" fill="url(#lb)"/>
  <rect x="${X}" y="${L.rule1}" width="145" height="4" fill="${pal.gold}"/>
  ${L.leadLines.map((l, i) => T(X, L.leadBase + i * L.ls * 1.14, l, { size: L.ls, weight: 500, fill: pal.ink })).join("\n  ")}
  <image href="${dataUri(fig.buf)}" x="${X - 2}" y="${Math.round(L.valBase - fig.h)}" width="${fig.w}" height="${fig.h}"/>
  ${L.trailLines.map((l, i) => T(X, L.trailBase + i * L.ls * 1.14, l, { size: L.ls, weight: 500, fill: pal.ink })).join("\n  ")}
  ${iconR ? calendarIcon(iconCx, iconCy, iconR, pal) : ""}
  <rect x="${X}" y="${L.rule2}" width="145" height="3" fill="${pal.gold}"/>
  ${T(X, footY, label.toUpperCase(), { size: portrait ? 24 : 20, weight: 500, family: BODY, fill: "#FFFFFF", spacing: 5, opacity: 0.95 })}
</svg>`;
  return { svg, warnings: [], textBottom: L.bottom, limit: cv.bottomLimit + (portrait ? 250 : 120), fits: L.bottom <= cv.bottomLimit + (portrait ? 250 : 120) };
}
async function renderStatCard(params, outPng, canvas = "portrait") {
  const cv = CANVAS[canvas] || CANVAS.portrait;
  if (!params.value) throw new Error("renderStatCard: value is required (it must come from the verified data)");
  const r = await statSvg(params, cv);
  await sharp(Buffer.from(r.svg)).png().toFile(outPng);
  return { png: outPng, warnings: r.warnings, fits: r.fits, textBottom: r.textBottom, limit: r.limit };
}

// Comparison card -------------------------------------------------------------
// Two cards side by side: LEFT = a percentage (bar drawn from the number), RIGHT = "N of M" (M dots drawn,
// N highlighted). The pictures and the figures come from the same numbers, so they cannot disagree.
// params: { market, kind?, left:{ lead, percent, caption, icon? "sold" }, right:{ lead, highlighted, total, caption, icon? "reduced" },
//           period, brand }
function houseIcon(cx, cy, s, color, pal, badge) {
  const sw = Math.max(4, s * 0.075);
  let out = `<g fill="none" stroke="${color}" stroke-width="${sw}" stroke-linejoin="round" stroke-linecap="round">
    <path d="M ${cx - s * 0.95} ${cy + s * 0.05} L ${cx} ${cy - s * 0.85} L ${cx + s * 0.95} ${cy + s * 0.05}"/>
    <path d="M ${cx - s * 0.7} ${cy - s * 0.12} V ${cy + s * 0.85} H ${cx + s * 0.7} V ${cy - s * 0.12}"/></g>`;
  if (badge === "sold") {
    out += `<rect x="${cx - s * 0.62}" y="${cy + s * 0.5}" width="${s * 1.24}" height="${s * 0.5}" rx="${s * 0.08}" fill="${pal.gold}"/>
    <text x="${cx}" y="${cy + s * 0.5 + s * 0.35}" text-anchor="middle" font-family="${BODY}" font-weight="700" font-size="${s * 0.3}" letter-spacing="${s * 0.03}" fill="#FFFFFF">SOLD</text>`;
  } else if (badge === "reduced") {
    const bx = cx + s * 0.78, by = cy + s * 0.7, r = s * 0.42;
    out += `<circle cx="${bx}" cy="${by}" r="${r}" fill="${pal.gold}"/>
    <path d="M ${bx} ${by - r * 0.5} V ${by + r * 0.45} M ${bx - r * 0.42} ${by + r * 0.05} L ${bx} ${by + r * 0.5} L ${bx + r * 0.42} ${by + r * 0.05}" fill="none" stroke="#FFFFFF" stroke-width="${Math.max(3, r * 0.2)}" stroke-linecap="round" stroke-linejoin="round"/>`;
  }
  return out;
}

async function compareSvg(p, cv) {
  const pal = palette(p.brand);
  const portrait = cv === CANVAS.portrait;
  const X = cv.marginX;
  const l = p.left || {}, r = p.right || {};
  const label = (p.sourceLabel || "MetroList") + (periodLabel(p.period) ? `  ·  ${periodLabel(p.period)}` : "");
  // figures derived from the numbers
  const pct = Number(l.percent);
  if (!(pct >= 0 && pct <= 200)) throw new Error("renderCompareCard: left.percent must be a number");
  const total = Number(r.total), hi = Number(r.highlighted);
  if (!Number.isInteger(total) || !Number.isInteger(hi) || total < 1 || total > 42 || hi < 0 || hi > total) throw new Error("renderCompareCard: right.highlighted/total must be whole numbers with highlighted <= total <= 42");
  const leftParts = [{ t: `${Number.isInteger(pct) ? pct : pct.toFixed(1)}%` }];
  const rightParts = [{ t: String(hi) }, { t: " of ", s: 0.55 }, { t: String(total) }];

  // geometry
  const gap = portrait ? 50 : 60, cardsX = portrait ? 60 : X;
  const cardW = Math.floor((cv.w - 2 * cardsX - gap) / 2);
  const cardTop = portrait ? 840 : 330, cardH = portrait ? 740 : 520;
  const pad = portrait ? 34 : 40, innerW = cardW - 2 * pad;
  const titleMax = portrait ? cv.maxW : 1000;
  const ts = await fitSize(p.market || "", titleMax, portrait ? 130 : 96, 60, { family: DISPLAY, weight: 800 });
  const ks = await fitSize(p.kind || "Market Update", titleMax, Math.round(ts * 0.68), 40, { family: DISPLAY, weight: 600 });
  const cards = [{ d: l, x: cardsX, light: true }, { d: r, x: cardsX + cardW + gap, light: false }];
  let svg = "";
  for (const c of cards) {
    const ink = c.light ? pal.ink : "#FFFFFF", fill = c.light ? "#FBF8F1" : pal.ink;
    const leadS = portrait ? 42 : 40, capS = portrait ? 22 : 20;
    const leadLines = await wrap(c.d.lead || "", innerW, { family: DISPLAY, weight: 600, size: leadS });
    const capLines = await wrap(String(c.d.caption || "").toUpperCase(), innerW, { family: BODY, weight: 600, size: capS, spacing: 4 });
    // Never drop script words silently: too-long text is an error so the planner can shorten and retry.
    if (leadLines.length > 3 || capLines.length > 4) throw new Error(`renderCompareCard: text too long for the card (lead ${leadLines.length} lines, caption ${capLines.length} lines; max 3 and 4)`);
    const vs = await fitSize(c.light ? leftParts[0].t : `${hi} of ${total}`, innerW, portrait ? 150 : 128, 70, { family: DISPLAY, weight: 800 });
    const fig = await liningFigure(c.light ? leftParts : rightParts, vs, pal.gold);
    const figW = Math.min(fig.w, innerW);
    const cx = c.x + cardW / 2;
    let y = cardTop;
    svg += `<rect x="${c.x}" y="${cardTop}" width="${cardW}" height="${cardH}" rx="26" fill="${fill}" fill-opacity="0.96" stroke="${pal.gold}" stroke-width="4"/>`;
    const iconS = portrait ? 52 : 40;
    svg += houseIcon(cx, cardTop + (portrait ? 112 : 84), iconS, ink, pal, c.d.icon || (c.light ? "sold" : "reduced"));
    y = cardTop + (portrait ? 215 : 150);
    svg += `<rect x="${cx - 60}" y="${y}" width="120" height="3" fill="${pal.gold}"/>`;
    y += portrait ? 34 : 24;
    leadLines.forEach((t, i) => { svg += T(cx, y + leadS * 0.85 + i * leadS * 1.15, t, { size: leadS, weight: 600, fill: ink, anchor: "middle" }); });
    y += leadLines.length * leadS * 1.15 + (portrait ? 22 : 18);
    svg += `<image href="${dataUri(fig.buf)}" x="${Math.round(cx - fig.w / 2)}" y="${Math.round(y)}" width="${fig.w}" height="${fig.h}"/>`;
    y += fig.h + (portrait ? 34 : 26);
    if (c.light) {
      // percentage bar: the filled share is the number itself (capped at the track)
      const bw = innerW - 20, bh = portrait ? 18 : 16, bx = cx - bw / 2;
      svg += `<rect x="${bx}" y="${y}" width="${bw}" height="${bh}" rx="${bh / 2}" fill="${pal.ink}" fill-opacity="0.14"/>
  <rect x="${bx}" y="${y}" width="${Math.max(bh, Math.round(bw * Math.min(pct, 100) / 100))}" height="${bh}" rx="${bh / 2}" fill="${pal.gold}"/>`;
      y += bh + (portrait ? 30 : 24);
    } else {
      // dot grid: total dots, first `hi` highlighted; rows fill left to right
      const cols = total <= 14 ? 7 : Math.min(10, Math.ceil(total / 3)), rows = Math.ceil(total / cols);
      const dr = portrait ? 12 : 11, step = Math.min(portrait ? 40 : 38, Math.floor((innerW - 10) / cols));
      const gw = (cols - 1) * step, gx = cx - gw / 2;
      for (let i = 0; i < total; i++) {
        const dx = gx + (i % cols) * step, dy = y + dr + Math.floor(i / cols) * step;
        svg += i < hi
          ? `<circle cx="${dx}" cy="${dy}" r="${dr}" fill="${pal.gold}"/>`
          : `<circle cx="${dx}" cy="${dy}" r="${dr - 1.5}" fill="#FFFFFF" fill-opacity="0.12" stroke="#FFFFFF" stroke-opacity="0.55" stroke-width="2"/>`;
      }
      y += rows * step + (portrait ? 14 : 10);
    }
    capLines.forEach((t, i) => { svg += T(cx, y + capS + i * capS * 1.5, t, { size: capS, weight: 600, family: BODY, fill: ink, anchor: "middle", spacing: 4, opacity: 0.95 }); });
    c.bottom = y + capS + (capLines.length - 1) * capS * 1.5;
    c.fits = c.bottom <= cardTop + cardH - 18;
  }
  const footY = portrait ? cv.h - 64 : cv.h - 46;
  const body = `<svg xmlns="http://www.w3.org/2000/svg" width="${cv.w}" height="${cv.h}">
  <defs>
    <linearGradient id="lt" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#fff" stop-opacity="0.5"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></linearGradient>
    <linearGradient id="lb" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#000" stop-opacity="0"/><stop offset="1" stop-color="#000" stop-opacity="0.45"/></linearGradient>
  </defs>
  <rect x="0" y="0" width="${cv.w}" height="${Math.round(cv.h * 0.5)}" fill="url(#lt)"/>
  <rect x="0" y="${cv.h - 260}" width="${cv.w}" height="260" fill="url(#lb)"/>
  <rect x="${X}" y="${cv.top + 4}" width="145" height="4" fill="${pal.gold}"/>
  ${T(X - 4, cv.top + 4 + 40 + ts * 0.8, p.market || "", { size: ts, weight: 800, fill: pal.ink })}
  ${T(X - 2, cv.top + 4 + 40 + ts * 0.8 + 10 + ks * 0.82, p.kind || "Market Update", { size: ks, weight: 600, fill: pal.gold })}
  ${svg}
  ${T(cv.w / 2, footY, label.toUpperCase(), { size: portrait ? 24 : 20, weight: 500, family: BODY, fill: "#FFFFFF", spacing: 5, anchor: "middle", opacity: 0.95 })}
</svg>`;
  return { svg: body, warnings: [], fits: cards.every((c) => c.fits), textBottom: Math.max(...cards.map((c) => c.bottom)), limit: cardTop + cardH - 18 };
}
async function renderCompareCard(params, outPng, canvas = "portrait") {
  const cv = CANVAS[canvas] || CANVAS.portrait;
  const r = await compareSvg(params, cv);
  await sharp(Buffer.from(r.svg)).png().toFile(outPng);
  return { png: outPng, warnings: r.warnings, fits: r.fits, textBottom: r.textBottom, limit: r.limit };
}

// Expert-takeaway card --------------------------------------------------------
// Editorial statement scene: two-tone headline, a statement, a supporting line, and 1-3 icon points
// ("Condition" / "Pricing" ...). All text is script or planner text and goes through the gate; no figures here.
// params: { headline:[dark, accent], statement, support?, points:[{ title, caption, icon }], period, brand }
// icon: "condition" | "pricing" | "timing" | "check"
function pointIcon(kind, cx, cy, r, pal) {
  const sw = Math.max(3.5, r * 0.07), g = `fill="none" stroke="${pal.gold}" stroke-width="${sw}" stroke-linecap="round" stroke-linejoin="round"`;
  let glyph = "";
  if (kind === "condition") {
    const s = r * 0.5;
    glyph = `<path ${g} d="M ${cx - s * 1.05} ${cy + s * 0.1} L ${cx - s * 0.1} ${cy - s * 0.85} L ${cx + s * 0.85} ${cy + s * 0.1} M ${cx - s * 0.75} ${cy - s * 0.05} V ${cy + s * 0.95} H ${cx + s * 0.55} V ${cy - s * 0.05}"/>
    <path ${g} d="M ${cx + s * 0.8} ${cy - s * 1.05} v ${s * 0.5} M ${cx + s * 0.55} ${cy - s * 0.8} h ${s * 0.5}"/>`;
  } else if (kind === "pricing") {
    const s = r * 0.5;
    glyph = `<path ${g} d="M ${cx - s * 0.2} ${cy - s * 1.0} H ${cx + s * 0.95} V ${cy + s * 0.15} L ${cx - s * 0.1} ${cy + s * 1.0} L ${cx - s * 1.05} ${cy + s * 0.05} Z"/>
    <circle cx="${cx + s * 0.5}" cy="${cy - s * 0.5}" r="${s * 0.12}" fill="${pal.gold}"/>
    <text x="${cx - s * 0.1}" y="${cy + s * 0.32}" text-anchor="middle" font-family="${BODY}" font-weight="700" font-size="${s * 0.9}" fill="${pal.gold}">$</text>`;
  } else if (kind === "timing") {
    const s = r * 0.52;
    glyph = `<circle cx="${cx}" cy="${cy}" r="${s}" ${g}/><path ${g} d="M ${cx} ${cy - s * 0.55} V ${cy} L ${cx + s * 0.4} ${cy + s * 0.25}"/>`;
  } else {
    const s = r * 0.5;
    glyph = `<path ${g} d="M ${cx - s * 0.8} ${cy} L ${cx - s * 0.2} ${cy + s * 0.65} L ${cx + s * 0.85} ${cy - s * 0.6}"/>`;
  }
  return `<circle cx="${cx}" cy="${cy}" r="${r}" fill="#F0E9D8"/>${glyph}`;
}

async function expertSvg(p, cv) {
  const pal = palette(p.brand);
  const portrait = cv === CANVAS.portrait;
  const X = cv.marginX;
  const pts = p.points || [];
  if (pts.length < 1 || pts.length > 3) throw new Error("renderExpertCard: points must have 1 to 3 items");
  const dark = (p.headline && p.headline[0]) || "", accent = (p.headline && p.headline[1]) || "";
  const label = (p.sourceLabel || "MetroList") + (periodLabel(p.period) ? `  ·  ${periodLabel(p.period)}` : "");
  const colW = portrait ? cv.maxW : 1000;
  const limit = portrait ? 1540 : 840;
  let L;
  for (let scale = 1; scale >= 0.6; scale -= 0.04) {
    const hs = Math.round((portrait ? 112 : 78) * scale), ss = Math.round((portrait ? 60 : 44) * scale), us = Math.round((portrait ? 36 : 27) * scale);
    const dl = await wrap(dark, colW, { family: DISPLAY, weight: 800, size: hs });
    const al = await wrap(accent, colW, { family: DISPLAY, weight: 800, size: hs });
    const stl = await wrap(p.statement || "", portrait ? 800 : 900, { family: DISPLAY, weight: 500, size: ss });
    const sul = p.support ? await wrap(p.support, portrait ? 760 : 900, { family: BODY, weight: 500, size: us }) : [];
    let y = cv.top + 4;
    const rule1 = y; y += 40;
    const hBase = y + hs * 0.82; y = hBase + (dl.length + al.length - 1) * hs * 1.14 + 28;
    const rule2 = y; y += 44;
    const stBase = y + ss * 0.8; y += stl.length * ss * 1.14 + 16;
    const suBase = y + us * 0.85; y += sul.length * us * 1.45 + (sul.length ? 20 : 0);
    // points: stacked rows (9:16) or side-by-side columns (16:9)
    const ir = portrait ? 56 : 46, rowH = portrait ? 150 : 0;
    const capS = portrait ? 22 : 19, ttS = portrait ? 44 : 36;
    const colWp = portrait ? colW : Math.floor((colW + 150 - 40 * (pts.length - 1)) / pts.length);
    const items = [];
    for (const q of pts) {
      const tw = portrait ? colW - ir * 2 - 70 : colWp - ir * 2 - 50;
      const capL = await wrap(String(q.caption || "").toUpperCase(), tw, { family: BODY, weight: 600, size: capS, spacing: 3 });
      if (capL.length > 3) throw new Error("renderExpertCard: a point caption is too long (max 3 lines)");
      items.push({ q, capL, tw });
    }
    y += 24;
    const ptsTop = y;
    const ptsH = portrait ? pts.length * rowH : ir * 2 + 10;
    y += ptsH;
    L = { hs, dl, al, stl, ss, sul, us, rule1, hBase, rule2, stBase, suBase, ir, rowH, capS, ttS, items, ptsTop, colWp, bottom: y };
    if (y <= limit) break;
  }
  let svg = "";
  L.items.forEach((it, i) => {
    const q = it.q, ir = L.ir;
    const x0 = portrait ? X : X + i * (L.colWp + 40);
    const y0 = portrait ? L.ptsTop + i * L.rowH : L.ptsTop;
    const cy = y0 + ir + 4;
    svg += pointIcon(q.icon || "check", x0 + ir, cy, ir, pal);
    const tx = x0 + ir * 2 + 38;
    const barH = L.ttS + 14 + it.capL.length * L.capS * 1.45;
    svg += `<rect x="${tx - 20}" y="${Math.round(cy - barH / 2)}" width="4" height="${Math.round(barH)}" fill="${pal.gold}"/>`;
    const top = cy - barH / 2;
    svg += T(tx, top + L.ttS * 0.84, q.title || "", { size: L.ttS, weight: 600, fill: pal.ink });
    it.capL.forEach((t, k) => { svg += T(tx, top + L.ttS + 10 + L.capS + k * L.capS * 1.45, t, { size: L.capS, weight: 600, family: BODY, fill: pal.ink, spacing: 3, opacity: 0.9 }); });
  });
  const footY = portrait ? cv.h - 64 : cv.h - 46;
  const body = `<svg xmlns="http://www.w3.org/2000/svg" width="${cv.w}" height="${cv.h}">
  <defs>
    <linearGradient id="lt" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#fff" stop-opacity="0.5"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></linearGradient>
    <linearGradient id="ll" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#fff" stop-opacity="0.5"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></linearGradient>
    <linearGradient id="lb" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#000" stop-opacity="0"/><stop offset="1" stop-color="#000" stop-opacity="0.45"/></linearGradient>
  </defs>
  ${portrait ? `<rect x="0" y="0" width="${cv.w}" height="${Math.round(cv.h * 0.7)}" fill="url(#lt)"/>` : `<rect x="0" y="0" width="${Math.round(cv.w * 0.7)}" height="${cv.h}" fill="url(#ll)"/>`}
  <rect x="0" y="${cv.h - 260}" width="${cv.w}" height="260" fill="url(#lb)"/>
  <rect x="${X}" y="${L.rule1}" width="145" height="4" fill="${pal.gold}"/>
  ${[...L.dl.map((l) => ({ l, c: pal.ink })), ...L.al.map((l) => ({ l, c: pal.gold }))].map((r, i) => T(X - 4, L.hBase + i * L.hs * 1.14, r.l, { size: L.hs, weight: 800, fill: r.c })).join("\n  ")}
  <rect x="${X}" y="${L.rule2}" width="110" height="3" fill="${pal.ink}"/>
  ${L.stl.map((l, i) => T(X, L.stBase + i * L.ss * 1.14, l, { size: L.ss, weight: 500, fill: pal.ink })).join("\n  ")}
  ${L.sul.map((l, i) => T(X, L.suBase + i * L.us * 1.45, l, { size: L.us, weight: 500, family: BODY, fill: pal.ink, opacity: 0.9 })).join("\n  ")}
  ${svg}
  ${T(X, footY, label.toUpperCase(), { size: portrait ? 24 : 20, weight: 500, family: BODY, fill: "#FFFFFF", spacing: 5, opacity: 0.95 })}
</svg>`;
  return { svg: body, warnings: [], fits: L.bottom <= limit, textBottom: L.bottom, limit };
}
async function renderExpertCard(params, outPng, canvas = "portrait") {
  const cv = CANVAS[canvas] || CANVAS.portrait;
  const r = await expertSvg(params, cv);
  await sharp(Buffer.from(r.svg)).png().toFile(outPng);
  return { png: outPng, warnings: r.warnings, fits: r.fits, textBottom: r.textBottom, limit: r.limit };
}

// Public API ------------------------------------------------------------------
// params (opening): { market, kind?, headline, eyebrow?, period:{start,end}, noticeText?, brand?, agent:{name,brokerage}, headshotBuf? }
// params (closing): { headline:[darkPart, accentPart], support, tagline?, variant?, period, ... same agent/brand }
async function renderOpeningCard(params, outPng, canvas = "portrait") {
  const cv = CANVAS[canvas] || CANVAS.portrait;
  const r = await openingSvg(params, cv);
  await sharp(Buffer.from(r.svg)).png().toFile(outPng);
  return { png: outPng, warnings: r.warnings, fits: r.fits, textBottom: r.textBottom, limit: r.limit };
}
async function renderClosingCard(params, outPng, canvas = "portrait") {
  const cv = CANVAS[canvas] || CANVAS.portrait;
  const r = await closingSvg(params, cv);
  await sharp(Buffer.from(r.svg)).png().toFile(outPng);
  return { png: outPng, warnings: r.warnings, fits: r.fits, textBottom: r.textBottom, limit: r.limit };
}

module.exports = { renderOpeningCard, renderClosingCard, renderStatCard, renderCompareCard, renderExpertCard, metroListNotice, periodLabel, CAPTION_BAND, PORTRAIT_ZONE, CANVAS };
