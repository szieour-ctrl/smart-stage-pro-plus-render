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
  const markup = `<span font_family="${DISPLAY}" font_weight="${weight}" font_features="lnum" foreground="${color}" size="${Math.round(size * 1024)}">${esc(text)}</span>`;
  const { data, info } = await sharp({ text: { text: markup, rgba: true, dpi: 72 } }).trim().png().toBuffer({ resolveWithObject: true });
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

module.exports = { renderOpeningCard, renderClosingCard, renderStatCard, metroListNotice, periodLabel, CAPTION_BAND, PORTRAIT_ZONE, CANVAS };
