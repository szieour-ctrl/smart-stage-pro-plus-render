// agentMarketingLayouts.js — more opening and closing compositions for the Agent Marketing cards (Oct 26, 2026).
//
// Same Brand DNA as the base cards (colors, subscriber font, the SAME headshot, the SAME code-built MetroList notice); only the
// layout changes:
//   editorial_portrait  a large headshot panel on one side (side: "left" | "right"), headline stack beside it, name block by it
//   news_brief          a news-style top: tag chip, market name, then a dark lower-third band with the headline, a small
//                       headshot circle and the name block
// Built as a factory so it can use the card helpers without a circular require.
"use strict";

module.exports = function makeLayouts(h) {
  const { sharp, CANVAS, palette, measure, wrap, fitSize, T, dataUri, usableImage, circleHeadshot, noticeBar, needNotice, periodLabel, D, B, esc } = h;

  // Geometry per canvas. Zones (agentMarketingZones.js) read the same numbers.
  const GEOM = {
    portrait: {
      ep: { panelW: 470, panelH: 560, panelTop: 960, stackLimit: 930, nameGap: 40 },
      nb: { chipTop: 150, chipH: 64, bigTop: 250, bigLimit: 930, bandTop: 1010, bandH: 480, circle: 200 },
    },
    landscape: {
      ep: { panelW: 470, panelH: 640, panelTop: 130, stackLimit: 640, colGap: 70 },
      nb: { chipTop: 96, chipH: 54, bigTop: 180, bigLimit: 480, bandTop: 510, bandH: 320, circle: 170 },
    },
  };
  const canvasName = (cv) => (cv === CANVAS.portrait ? "portrait" : "landscape");

  function content(kind, p, pal) {
    if (kind === "opening") {
      return {
        big: [{ t: p.market || "", c: pal.ink }], kindLabel: p.kind || "Market Update", sub: p.headline || "",
        eyebrow: [p.eyebrow, periodLabel(p.period)].filter(Boolean).join("  ·  ").toUpperCase(),
      };
    }
    const hd = p.headline || [];
    return { big: [{ t: hd[0] || "", c: pal.ink }, { t: hd[1] || "", c: pal.gold }], kindLabel: "", sub: p.support || "", eyebrow: String(p.tagline || "").toUpperCase() };
  }

  // A headline stack that shrinks until it fits above `limit`. Returns { svg, bottom, fits }.
  async function stack(cv, pal, c, { X, maxW, subMaxW, top, limit, bigSize, kindSize, subSize, eyeSize }) {
    let out = null;
    for (let scale = 1; scale >= 0.5; scale -= 0.04) {
      const bs = Math.round(bigSize * scale);
      const bigLines = [];
      for (const part of c.big) if (part.t) (await wrap(part.t, maxW, { family: D(), weight: 800, size: bs })).forEach((l) => bigLines.push({ l, c: part.c }));
      const ss = Math.round(subSize * scale), es = Math.round(eyeSize * scale);
      const subLines = c.sub ? await wrap(c.sub, subMaxW, { family: D(), weight: 500, size: ss }) : [];
      const eyeLines = c.eyebrow ? await wrap(c.eyebrow, maxW, { family: B(), weight: 500, size: es, spacing: 5 }) : [];
      let ks = 0;
      if (c.kindLabel) ks = await fitSize(c.kindLabel, maxW, Math.min(Math.round(kindSize * scale), Math.round(bs * 0.68)), 36, { family: D(), weight: 600 });
      let y = top + 4;
      const rule1 = y; y += 36;
      const bigBase = y + bs * 0.8; y = bigBase + (bigLines.length - 1) * bs * 1.06 + 12;
      let kindBase = 0; if (ks) { kindBase = y + ks * 0.82; y = kindBase + 30; }
      const rule2 = y; y += 44;
      const subBase = y + ss * 0.8; y += subLines.length * ss * 1.14 + 16;
      const eyeBase = eyeLines.length ? y + es : 0; if (eyeLines.length) y = eyeBase + (eyeLines.length - 1) * es * 1.5 + 26;
      const rule3 = y; y += 6;
      const svg = `<rect x="${X}" y="${rule1}" width="145" height="4" fill="${pal.gold}"/>
  ${bigLines.map((r, i) => T(X - 4, bigBase + i * bs * 1.06, r.l, { size: bs, weight: 800, fill: r.c })).join("\n  ")}
  ${ks ? T(X - 2, kindBase, c.kindLabel, { size: ks, weight: 600, fill: pal.gold }) : ""}
  <rect x="${X}" y="${rule2}" width="110" height="3" fill="${pal.ink}"/>
  ${subLines.map((l, i) => T(X, subBase + i * ss * 1.14, l, { size: ss, weight: 500, fill: pal.ink })).join("\n  ")}
  ${eyeLines.map((l, i) => T(X, eyeBase + i * es * 1.5, l, { size: es, weight: 500, family: B(), fill: pal.ink, spacing: 5 })).join("\n  ")}
  <rect x="${X}" y="${rule3}" width="90" height="3" fill="${pal.gold}"/>`;
      out = { svg, bottom: y, fits: y <= limit };
      if (y <= limit && bigLines.length <= 3) break;
    }
    return out;
  }

  // A name block (name + brokerage) at x, with its vertical middle at cy. Returns svg and its height.
  async function nameBlock(cv, pal, agent, { x, cy, maxW, color = "#FFFFFF", nameSize, brkSize, bar = true }) {
    const name = agent.name || "";
    const brokerage = String(agent.brokerage || "").toUpperCase();
    const ns = await fitSize(name, maxW, nameSize, 30, { family: D(), weight: 600 });
    const nameLines = (await measure(name, { family: D(), weight: 600, size: ns })) <= maxW ? [name] : await wrap(name, maxW, { family: D(), weight: 600, size: ns });
    const brkFont = { family: B(), weight: 500, size: brkSize, spacing: 5 };
    const brkLines = brokerage ? (await wrap(brokerage, maxW, brkFont)).slice(0, 3) : [];
    const nLH = Math.round(ns * 1.12), bLH = brkSize + 12;
    const hgt = nameLines.length * nLH + (brkLines.length ? 14 + brkLines.length * bLH : 0);
    const top = cy - hgt / 2;
    let svg = bar ? `<rect x="${x - 22}" y="${Math.round(top)}" width="4" height="${Math.round(hgt)}" fill="${pal.gold}"/>` : "";
    nameLines.forEach((l, i) => { svg += T(x, top + ns * 0.86 + i * nLH, l, { size: ns, weight: 600, fill: color }); });
    brkLines.forEach((l, i) => { svg += T(x, top + nameLines.length * nLH + 14 + brkSize + i * bLH, l, { size: brkSize, weight: 500, family: B(), fill: color, spacing: 5, opacity: 0.95 }); });
    return { svg, height: hgt, top };
  }

  async function panelImage(buf, w, hgt) {
    return sharp(buf).rotate().resize(w, hgt, { fit: "cover", position: "attention" }).png().toBuffer();
  }

  // ---------- editorial portrait ----------
  async function editorialPortrait(kind, p, cv) {
    const pal = palette(p.brand), agent = p.agent || {}, g = GEOM[canvasName(cv)].ep, portrait = cv === CANVAS.portrait;
    const side = p.side === "right" ? "right" : "left";
    const notice = needNotice(p);
    const nb = await noticeBar(cv, pal, notice);
    const haveHs = !!(await usableImage(p.headshotBuf));
    const c = content(kind, p, pal);
    const warnings = notice ? [] : ["no MetroList notice (noNotice was set)"];
    if (!haveHs) warnings.push("no headshot: the portrait panel was left out");
    const panelX = portrait ? (side === "left" ? cv.marginX : cv.w - cv.marginX - g.panelW) : (side === "left" ? cv.marginX : cv.w - cv.marginX - g.panelW);
    // Text column: full width above the panel (9:16) or beside it (16:9).
    const X = portrait ? cv.marginX : (haveHs && side === "left" ? cv.marginX + g.panelW + g.colGap : cv.marginX);
    const maxW = portrait ? cv.maxW : (haveHs ? cv.w - 2 * cv.marginX - g.panelW - g.colGap : cv.maxW);
    const st = await stack(cv, pal, c, { X, maxW, subMaxW: portrait ? 700 : Math.min(maxW, cv.subMaxW), top: cv.top, limit: g.stackLimit, bigSize: portrait ? 150 : 128, kindSize: portrait ? 104 : 88, subSize: cv.sub, eyeSize: cv.eyebrow });
    let panel = "", identity = "";
    if (haveHs) {
      const img = await panelImage(p.headshotBuf, g.panelW, g.panelH);
      const pt = portrait ? g.panelTop : g.panelTop;
      panel = `<clipPath id="pnl"><rect x="${panelX}" y="${pt}" width="${g.panelW}" height="${g.panelH}" rx="30"/></clipPath>
  <image href="${dataUri(img)}" x="${panelX}" y="${pt}" width="${g.panelW}" height="${g.panelH}" clip-path="url(#pnl)" preserveAspectRatio="xMidYMid slice"/>
  <rect x="${panelX}" y="${pt}" width="${g.panelW}" height="${g.panelH}" rx="30" fill="none" stroke="${pal.gold}" stroke-width="6"/>`;
    }
    if (agent.name) {
      if (portrait) {
        const nx = (haveHs ? (side === "left" ? panelX + g.panelW + g.nameGap + 22 : cv.marginX + 22) : cv.marginX + 22);
        const nw = haveHs ? cv.maxW - g.panelW - g.nameGap - 22 : cv.maxW - 22;
        const nbk = await nameBlock(cv, pal, agent, { x: nx, cy: g.panelTop + g.panelH - 120, maxW: nw, color: pal.ink, nameSize: 58, brkSize: 22 });
        identity = nbk.svg;
      } else {
        const nbk = await nameBlock(cv, pal, agent, { x: X + 26, cy: cv.lockupCy, maxW: Math.min(maxW, 760), color: pal.ink, nameSize: 52, brkSize: 20 });
        identity = nbk.svg;
      }
    }
    // Soft scrim under the text column so the headline reads on a busy plate.
    const scrim = portrait
      ? `<linearGradient id="st" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#fff" stop-opacity="0.55"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></linearGradient>
  <rect x="0" y="0" width="${cv.w}" height="${Math.round(cv.h * 0.56)}" fill="url(#st)"/>
  <linearGradient id="sb" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#fff" stop-opacity="0"/><stop offset="1" stop-color="#fff" stop-opacity="0.45"/></linearGradient>
  <rect x="0" y="${g.panelTop - 40}" width="${cv.w}" height="${g.panelH + 80}" fill="url(#sb)"/>`
      : `<linearGradient id="st" x1="${side === "left" && haveHs ? 1 : 0}" y1="0" x2="${side === "left" && haveHs ? 0 : 1}" y2="0"><stop offset="0" stop-color="#fff" stop-opacity="0.55"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></linearGradient>
  <rect x="${side === "left" && haveHs ? Math.round(cv.w * 0.3) : 0}" y="0" width="${Math.round(cv.w * 0.7)}" height="${cv.h}" fill="url(#st)"/>`;
    const lbScrim = `<linearGradient id="lb" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#000" stop-opacity="0"/><stop offset="1" stop-color="#000" stop-opacity="0.6"/></linearGradient>
  <rect x="0" y="${cv.h - cv.notice.h - 260}" width="${cv.w}" height="${cv.notice.h + 260}" fill="url(#lb)"/>`;
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${cv.w}" height="${cv.h}">
  <defs></defs>
  ${scrim}
  ${lbScrim}
  ${st.svg}
  ${panel}
  ${identity}
  ${nb.svg}
</svg>`;
    return { svg, warnings, textBottom: st.bottom, limit: g.stackLimit, fits: st.fits };
  }

  // ---------- news brief ----------
  async function newsBrief(kind, p, cv) {
    const pal = palette(p.brand), agent = p.agent || {}, g = GEOM[canvasName(cv)].nb, portrait = cv === CANVAS.portrait;
    const notice = needNotice(p);
    const nb = await noticeBar(cv, pal, notice);
    const X = cv.marginX;
    const haveHs = !!(await usableImage(p.headshotBuf));
    const warnings = notice ? [] : ["no MetroList notice (noNotice was set)"];
    const c = content(kind, p, pal);
    // Tag chip: the kind label for an opening ("MARKET UPDATE"), the tagline for a closing.
    const tagText = (kind === "opening" ? c.kindLabel : (c.eyebrow || "LET'S TALK")).toUpperCase();
    const tagSize = portrait ? 28 : 22;
    const tagW = Math.round((await measure(tagText, { family: B(), weight: 700, size: tagSize, spacing: 4 })) + 48);
    const chip = `<rect x="${X}" y="${g.chipTop}" width="${tagW}" height="${g.chipH}" rx="8" fill="${pal.gold}"/>
  ${T(X + 24, g.chipTop + g.chipH / 2 + tagSize * 0.36, tagText, { size: tagSize, weight: 700, family: B(), fill: pal.ink, spacing: 4 })}`;
    const datedLine = kind === "opening" ? [p.eyebrow, periodLabel(p.period)].filter(Boolean).join("  ·  ").toUpperCase() : "";
    const dated = datedLine ? T(X + tagW + 28, g.chipTop + g.chipH / 2 + tagSize * 0.36, datedLine, { size: tagSize - 2, weight: 500, family: B(), fill: pal.ink, spacing: 4, opacity: 0.9 }) : "";

    // Big text: the market (opening) or the headline lines (closing), shrunk to fit above the band.
    const bigParts = c.big.filter((x) => x.t);
    const maxBig = portrait ? 150 : 124;
    let bs = maxBig, bigLines = [];
    for (; bs >= 56; bs -= 4) {
      bigLines = [];
      for (const part of bigParts) (await wrap(part.t, cv.maxW, { family: D(), weight: 800, size: bs })).forEach((l) => bigLines.push({ l, c: part.c }));
      if (g.bigTop + bs * 0.8 + (bigLines.length - 1) * bs * 1.06 <= g.bigLimit && bigLines.length <= 4) break;
    }
    const bigBottom = g.bigTop + bs * 0.8 + (bigLines.length - 1) * bs * 1.06;
    const big = bigLines.map((r, i) => T(X - 4, g.bigTop + bs * 0.8 + i * bs * 1.06, r.l, { size: bs, weight: 800, fill: r.c })).join("\n  ");

    // The dark lower-third band: headline in white, headshot circle, name block.
    const circleD = g.circle;
    const textW = haveHs ? cv.maxW - circleD - 40 : cv.maxW;
    let ss = Math.round(cv.sub * (portrait ? 0.95 : 0.9)), subLines = [];
    for (; ss >= 34; ss -= 2) {
      subLines = await wrap(kind === "opening" ? c.sub : c.sub, textW, { family: D(), weight: 500, size: ss });
      if (subLines.length * ss * 1.14 <= g.bandH - (portrait ? 230 : 150)) break;
    }
    const subTop = g.bandTop + (portrait ? 70 : 46);
    const subSvg = subLines.map((l, i) => T(X, subTop + ss * 0.8 + i * ss * 1.14, l, { size: ss, weight: 500, fill: "#FFFFFF" })).join("\n  ");
    const subBottom = subTop + subLines.length * ss * 1.14;
    let identity = "";
    if (agent.name) {
      const nbk = await nameBlock(cv, pal, agent, { x: X + 22, cy: g.bandTop + g.bandH - (portrait ? 88 : 62), maxW: Math.min(textW, 720), color: "#FFFFFF", nameSize: portrait ? 46 : 38, brkSize: portrait ? 20 : 17 });
      identity = nbk.svg;
    }
    let circle = "";
    if (haveHs) {
      const img = await circleHeadshot(p.headshotBuf, circleD);
      const cx = cv.w - X - circleD, cy = g.bandTop + (portrait ? 70 : 40);
      circle = `<clipPath id="nbc"><circle cx="${cx + circleD / 2}" cy="${cy + circleD / 2}" r="${circleD / 2}"/></clipPath>
  <image href="${dataUri(img)}" x="${cx}" y="${cy}" width="${circleD}" height="${circleD}" clip-path="url(#nbc)" preserveAspectRatio="xMidYMid slice"/>
  <circle cx="${cx + circleD / 2}" cy="${cy + circleD / 2}" r="${circleD / 2}" fill="none" stroke="${pal.gold}" stroke-width="5"/>`;
    } else warnings.push("no headshot: the circle was left out");
    const scrim = `<linearGradient id="nt" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#fff" stop-opacity="0.55"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></linearGradient>
  <rect x="0" y="0" width="${cv.w}" height="${g.bandTop}" fill="url(#nt)"/>
  <linearGradient id="lb" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#000" stop-opacity="0"/><stop offset="1" stop-color="#000" stop-opacity="0.6"/></linearGradient>
  <rect x="0" y="${g.bandTop + g.bandH}" width="${cv.w}" height="${cv.h - g.bandTop - g.bandH}" fill="url(#lb)"/>`;
    const band = `<rect x="0" y="${g.bandTop}" width="${cv.w}" height="${g.bandH}" fill="${pal.ink}" fill-opacity="0.94"/>
  <rect x="0" y="${g.bandTop}" width="${cv.w}" height="5" fill="${pal.gold}"/>
  <rect x="0" y="${g.bandTop}" width="16" height="${g.bandH}" fill="${pal.gold}"/>`;
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${cv.w}" height="${cv.h}">
  ${scrim}
  ${chip}
  ${dated}
  ${big}
  ${band}
  ${subSvg}
  ${circle}
  ${identity}
  ${nb.svg}
</svg>`;
    const fits = bigBottom <= g.bigLimit && subBottom <= g.bandTop + g.bandH - (portrait ? 150 : 120);
    return { svg, warnings, textBottom: Math.round(bigBottom), limit: g.bigLimit, fits };
  }

  const LAYOUTS = { editorial_portrait: editorialPortrait, news_brief: newsBrief };
  return { LAYOUTS, GEOM };
};
