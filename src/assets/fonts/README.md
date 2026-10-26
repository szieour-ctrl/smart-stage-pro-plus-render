Montserrat Medium / SemiBold / Bold (SIL Open Font License 1.1), bundled so the
Railway image can render Reel overlay text. Dockerfile installs these into
/usr/local/share/fonts and runs fc-cache; sharp/librsvg then resolves
font-family="Montserrat" through fontconfig.

Playfair Display (variable, weights 400-900; SIL Open Font License 1.1) is used by the
Agent Marketing editorial cards (src/lib/agentMarketingCards.js) for headlines and names.
