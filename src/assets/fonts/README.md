Montserrat Medium / SemiBold / Bold (SIL Open Font License 1.1), bundled so the
Railway image can render Reel overlay text. Dockerfile installs these into
/usr/local/share/fonts and runs fc-cache; sharp/librsvg then resolves
font-family="Montserrat" through fontconfig.

Playfair Display (variable, weights 400-900; SIL Open Font License 1.1) is used by the
Agent Marketing editorial cards (src/lib/agentMarketingCards.js) for headlines and names.

Subscriber brand fonts (all SIL Open Font License 1.1, from the @expo-google-fonts packages): Barlow, Bodoni Moda,
Manrope, Open Sans, Roboto at Medium / SemiBold / Bold, and Lato at Regular / Bold (Lato has no Medium or SemiBold; the
nearest weight is used). These are the families a subscriber can choose in Brand DNA (brand-themes.js on Netlify).
