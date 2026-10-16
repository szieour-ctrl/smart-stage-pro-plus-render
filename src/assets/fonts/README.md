Montserrat Medium / SemiBold / Bold (SIL Open Font License 1.1), bundled so the
Railway image can render Reel overlay text. Dockerfile installs these into
/usr/local/share/fonts and runs fc-cache; sharp/librsvg then resolves
font-family="Montserrat" through fontconfig.
