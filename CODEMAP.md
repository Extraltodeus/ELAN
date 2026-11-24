# ÉLAN

Jeu de vol à inertie en 3D : rendu raymarché entièrement en PyTorch, fenêtre/entrées/son via Pygame.

Lancer : `run.bat` ou `python main.py` (options : `--windowed`, `--screen N` écran 1, 2, 3… (sinon réglage SCREEN), `--gpu N`, `--bake-gpu N` génération + audio sur un autre GPU, `--scale 0.75|auto`, `--fps N`, `--vsync`, `--seed N`, `--mute`, `--perf`, `--autoplay`, `--frames N`, `--shots DIR`, `--sector N` départ au secteur N, `--quota N` quota plafonné pour les tests, `--pilot FICHIER` autre pilote IA, `--developper` mode développeur, `--mode chill|story` force le mode, `--finale` démarre directement dans la Singularité, `--finale-track FICHIER.wav` joue un autre WAV dans la finale, avec les points de boucle du `.json` voisin).

## Fichiers
- main.py — entrée, arguments
- elan/config.py — toutes les constantes (monde, tunnel, rendu, vol, règles)
- elan/app.py — boucle principale, états (titre, options, scores, crédits, jeu, pause, game over / fin d'histoire), enchaînement secteur → tunnel → secteur selon le mode, records par mode, encaissement du score à chaque fin de partie (`_bank` / `_record`), verrou du pilote IA (`_toggle_pilot`, `piloted`)
- elan/modes.py — modes de jeu (Mode, ChillMode, StoryMode) : pente de consommation, nombre de niveaux, chrono, clés de records, nature du tunnel / monde suivant
- elan/title.py — menu du titre (TitleMenu) : rangée MODE, rangée PILOT, liens OPTIONS / SCORES / CREDITS (`LINKS`), souris / molette, mode mémorisé
- elan/scores.py — tableau des scores (ScoreBoard) : top `AppCfg.SCORES_KEPT` par mode dans `elan_scores.json`, repris du record existant au premier lancement
- elan/credits.py — générique : contenu (`CREDITS`) et défilement (CreditsRoll : auto, molette, retour), qui ralentit (`EASE`, `CREEP`) puis s'arrête sur le dernier message centré au lieu de boucler ; la butée (`limit`) vient de CreditsScreen, qui isole ce message d'un demi-écran
- elan/display.py — fenêtre, choix de l'écran (`screen`, `move_to` : déplacement à chaud via `pygame._sdl2`, plein écran compris), copie GPU → surface, cadence, résolution dynamique
- elan/input.py — touches réassignables (2 emplacements/action), lissage souris, assignations manette, navigation menus, MenuKeys
- elan/settings.py — réglages et leurs valeurs par défaut (`DEFAULTS`), assignations par défaut (`DEFAULT_BINDINGS`, `DEFAULT_PAD`, `DEFAULT_PAD_LOOK`), mode choisi, records chill (score, secteur) et story (score, niveau, temps, nombre de fins), mondes vus (JSON)
- elan/career.py — progression par mode : secteur / niveau record, rang, mondes découverts, messages NEW FRONTIER / NEW WORLD
- elan/menu.py — menu pause (PauseMenu) : page principale, page OPTIONS (mêmes réglages dont SCREEN, ouverte depuis le titre par `open(True)`), page CONTROLS (clavier/souris + manette) avec capture
- elan/layout.py — emplacement de la minimap (partagé rendu / HUD)
- elan/gamepad.py — manette : lecture standardisée (API contrôleur SDL, repli joystick brut), boutons/axes virtuels, libellés
- elan/gyro.py — gyroscope de la manette (capteur SDL via ctypes) et pilotage par inclinaison
- elan/route.py — planificateur de ribambelles (numpy) : ribambelle en cours, curseur, entrée de la suivante, liens suggérés ; partagé HUD / minimap / pilote IA
- elan/ai/ — pilote appris par renforcement (v2)
  - sim.py — simulateur par lots sur GPU (physique du jeu, flux, boost, tunnels, bouclier, récompense orientée vitesse ; options expérimentales : vitesse moyenne glissante `Reward.WINDOW`, vitesse linéaire `Reward.CURVE`)
  - field.py, course.py — champ de distance multi-mondes, motes, anneaux, planification de route par lots (miroir de route.py)
  - observe.py — observation (vitesse, flux, points de passage, entrée suivante, sondes d'obstacles)
  - coach.py — contrôleur scripté (poursuite + évitement) servant de professeur pour le pré-entraînement par imitation
  - policy.py, train.py — réseau acteur-critique, imitation puis PPO avec curriculum de difficulté (`python -m elan.ai.train`, `--resume` pour affiner — le critique est recalé d'abord, `--watch` pour évaluer ; expérimental et instable en affinage : `--window S` vitesse moyenne sur S secondes, `--linear` vitesse linéaire)
  - shield.py — bouclier de trajectoire : simule l'arc de virage de la commande sur ~1 s et la remplace par la commande sûre la plus proche si l'arc touche une paroi (Shield en jeu, BatchShield identique pour le simulateur)
  - pilot.py — pilote embarqué (numpy), utilisé si `assets/ai/pilot.pt` est en version 2 ; sinon autopilote classique. Garde-fous scriptés : bouclier, réserve de flux avant boost, pas de boost en gate rush, approche de porte alignée, frein prédictif. Le pilote garde la main en panne sèche (plus de repli sur l'autopilote classique)
  - bench.py — banc d'essai sans rendu sur les vraies règles du jeu (`python -m elan.ai.bench [pilote|classic] --seconds 300`)
- elan/crashlog.py — journal d'erreurs elan_error.log (exceptions, threads, plantages durs)
- elan/devtools.py — mode développeur (`--developper`) : raccourcis de test, légende HUD, parties sans record
- elan/physics.py — planeur : portance (décroche à basse vitesse, adhérence `grip`), traînée, sillage, grappin, collisions, ricochets ; sous l'eau forme dard ; poussée warp en tunnel
- elan/rail.py — guidage en tunnel (TunnelRail) : rappel vers l'axe, suivi des courbes, visée assistée bornée, pas de paroi ni de marche arrière
- elan/field.py — champ de distance côté CPU (collisions), résolution déduite du volume
- elan/game.py — règles : flux, panne sèche (adhérence qui fond avec la vitesse) et décrochage, motes (ramassage balayé), anneaux, paliers FLOW, combo, quota, porte, phase tunnel (trajet, sortie, surge de distorsion), plongée dans le trou noir (`plunge`, `horizon`), finale (`_settle` : score et chrono scellés, flux infini, portail ouvert), HUD state
- elan/camera.py — caméra poursuite/cockpit, orientation du planeur affiché, vrille et recul en forme dard
- elan/cruise.py — croisière du fond de titre quand le monde est la Singularité (après une fin d'histoire) : ni motes ni porte, donc vol calme à vitesse modérée, cap qui dérive doucement, évitement souple par rayons, altitude tenue
- elan/autopilot.py — pilote automatique classique (repli), suit la ligne du tunnel, vol plané de récupération quand le flux est à zéro

### elan/world — génération (GPU)
- noise.py — bruit spectral (FFT)
- palette.py — biomes (12 : couleurs, forme, style de matière), ordre des secteurs, palette des tunnels, tunnel EVENT HORIZON (`plunge`), monde final SINGULARITY (`finale`)
- forms.py — formes de terrain : tours, arches, îles, canyon en terrasses, orgues basaltiques (Voronoï), treillis gyroïde, tores, pliage kaléidoscopique ; finale : archipel (`isles`), petite cité dense (`citadel`), flèches perçant les nuages (`spires`)
- clouds.py — mer de nuages (CloudBaker) : volume densité + éclairage solaire pré-calculé, activée par le style `cloud_deck`
- terrain.py — assemblage des couches, érosion, strates, redistanciation
- tunnel.py — tunnel warp : trajectoire périodique à 3 passes (TunnelPath : axe, cap, courbure), champ de distance, coordonnées (s, angle), lignes de motes en hélice autour de l'axe, anneaux
- fieldops.py — gradients, EDT, échantillonneur périodique
- lighting.py — ombres du soleil et ouverture du ciel pré-calculées
- sky.py — ciel équirectangulaire, planète annelée, étoiles, brouillard, aurores, pliage mandala, lune
- detail.py — volume de détail roche, rides de la mer
- paths.py — rubans de motes (marge aux parois, cônes de sortie), rubans immergés, anneaux, départ aligné sur une ribambelle, porte ; style `cruise` : couloirs en altitude au-dessus du décor
- bake.py — orchestration, objet World (secteur ou tunnel), envoi sur le GPU de rendu ; natures demandées : `sector`, `tunnel`, `plunge`, `singularity`
- baker_thread.py — file de génération en arrière-plan (tunnel puis secteur suivant, GPU de rendu sauf `--bake-gpu`), demandes urgentes, purge des mondes non attendus (`expect`), cache GPU rendu hors vol seulement (`trim`, appelé en pause / titre / game over)

### elan/render — rendu (PyTorch)
- renderer.py — orchestration d'une frame, View
- uniforms.py — paramètres par frame en un seul envoi
- space.py — conversions monde ↔ volume, rééchantillonnage
- march.py — marche de rayons multi-résolution (cône 1/4, 1/2, plein)
- shade.py — normales, matière, éclairage, caustiques ; styles : néon (veines + lignes de niveau), iridescence, phare, parois de tunnel (anneaux et traits lumineux)
- lightfield.py — champ de lumière des motes / anneaux / porte (volume flouté, mis à jour quand on ramasse)
- flow.py — couleurs et défilement du tunnel selon l'avancée (arc-en-ciel en hyper), phase d'iridescence
- atmosphere.py — ciel, mer miroir, brouillard, rayons volumétriques, eau trouble en plongée, brume de tunnel
- clouds.py — nuages volumétriques (CloudLayer) : marche au quart de résolution dans la tranche d'altitude, tramage ordonné + filtrage, masquage par la profondeur du décor
- horizon.py — lentille du trou noir (HorizonLens) : chute (ombre, anneau d'Einstein, image miroir, torsion, séparation chromatique) et émergence (hublot qui s'ouvre sur le monde final)
- scene.py — points lumineux : poussière, étincelles, grappin, étoiles, assemblage
- glow.py — motes (grappes à deux coquilles + rails), balises, anneaux, portail sphérique
- lens.py — distorsion wormhole autour du portail ouvert
- craft.py — planeur filaire, morphing planeur ↔ dard, étirement et ailes en flèche avec la vitesse, gaine de cavitation, traînées, teinte rouge d'alarme (`alarm`)
- ramp.py — dégradé de couleur des motes selon la proximité (OKLCH, échelle logarithmique)
- points.py — projection et splat des points ; calque des motes (tampons colorés, contour sombre)
- minimap.py — carte de ribambelles vue de dessus (résolution d'écran) : coupe du relief à l'altitude du joueur, ribambelles colorées selon l'altitude relative, extrémités, anneaux, porte, liens suggérés, trajectoire projetée
- post.py — rémanence, lentille du trou noir (avant le bloom), bloom, aberration de vitesse, distorsion warp, torsion + étirement de sortie de tunnel (surge), tonemap, incrustation minimap, sortie BGRX

### elan/audio — son procédural
- library.py — dossier musique : `assets/music` (WAV) s'il existe, sinon `assets/music_ogg` ; `locate` (.wav puis .ogg), décodage OGG via pygame.mixer
- synth.py, music.py, soundbank.py — synthèse des boucles et effets (dont `flux_low` : battement de cœur sourd avant la panne, `flux_dry` : souffle grave de la panne sèche)
- director.py — mixage adaptatif, rotation des morceaux par secteur avec préchargement, effets joués sur des voix dédiées, battement d'alerte qui accélère et monte quand le flux approche de zéro (`FLUX_PERIOD`, `FLUX_VOLUME`)
- lowpass.py — filtre passe-bas par FFT sur CPU (couche « sous l'eau » des morceaux), sans mémoire GPU
- voices.py — réserve de voix pour les effets (VoicePool) : une pour les motes, une pour le reste, jamais les canaux de boucles ou de musique
- playlist.py — liste de lecture (`music.json`), rotation des morceaux
- soundtrack.py — musique générée en lecture linéaire (`assets/music/`), changement de morceau en fondu
- dj.py — DJ interne : stems découpés par mesure, sections, sauts calés sur les phrases, mixage selon le jeu, bascule de morceau ; les 4 pistes changent de mesure ensemble (réalignement automatique, arrêt sans mesure résiduelle)
- feature.py — pistes dédiées (FeatureTrack) : `assets/music/finale.wav` (finale, coupe DJ / linéaire / synthé) et `credits.wav` (générique, coupe la musique du menu), jouées depuis le début puis bouclées selon leur entrée de `music.json`, chacune sur son canal
- finale sans `finale.wav` (`mix.serene`) : DJ = stem mélodie seul (ni batterie ni basse), linéaire = couche filtrée, synthé = nappe + arpège doux

### elan/ui — interface
- hud.py (dont barre de tunnel, marqueur de l'entrée suivante, panneau développeur, chrono, libellé niveau / secteur, score remplacé par AUTO-PLAY dans une partie sans score), screens.py (titre à deux rangées + liens en bas à droite, records du mode, écran de fin : game over ou fin d'histoire avec temps et top 10 du mode, ScoreTable / ScoresScreen : tableaux CHILL et STORY, générique défilant), text.py, shapes.py, theme.py

## Contrôles (réassignables dans OPTIONS ou Pause → CONTROLS)
Souris : viser · W/Z : poussée · S : aérofrein · A/Q, D : latéral · Espace : boost · Clic gauche ou E (tenu) : grappin · Shift gauche (tenu) : dérive · V : caméra poursuite/cockpit · Échap : pause · P : recommencer · F2 : pilote IA (parties AUTO-PLAY seulement) · F3 : stats · F11 : plein écran · F12 : capture · M : muet.
Manette : stick gauche viser · RT poussée · LT frein · stick droit latéral · RB boost · A grappin · LB dérive · Y caméra · X gyro · B recentrer le gyro · Start pause.

## Modes
- CHILL : infini, consommation `FLUX_DRAIN + secteur × FLUX_DRAIN_SECTOR`, but = score.
- STORY : `StoryCfg.LEVELS` (42) niveaux, pente de consommation × `StoryCfg.DRAIN_GAIN` (1.8), chronométré, game over = retour au niveau 1. Après le niveau 42 : tunnel EVENT HORIZON (flux infini, poussée croissante, le trou noir avale l'écran) → THE SINGULARITY : score et chrono figés à l'arrivée, flux infini, plus de motes ni d'anneaux, sillage FLOW qui monte à plein en `1 / StoryCfg.FINALE_FLOW` s et y reste (~230 m/s en boost au lieu de ~150), portail blanc déjà ouvert (le pilote IA ne le vise qu'après `StoryCfg.PORTAL_DELAY` s) → écran des scores → confirmation = menu principal. Records story séparés (score, meilleur temps, nombre de fins).
- Le fond du titre vole toujours en règles chill ; le mode s'applique au lancement d'une partie.
- Pilote : FLY ou AUTO-PLAY, choisi sur le titre pour toute la partie. FLY : F2 est verrouillé. AUTO-PLAY : F2 rend / reprend la main, aucun score n'est compté (`Game.scoring`), rien n'entre au tableau.
- Scores : toute partie FLY non assistée est encaissée à sa fin (défaite, fin d'histoire, retour au menu, recommencer, quitter) : record du mode + tableau.

## Jeu
Le flux se vide ; motes, anneaux, frôlements, ricochets et sorties d'eau le rechargent. Enchaîner crée un sillage (moins de traînée) → paliers FLOW I/II/III. Mer : angle rasant = ricochet, angle franc = plongée. Quota rempli → porte → tunnel warp sur rail (poussée automatique, recentrage, ni paroi ni marche arrière, flux qui ne se vide pas, lignes de motes en hélice à 5–15 m de l'axe : une partie se ramasse sans effort, le reste en suivant la ligne) → distorsion de sortie → secteur suivant, départ aligné sur une ribambelle. Tous les 10 secteurs : tunnel hyper (3× plus long, flux infini, poussée supplémentaire).
Flux à zéro : plus de poussée, et l'adhérence fond avec la vitesse selon une courbe (`DRY_GRIP_SPEED` 120 : début, ~1/3 perdu à 60, dérive totale à `DRY_DRIFT_SPEED` 15 ; exposant `DRY_CURVE`) ; à l'arrêt près du sol ou de la mer → STALLED → game over. Sous 15 % de flux, un battement de cœur sourd s'accélère jusqu'à la panne ; tant que le flux est à zéro, le vaisseau respire en rouge au rythme d'un souffle grave (`Game._breathe`, `RulesCfg.BREATH_*`).
Biomes : EMBER DUSK, GLACIER DAWN, NEON CANYON (nuit), TOXIC BLOOM, ABYSSAL REEF, BLOOD MOON, BASALT ORGAN, MIDNIGHT AURORA (nuit), MIRROR VOID (espace), VIOLET HOUR, PRISM MANDALA (kaléidoscope), LIQUID DREAM (iridescent) ; ordre fixe au premier cycle, mélangé ensuite.

## IA : recette d'entraînement
`python -m elan.ai.train --gpu 0 --minutes 50 --out assets/ai/x.pt` (imitation du coach puis PPO, curriculum de flux), puis affinages `--resume x.pt --crash 3 --noise 0.08 --minutes 20`. Comparer avec `python -m elan.ai.bench x.pt` (secteurs franchis, game overs, impacts/min) sur au moins 24 vols avant de copier vers `assets/ai/pilot.pt`. Toute règle ajoutée dans pilot.py doit être reproduite dans sim.py (et route.py ↔ course.py, Shield ↔ BatchShield) ; exception : la finale n'est pas simulée (balade sans plafond de vitesse le long des couloirs fantômes `game.course()`, portail après délai `game.homing()`). Le rail de tunnel (rail.py) n'est pas reproduit dans sim.py : le tunnel simulé reste libre, donc plus dur que le vrai.

## Tests manuels utiles
`python main.py --windowed --autoplay --frames 600 --perf --shots D:/tmp/elan/shots` (pilote automatique, captures).
`python main.py --windowed --autoplay --sector 10 --quota 4` (atteindre vite un tunnel hyper).
`python main.py --windowed --developper` : F1 légende · F4 flux à zéro · F5 flux plein · F6 ouvrir la porte · F7 tunnel warp · F8 tunnel hyper · F9 fin du tunnel · F10 trou noir tout de suite (story) · PageDown / PageUp secteur suivant / précédent.
`python main.py --windowed --mode story --sector 42 --quota 4 --autoplay` (fin d'histoire complète en ~1 min) ; `python main.py --windowed --finale` (monde final seul).
Sans sortie sonore : variable d'environnement `SDL_AUDIODRIVER=dummy` (le moteur audio tourne, rien ne sort des enceintes).
Les parties AUTO-PLAY ou lancées avec `--autoplay`, `--sector`, `--quota`, `--finale`, `--developper` ne touchent ni aux records ni au tableau des scores (F2 reste libre en mode développeur).

## Matériel
- Par défaut tout tourne sur un seul GPU : rendu, génération des mondes et synthèse audio sur `--gpu` (0 par défaut, `App._devices`) — plus fluide en pratique que la génération sur le 2e GPU ; `--bake-gpu N` la déplace sur un autre. Vérifié sur une RTX 4070 seule : ~58 fps en 1280×720 à 50 %, quelques à-coups pendant la génération du monde suivant, pic de ~3,6 Go alloués (~5,4 Go réservés). Sans CUDA : repli CPU qui démarre (~2 min de chargement) mais reste injouable (~1 image/s en 320×180), signalé sur l'écran de chargement ; un GPU NVIDIA est requis en pratique.
- Sans musique (ni `assets/music` ni `assets/music_ogg`) : musique synthétisée ; sans `assets/ai/pilot.pt` : autopilote classique ; sans `assets/splash.png` : pas d'image de chargement.

## Outils
- tools/convert_music.py — `assets/music` (WAV) → `assets/music_ogg` (Vorbis q5 via ffmpeg, mêmes noms, `music.json` copié ; seuls les fichiers modifiés, `--force` pour tout refaire)
- tools/make_icon.py — icône procédurale → `assets/icon.png` (fenêtre) et `assets/icon.ico`

## Fichiers suivis dans assets/
`assets/icon.png`, `assets/icon.ico`, `assets/ai/pilot.pt`, `assets/music_ogg/`

## Fichiers ignorés par git
`elan_settings.json`, `elan_scores.json`, `elan_error.log`, `screenshots/`, `TODO*.txt`, le reste de `assets/` (WAV d'origine dont `finale.wav`, stems Demucs, autres pilotes IA)
