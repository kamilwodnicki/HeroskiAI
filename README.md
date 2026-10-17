# HeroskiAI

Projekt uczenia maszynowego do gry w Heroes of Might and Magic III (+ Horn of the Abyss).
Na start: model skupiony wyłącznie na **bitwach** (silnik VCMI headless + biblioteka `vcmi-gym`
dla Gymnasium/Pythona). Model strategiczny (mapa, ekonomia, miasta) jest odłożony na później.

## Wymagania

- Docker (Docker Desktop na Windowsie działa tu w trybie linuksowym, bez dodatkowych kroków).
- Własne, legalne pliki gry Heroes III: Complete + Horn of the Abyss.

## Jak uruchomić

1. Skopiuj do folderu `game_data/` trzy foldery ze swojej instalacji gry: `Data`, `Maps`, `Mp3`.
   Plik `vcmi-config/modSettings.json` (już w repo) aktywuje mod `ml`, wymagany przez mapy
   treningowe `vcmi-gym` — jest podłączany osobno w `docker-compose.yml`, nie trzeba go
   kopiować do `game_data`.
2. Zbuduj obraz:
   ```
   docker compose build
   ```
3. Odpal kontener i wejdź do niego:
   ```
   docker compose up -d
   docker compose exec vcmi bash
   ```
4. W kontenerze sprawdź, że silnik działa i widzi dane gry:
   ```
   build/bin/vcmiserver --dummy-run
   ```

## Aktualny stan / problem

Silnik VCMI (w `/vcmi`) jest zbudowany i zweryfikowany — `--dummy-run` przechodzi bez błędów,
dane gry są widoczne poprawnie. Środowisko Pythona (`vcmi-gym`, w `/vcmi-gym`) jest zainstalowane.

Obecnie w trakcie: budowanie **connectora** — biblioteki C++ (przez Conan + CMake), która łączy
Pythona z silnikiem VCMI. To ostatni krok przed odpaleniem pierwszego treningu.

## Pinowanie wersji VCMI / vcmi-gym

Gałąź `mmai` (fork VCMI używany przez ten projekt) jest aktywnie rozwijana i bywa niestabilna —
dlatego `Dockerfile` nie ściąga jej najnowszego stanu, tylko konkretny, sprawdzony commit.
Jeśli coś się zepsuje po aktualizacji albo chcesz świadomie przejść na nowszą wersję, zmień
wartości na początku `Dockerfile`:

```
ARG VCMI_COMMIT=...     # fork VCMI (branch mmai)
ARG VCMIGYM_COMMIT=...  # repo vcmi-gym (pociąga za sobą też swoją własną wersję VCMI)
```

Po zmianie commitu trzeba przebudować obraz od zera i być może ponownie poprawić błędy
kompilacji opisane jako patche (`sed`) w `Dockerfile` — mogą już nie być potrzebne albo
mogą się pojawić nowe, inne.

Uwaga: gałąź `mmai` bywa force-pushowana (historia jest przepisywana), więc stary, zapamiętany
commit może "wypaść" z gałęzi. Dlatego `Dockerfile` ściąga commit bezpośrednio po SHA
(`git fetch origin <SHA>`), nie przez klonowanie gałęzi — to działa nawet wtedy.
