# Artwork Embedder

Simple tool to find album art for MP3/WAV files and embed it into the tracks.

**Features**

-  Music-focused providers: iTunes, Discogs, Juno, Spotify, SoundCloud, Google CSE (configurable order)
-  Caching of chosen artwork (`data/artwork_cache.json`)
-  Copies/moves originals to `Processed/` / `Unprocessed/` depending on result and config
-  Configurable via `conf/config.json` and overrideable via `.env` (recommended for secrets)
-  Logging with configurable level, filename and line number

---

## Files & folder structure

### Zip structure

```
music_embedding/
├── embed_artwork.exe         # exe
├── .mp3/.wav files           # place audio files here
├── Processed/                # auto-created output folder (tracks with artwork)
├── Unprocessed/              # auto-created output folder (tracks without artwork)
├── conf/
│   └── config.json           # non-sensitive defaults and preferences
├── data/
│   └── artwork_cache.json    # auto-created cache of accepted artwork
├── .env                      # secrets + environment-specific overrides
└── _internal                 # libraries bundled by PyInstaller

```

### Code structure

```
music_embedding/
├── embed_artwork.py          # main script
├── conf/
│   └── config.json           # non-sensitive defaults and preferences
├── data/
│   └── artwork_cache.json    # auto-created cache of accepted artwork
├── .env                      # secrets + environment-specific overrides
├── Processed/                # auto-created output folder (tracks with artwork)
├── Unprocessed/              # auto-created output folder (tracks without artwork)
└── README.md                 # usage instructions

```

---

## Usage

### After unzipping

**Place audio files (MP3/WAV) in the same folder as the exe, then run exe file. When prompted to use the artwork or not, input y for yes, n for no.**

What happens:

-  Script searches the folder for `.mp3` and `.wav`.
-  For each track it builds a search query from the filename and queries providers in configured order.
-  Candidate images are fetched (in parallel) and displayed in a photo window.
-  **You input **y**(yes) or **n**(no) for each candidate.**
-  If accepted, the script embeds the image into a copy in `Processed/`.
-  If embedding fails or no artwork approved, the original is copied/moved to `Unprocessed/` per `FILE_ACTION`.

---

## Quick start (recommended)

### 1) Create and activate conda env

Use Python 3.13:

```bash
conda create -n artembed python=3.13 -y
conda activate artembed
```

### 2) Install dependencies

```bash
python -m pip install --upgrade pip setuptools wheel
pip install mutagen pillow requests tqdm python-dotenv google-api-python-client pyinstaller
```

> If you don’t plan to use Google provider you can omit `google-api-python-client`.
> `python-dotenv` lets the script read `.env` for secrets.

---

## Example `config.json`

Save next to `embed_artwork.py`. This file is safe for version control (do **not** put secrets here).

```json
{
	"LOG_LEVEL": "INFO",
	"ENABLED_PROVIDERS": [
		"juno",
		"discogs",
		"itunes",
		"spotify",
		"soundcloud",
		"google"
	],
	"MIN_RESOLUTION": 500,
	"REQUIRE_SQUARE": true,
	"SKIP_EXISTING": true,
	"PROCESSED_DIR": "Processed",
	"UNPROCESSED_DIR": "Unprocessed",
	"FILE_ACTION": "copy",
	"GOOGLE_API_KEY": "",
	"GOOGLE_CX_ID": "",
	"DISCOGS_TOKEN": "",
	"SPOTIFY_CLIENT_ID": "",
	"SPOTIFY_CLIENT_SECRET": "",
	"SOUNDCLOUD_CLIENT_ID": ""
}
```

---

## Example `.env`

Put secrets and environment-specific overrides here — **do not commit** this file.

```
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CX_ID=your_google_cx_id
DISCOGS_TOKEN=your_discogs_token
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
SOUNDCLOUD_CLIENT_ID=your_soundcloud_client_id

LOG_LEVEL=DEBUG
FILE_ACTION=copy
MIN_RESOLUTION=1024
```

> `.env` values override `config.json` at runtime.

---

## Usage

### Run as script (recommended while developing)

**Place audio files (MP3/WAV) in the same folder or run the script from the folder that contains the files:**

```bash
python embed_artwork.py
```

What happens:

-  Script searches the folder for `.mp3` and `.wav`.
-  For each track it builds a search query from the filename and queries providers in configured order.
-  Candidate images are fetched (in parallel) and displayed in a Tkinter preview window.
-  You click **Use this artwork** or **Skip** for each candidate.
-  If accepted, the script embeds the image into a copy in `Processed/`.
-  If embedding fails or no artwork approved, the original is copied/moved to `Unprocessed/` per `FILE_ACTION`.

---

## Build a distributable with PyInstaller

**Folder mode (recommended)** — faster startup, simpler path handling:

```bash
pyinstaller embed_artwork.py
# produced executable under dist/embed_artwork/embed_artwork.exe (and supporting files in that folder)
```

**One-file mode** — single `.exe` for distribution (slower startup because it unpacks to a temp folder):

```bash
pyinstaller --onefile --add-data "config.json;." --add-data "artwork_cache.json;." embed_artwork.py
```

-  On macOS / Linux use `:` instead of `;` in `--add-data`.
-  For `--onefile` the script uses `sys.argv[0]` to determine the exe location so Processed/Unprocessed will be next to your exe.

**PyInstaller tips**

-  If the Google client raises `ModuleNotFoundError`, add `--hidden-import googleapiclient.discovery` etc.
-  Keep an external `config.json` next to the exe if you want to change settings without rebuilding.

---

## Config options (high level)

-  `ENABLED_PROVIDERS`: provider order matters — script tries them in this order.
-  `MIN_RESOLUTION`: minimum width/height in px (default `1024`).
-  `REQUIRE_SQUARE`: if `true`, prefer near-square images (tolerance applied).
-  `SKIP_EXISTING`: if `true`, files already present in `Processed/` or `Unprocessed/` are skipped.
-  `FILE_ACTION`: `"copy"` (default) keeps original and copies processed/unprocessed outputs; `"move"` moves originals to `Processed/originals/` or `Unprocessed/`.
-  `GOOGLE_IMG_SIZE`: one of `ICON, SMALL, MEDIUM, LARGE, XLARGE, XXLARGE, HUGE` — Google API requires uppercase.
-  `LOG_LEVEL`: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.

---

## Behavior details

-  **iTunes**: script replaces `100x100bb` URL with `MIN_RESOLUTION x MIN_RESOLUTION` (Apple might not have every resolution — 404s are handled).
-  **Candidate filtering**: images are filtered by `MIN_RESOLUTION` and `REQUIRE_SQUARE`. If you set `MIN_RESOLUTION` too high you may get zero candidates — lower it to see more.
-  **Cache**: accepted artwork URLs are stored in `artwork_cache.json`. Delete to force re-search.
-  **Preview**: Tkinter GUI preview closes after you choose; no external viewer left open.
-  **Replace artwork**: old artwork frames (APIC) are removed before adding the new one to avoid duplicates.
-  **Copy-first workflow**: the original is copied into `Processed/` first, embed is attempted on that copy; on failure processed copy is removed and original transferred to `Unprocessed/` as configured.

---

## Troubleshooting

-  **No images shown**: lower `MIN_RESOLUTION` or set `REQUIRE_SQUARE = false`.
-  **Google API returns different results than manual search**:

   -  Ensure your CSE is set to "Search the entire web" if you want broader results.
   -  Google API results are more restrictive than browser images — consider using Bing Image Search API for different results.

-  **PyInstaller one-file temp folder**:

   -  Use `sys.argv[0]` to derive exe location when using `--onefile`. Folder mode avoids this complexity.

-  **AV false positives**:

   -  One-file exes can be large and sometimes flagged by AV tools. Test on trusted machines and distribute with checksum.

---

## Recommended workflow

1. Create the conda env and install deps.
2. Put `embed_artwork.py` and `conf/config.json` in a folder.
3. Put API keys in `.env` (beside script).
4. Test with `python embed_artwork.py` in a folder with a few MP3s.
5. Once happy, build with PyInstaller if you need to share.

---

## Example commands (Windows PowerShell)

```powershell
conda create -n artembed python=3.13 -y
conda activate artembed
pip install mutagen pillow requests tqdm python-dotenv google-api-python-client pyinstaller

cd K:\0work\pet\music_embedding
python embed_artwork.py

# or build exe (folder mode)
pyinstaller embed_artwork.py
# run dist\embed_artwork\embed_artwork.exe
```

---
