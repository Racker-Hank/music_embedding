# 🎵 Artwork Embedder

Simple tool to find album art for **MP3/WAV** files and embed it into the tracks.

---

## ✨ Features

-  🎧 Music-focused providers: iTunes, Discogs, Juno, Spotify, SoundCloud, Google CSE (configurable order)
-  💾 Caching of chosen artwork (`data/artwork_cache.json`)
-  📂 Copies/moves originals to `Processed/` / `Unprocessed/` depending on result and config
-  ⚙️ Configurable via `conf/config.json` and overrideable via `.env` (recommended for secrets)
-  📝 Logging with configurable level, filename and line number

---

## 📂 Files & Folder Structure

### 📦 Zip Structure

```

music_embedding/
├── 🎶 embed_artwork.exe # exe
├── 🎵 .mp3/.wav files # place audio files here
├── 📂 Processed/ # auto-created output folder (tracks with artwork)
├── 📂 Unprocessed/ # auto-created output folder (tracks without artwork)
├── ⚙️ conf/
│ └── config.json # non-sensitive defaults and preferences
├── 💾 data/
│ └── artwork_cache.json # auto-created cache of accepted artwork
├── 🔑 .env # secrets + environment-specific overrides
└── 📦 \_internal # libraries bundled by PyInstaller

```

### 🛠 Code Structure

```

music_embedding/
├── 🎶 embed_artwork.py # main script
├── ⚙️ conf/
│ └── config.json # non-sensitive defaults and preferences
├── 💾 data/
│ └── artwork_cache.json # auto-created cache of accepted artwork
├── 🔑 .env # secrets + environment-specific overrides
├── 📂 Processed/ # auto-created output folder (tracks with artwork)
├── 📂 Unprocessed/ # auto-created output folder (tracks without artwork)
└── 📘 README.md # usage instructions

```

---

## 🚀 Usage

### ▶️ After unzipping

Place audio files (`.mp3` / `.wav`) in the same folder as the exe, then **run the exe file**.
When prompted to use the artwork or not, input **y** for yes, **n** for no.

What happens:

-  🔍 Script scans folder for `.mp3` / `.wav`
-  🏷 Builds a search query from the filename
-  🌐 Queries providers in configured order
-  🖼 Candidate images are displayed in a preview window
-  👆 You choose **Use this artwork** or **Skip**
-  ✅ Accepted → embedded into a copy in `Processed/`
-  ❌ Failed/no artwork → original copied/moved to `Unprocessed/`

---

## ⚡ Quick start

### 1️⃣ Create and activate conda env

```bash
conda create -n artembed python=3.13 -y
conda activate artembed
```

### 2️⃣ Install dependencies

```bash
python -m pip install --upgrade pip setuptools wheel
pip install mutagen pillow requests tqdm python-dotenv google-api-python-client pyinstaller
```

---

## 🛠 Example `config.json`

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

## 🔑 Example `.env`

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

`.env` overrides `config.json` at runtime.

---

## 🖥 Build with PyInstaller

**Folder mode (recommended)**

```bash
pyinstaller embed_artwork.py
```

**One-file mode**

```bash
pyinstaller --onefile --add-data "conf/config.json;conf" --add-data "data/artwork_cache.json;data" embed_artwork.py
```

---

## 🛡 Troubleshooting

-  ⚠️ No images? → Lower `MIN_RESOLUTION` or set `REQUIRE_SQUARE=false`
-  🔍 Google API vs manual search → API is stricter; enable “Search the entire web” in CSE
-  🏗 PyInstaller one-file → uses `sys.argv[0]` to resolve exe location
-  🛑 AV false positives → one-file exes are large, distribute with checksum

---

## 📌 Recommended workflow

1. 📦 Create conda env + install deps
2. ⚙️ Place `embed_artwork.py` and `conf/config.json` in project folder
3. 🔑 Put API keys in `.env`
4. ▶️ Run `python embed_artwork.py` with a few test MP3s
5. 🖥 Build exe with PyInstaller for distribution

---
