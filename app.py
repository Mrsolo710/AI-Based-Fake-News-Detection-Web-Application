import os
import re
import time
import yt_dlp
import sqlite3
import requests
import json
import wikipedia
import pytesseract
import pdfplumber
import whisper
from docx import Document
from PIL import Image
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs
from google import genai
from google.genai import types

try:
    from ddgs import DDGS
except ImportError:
    from ddgs import DDGS

load_dotenv()

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)
CORS(app)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

print("Loading local Whisper model...")
whisper_model = whisper.load_model("base")
print("Whisper model loaded successfully!")

_CACHED_MODELS = []


def get_available_flash_models(client):
    """Queries Google's ListModels API and filters general-purpose Flash models."""
    global _CACHED_MODELS
    if _CACHED_MODELS:
        return _CACHED_MODELS

    preferred_order = [
        "gemini-3-flash-preview",
        "gemini-3.5-flash",
        "gemini-3.1-flash-lite-preview",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
    ]

    excluded_keywords = ("tts", "image", "audio", "embedding", "aqa", "imagen", "veo")

    discovered = []
    try:
        for m in client.models.list():
            name = getattr(m, "name", "") or ""
            short_name = name.replace("models/", "")
            lower_name = short_name.lower()

            if "flash" not in lower_name:
                continue
            if any(bad in lower_name for bad in excluded_keywords):
                continue

            methods = getattr(m, "supported_generation_methods", None) or getattr(m, "supported_actions", None) or []
            if not methods or "generateContent" in methods:
                discovered.append(short_name)
    except Exception as e:
        print("ListModels warning:", e)

    ordered = [m for m in preferred_order if m in discovered]
    for m in discovered:
        if m not in ordered:
            ordered.append(m)

    _CACHED_MODELS = ordered if ordered else preferred_order
    print("Active Gemini models for your key:", _CACHED_MODELS)
    return _CACHED_MODELS


def generate_with_fallback(client, contents, config=None):
    """Tries available models in order, automatically falling back on 404 or 429 limits."""
    models = get_available_flash_models(client)
    last_error = None

    for model_name in models:
        try:
            if config:
                return client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config
                )
            else:
                return client.models.generate_content(
                    model=model_name,
                    contents=contents
                )
        except Exception as e:
            err_str = str(e)
            last_error = e
            if (
                "404" in err_str
                or "NOT_FOUND" in err_str
                or "429" in err_str
                or "RESOURCE_EXHAUSTED" in err_str
                or "not supported for generateContent" in err_str
            ):
                print(f"Skipping {model_name} ({err_str[:75]}...), trying next model...")
                continue
            raise e

    raise last_error


def search_live_web_news(query, max_results=6):
    """Fetches real-time news & web sources for free using DuckDuckGo (no API key or billing needed)."""
    results = []
    seen_urls = set()

    try:
        with DDGS() as ddgs:
            # 1. Search recent news articles first
            try:
                news_hits = ddgs.news(query, max_results=max_results)
                for item in news_hits:
                    url = item.get("url") or item.get("href") or ""
                    if url.startswith("http") and url not in seen_urls:
                        seen_urls.add(url)
                        results.append({
                            "title": item.get("title") or "News Article",
                            "url": url,
                            "snippet": item.get("body") or item.get("excerpt") or "",
                            "source": item.get("source") or ""
                        })
            except Exception as e:
                print("DDGS news warning:", e)

            # 2. Supplement with general web search if fewer than 4 news results found
            if len(results) < 4:
                try:
                    web_hits = ddgs.text(query, max_results=max_results)
                    for item in web_hits:
                        url = item.get("href") or item.get("url") or ""
                        if url.startswith("http") and url not in seen_urls:
                            seen_urls.add(url)
                            results.append({
                                "title": item.get("title") or "Web Source",
                                "url": url,
                                "snippet": item.get("body") or "",
                                "source": ""
                            })
                except Exception as e:
                    print("DDGS web warning:", e)
    except Exception as e:
        print("Live web search error:", e)

    return results[:max_results]


def get_db_connection():
    conn = sqlite3.connect('newsverify.db')
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS verifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_type TEXT,
            original_input TEXT,
            extracted_claim TEXT,
            assessment TEXT,
            confidence INTEGER,
            summary TEXT,
            explanation TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS verification_sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            verification_id INTEGER,
            url TEXT,
            title TEXT,
            domain TEXT,
            snippet TEXT,
            stance TEXT,
            FOREIGN KEY (verification_id) REFERENCES verifications (id)
        )
    ''')
    conn.commit()
    conn.close()


init_db()


def extract_headline_from_url(url):
    """Extracts a readable headline from the URL slug as a fallback."""
    path = urlparse(url).path
    segments = [s for s in path.split('/') if s and not s.startswith('ar-') and not s.isdigit()]
    if segments:
        slug = max(segments, key=len)
        slug = re.sub(r'-\d+$', '', slug)
        cleaned = re.sub(r'[-_]+', ' ', slug).strip()
        if len(cleaned.split()) >= 4:
            return cleaned.title()
    return ""


def scrape_article(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9'
    }
    title = ""
    article_text = ""

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        og_title = soup.find('meta', property='og:title')
        if og_title and og_title.get('content'):
            title = og_title['content'].strip()
        elif soup.title and soup.title.string:
            title = soup.title.string.strip()

        paragraphs = soup.find_all('p')
        article_text = " ".join([p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 30])
    except Exception as e:
        print(f"Scraping blocked or failed for {url}: {e}")

    generic_titles = {"", "msn", "just a moment...", "access denied", "no title found"}
    if title.lower() in generic_titles or len(article_text) < 100:
        url_headline = extract_headline_from_url(url)
        if url_headline:
            title = url_headline

    return title, article_text[:2000]


def extract_clean_claim(raw_text):
    """Strips browser tabs, sidebar ads, and OCR noise to isolate the main news claim."""
    client = genai.Client(api_key=GEMINI_API_KEY)
    cleanup_prompt = f"""
Extract ONLY the single main news headline and core factual claim from the raw text below.
Ignore browser bookmarks, navigation menus, sidebar news widgets, unrelated headlines, and OCR garbled characters.
Fix obvious OCR typos if clear from context (e.g., "0-kilometer" -> "10-kilometer").
Return ONLY a clean 1-to-2 sentence factual summary of the main article/claim. Do not include any intro text.

Raw Text:
{raw_text[:2500]}
"""
    try:
        res = generate_with_fallback(
            client,
            cleanup_prompt,
            types.GenerateContentConfig(temperature=0.0)
        )
        cleaned = (res.text or "").strip()
        return cleaned if cleaned else raw_text[:400]
    except Exception:
        return raw_text[:400]


def extract_text_from_response(response):
    """Safely pulls text from response.text or candidate parts."""
    raw_text = response.text or ""
    if not raw_text and getattr(response, "candidates", None):
        for candidate in response.candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) or []
            raw_text += "".join(part.text for part in parts if getattr(part, "text", None))
    return raw_text.strip()


def parse_safe_json_dict(raw_text):
    """Strips markdown fences and safely parses JSON into a Python dict."""
    cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw_text.strip(), flags=re.IGNORECASE)
    match = re.search(r'\{.*\}', cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)

    try:
        parsed = json.loads(cleaned)
    except Exception:
        return None

    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
        parsed = parsed[0]

    if not isinstance(parsed, dict):
        return None

    return parsed


def analyze_claim_with_ai(clean_claim, extra_context=""):
    if not GEMINI_API_KEY:
        raise Exception("Missing Gemini API Key in .env file")

    # Fetch live web/news evidence via free DuckDuckGo search (avoids 0-quota Search Grounding 429)
    live_results = search_live_web_news(clean_claim[:180], max_results=6)

    evidence_block = ""
    for i, item in enumerate(live_results, start=1):
        evidence_block += (
            f"[{i}] Title: {item['title']}\n"
            f"    Snippet: {item['snippet']}\n"
            f"    URL: {item['url']}\n\n"
        )

    if not evidence_block:
        evidence_block = "No live search articles retrieved."

    prompt = f"""
You are an expert real-time fact-checking AI.
Analyze the claim below using the Live News Search Evidence and Reference Context provided.

News Claim to Verify:
"{clean_claim}"

Live News Search Evidence:
{evidence_block}

Additional Reference Context:
{extra_context if extra_context else "None"}

Instructions:
1. Compare the claim carefully against the Live News Search Evidence above.
2. Return ONLY a valid JSON object (no markdown formatting, no backticks) with these exact keys:
{{
  "assessment": "THIS IS 100 PERCENT TRUE NEWS | LIKELY_FALSE | MISLEADING | UNVERIFIED | INSUFFICIENT_EVIDENCE",
  "confidence": 95,
  "summary": "Short 1-2 sentence executive summary of the verdict.",
  "explanation": "Detailed breakdown of what the live news sources report.",
  "evidence_stances": ["SUPPORTS", "CONTRADICTS", "NEUTRAL", "UNCLEAR"]
}}
3. "evidence_stances" must have one stance string per numbered evidence item [1..{len(live_results)}] above.
4. Use "THIS IS 100 PERCENT TRUE NEWS" when reputable news sources confirm the story is real and accurate.
"""

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        # Plain text generation without tools=[{"google_search": {}}] works on the Free Tier!
        config = types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=2048,
        )

        response = generate_with_fallback(client, prompt, config)
        raw_text = extract_text_from_response(response)

        if not raw_text:
            return {
                "assessment": "UNVERIFIED",
                "confidence": 0,
                "summary": "The AI did not return a text response for this input.",
                "explanation": "Empty response body returned by the model.",
                "final_sources": []
            }

        ai_data = parse_safe_json_dict(raw_text)
        if not ai_data:
            return {
                "assessment": "UNVERIFIED",
                "confidence": 0,
                "summary": "Failed to parse structured verification output.",
                "explanation": raw_text[:1500],
                "final_sources": []
            }

        verdict = str(ai_data.get("assessment", "UNVERIFIED")).strip().upper()
        ai_data["assessment"] = verdict

        if verdict == "THIS IS 100 PERCENT TRUE NEWS":
            default_stance = "SUPPORTS"
        elif verdict in {"LIKELY_FALSE", "MISLEADING"}:
            default_stance = "CONTRADICTS"
        else:
            default_stance = "NEUTRAL"

        stances = ai_data.get("evidence_stances") or []
        final_sources = []

        for idx, item in enumerate(live_results):
            if idx < len(stances) and isinstance(stances[idx], str):
                stance = stances[idx].strip().upper()
                if stance not in {"SUPPORTS", "CONTRADICTS", "NEUTRAL", "UNCLEAR"}:
                    stance = default_stance
            else:
                stance = default_stance

            url = item["url"]
            domain = urlparse(url).netloc.replace("www.", "") if "://" in url else url
            final_sources.append({
                "title": item["title"],
                "url": url,
                "domain": domain,
                "snippet": item["snippet"],
                "stance": stance
            })

        ai_data["final_sources"] = final_sources
        return ai_data

    except Exception as e:
        raise Exception(f"Gemini AI Error: {str(e)}")


def analyze_media_with_gemini(file_path):
    """Uploads video/audio file to Gemini so it can watch visual text/captions AND listen to audio."""
    client = genai.Client(api_key=GEMINI_API_KEY)
    uploaded_media = client.files.upload(file=file_path)

    try:
        while uploaded_media.state.name == "PROCESSING":
            time.sleep(2)
            uploaded_media = client.files.get(name=uploaded_media.name)

        media_prompt = (
            "Watch and listen to this video/audio carefully. "
            "Transcribe all spoken words AND read any on-screen news headlines, captions, or text banners. "
            "Summarize the exact news claim being made."
        )
        media_res = generate_with_fallback(client, [uploaded_media, media_prompt])
        return extract_text_from_response(media_res)
    finally:
        try:
            client.files.delete(name=uploaded_media.name)
        except Exception:
            pass


def normalize_youtube_url(url):
    """Converts youtu.be or YouTube Shorts links into standard watch?v= URLs."""
    parsed = urlparse(url)
    if "youtu.be" in parsed.netloc:
        video_id = parsed.path.lstrip('/').split('/')[0]
        return f"https://www.youtube.com/watch?v={video_id}"
    if "/shorts/" in parsed.path:
        video_id = parsed.path.split('/shorts/')[1].split('/')[0]
        return f"https://www.youtube.com/watch?v={video_id}"
    if "watch" in parsed.path:
        video_id = parse_qs(parsed.query).get("v", [None])[0]
        if video_id:
            return f"https://www.youtube.com/watch?v={video_id}"
    return url


def get_youtube_summary_direct(url):
    """Passes YouTube/Shorts URL directly to Gemini with caption/title fallbacks."""
    clean_url = normalize_youtube_url(url)

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = generate_with_fallback(
            client,
            [
                types.Part.from_uri(file_uri=clean_url, mime_type="video/webm"),
                (
                    "Watch and listen to this YouTube video carefully. "
                    "Extract the main news claim, spoken statements, and any on-screen headlines or captions. "
                    "Provide a clear, detailed summary of what this video claims."
                )
            ]
        )
        text = extract_text_from_response(response)
        if text:
            return text
    except Exception as e:
        print("Gemini direct YouTube URL error:", e)

    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        parsed = urlparse(clean_url)
        video_id = parse_qs(parsed.query).get("v", [None])[0]
        if video_id:
            ytt_api = YouTubeTranscriptApi()
            fetched = ytt_api.fetch(video_id, languages=['en', 'hi'])
            transcript_text = " ".join(snippet.text for snippet in fetched)
            if transcript_text.strip():
                return transcript_text.strip()
    except Exception as e:
        print("YouTubeTranscriptApi fallback error:", e)

    try:
        title, desc = scrape_article(clean_url)
        if title and title.lower() not in {"youtube", "youtube\n"}:
            return f"YouTube Video Title: {title}. {desc}"
    except Exception:
        pass

    return None


def get_instagram_audio_transcript(url):
    """Downloads Instagram Reel audio via yt-dlp and transcribes it."""
    temp_filename = "temp_ig_audio"
    audio_path = f"{temp_filename}.mp3"

    if os.path.exists(audio_path):
        os.remove(audio_path)

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': temp_filename,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
        'no_warnings': True
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        if os.path.exists(audio_path):
            result = whisper_model.transcribe(audio_path, fp16=False, task="translate")
            text = result.get("text", "").strip()
            if not text:
                text = analyze_media_with_gemini(audio_path)
            if os.path.exists(audio_path):
                os.remove(audio_path)
            return text
    except Exception as e:
        print("Instagram Audio Error:", e)
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return None
    return None


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api/verify', methods=['POST'])
def verify():
    input_type = 'text'
    raw_extracted_content = ''
    original_input = ''

    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        filename = file.filename.lower()
        extracted_text = ""
        temp_path = os.path.join("temp_" + file.filename)

        try:
            if filename.endswith('.pdf'):
                input_type = 'pdf'
                with pdfplumber.open(file) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            extracted_text += text + "\n"

            elif filename.endswith('.docx'):
                input_type = 'docx'
                doc = Document(file)
                for para in doc.paragraphs:
                    if para.text:
                        extracted_text += para.text + "\n"

            elif filename.endswith('.txt'):
                input_type = 'txt'
                extracted_text = file.read().decode('utf-8', errors='ignore')

            elif filename.endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
                input_type = 'image'
                img = Image.open(file)
                extracted_text = pytesseract.image_to_string(img, lang='hin+eng')

            elif filename.endswith(('.mp3', '.wav', '.m4a', '.mp4', '.mov', '.webm')):
                input_type = 'audio_video'
                file.save(temp_path)

                try:
                    result = whisper_model.transcribe(temp_path, fp16=False, task="translate")
                    extracted_text = result.get("text", "").strip()
                except Exception as whisper_err:
                    print("Whisper fallback triggered:", whisper_err)
                    extracted_text = ""

                if not extracted_text:
                    extracted_text = analyze_media_with_gemini(temp_path)

            else:
                return jsonify({"error": "Unsupported file format."}), 400

            extracted_text = extracted_text.strip()
            if not extracted_text:
                return jsonify({"error": "Could not extract any readable text or speech from this file."}), 400

            original_input = f"[{filename.upper()} Content/Transcript]\n{extracted_text[:1500]}..."
            raw_extracted_content = extracted_text

        except Exception as e:
            return jsonify({"error": f"Failed to process file: {str(e)}"}), 500
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    else:
        text_input = request.form.get('text', '').strip()

        if not text_input and request.is_json:
            text_input = request.json.get('text', '').strip()

        if not text_input:
            return jsonify({"error": "Please enter a claim, a URL, or upload a file."}), 400

        original_input = text_input
        raw_extracted_content = text_input

        if text_input.startswith('http://') or text_input.startswith('https://'):
            try:
                input_type = 'url'

                if "youtube.com" in text_input or "youtu.be" in text_input:
                    yt_summary = get_youtube_summary_direct(text_input)
                    if yt_summary:
                        original_input = text_input
                        raw_extracted_content = yt_summary
                    else:
                        return jsonify({"error": "Could not read this YouTube video. Make sure the video is public and not age-restricted."}), 400

                elif "instagram.com" in text_input:
                    media_transcript = get_instagram_audio_transcript(text_input)
                    if media_transcript:
                        original_input = f"[INSTAGRAM VIDEO TRANSCRIPT]\n{media_transcript[:1500]}..."
                        raw_extracted_content = media_transcript
                    else:
                        return jsonify({"error": "Failed to download audio from this Instagram link. It might be private or login-protected."}), 400

                else:
                    title, article_text = scrape_article(text_input)
                    if not title and not article_text:
                        return jsonify({"error": "Could not extract headline or content from this URL."}), 400

                    original_input = text_input
                    raw_extracted_content = f"Headline: {title}\n\nArticle Content: {article_text}" if article_text else title

            except Exception as e:
                return jsonify({"error": f"Failed to process URL: {str(e)}"}), 400

    if len(raw_extracted_content) > 350:
        clean_claim = extract_clean_claim(raw_extracted_content)
    else:
        clean_claim = raw_extracted_content.strip()

    wiki_context = ""
    wiki_source = None
    try:
        wiki_search = wikipedia.search(clean_claim[:150])
        if wiki_search:
            wiki_page = wikipedia.page(wiki_search[0], auto_suggest=False)
            wiki_context = f"Wikipedia ({wiki_page.title}): {wiki_page.summary[:600]}"
            wiki_source = {
                "title": f"Wikipedia: {wiki_page.title}",
                "url": wiki_page.url,
                "domain": "en.wikipedia.org",
                "snippet": wiki_page.summary[:600] + "...",
                "stance": "NEUTRAL"
            }
    except Exception:
        pass

    try:
        ai_result = analyze_claim_with_ai(clean_claim, wiki_context)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    sources = ai_result.get("final_sources", [])
    if wiki_source:
        sources.append(wiki_source)

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO verifications (input_type, original_input, extracted_claim, assessment, confidence, summary, explanation)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        input_type,
        original_input,
        clean_claim,
        ai_result.get('assessment', 'UNVERIFIED'),
        ai_result.get('confidence', 0),
        ai_result.get('summary', ''),
        ai_result.get('explanation', '')
    ))

    verification_id = cursor.lastrowid
    saved_sources = []

    for src in sources:
        if not isinstance(src, dict):
            continue
        url = src.get('url', '')
        title = src.get('title') or 'Verified Source'
        domain = src.get('domain') or (url.split('/')[2] if '://' in url and len(url.split('/')) > 2 else url)
        stance = src.get('stance', 'NEUTRAL')

        cursor.execute('''
            INSERT INTO verification_sources (verification_id, url, title, domain, snippet, stance)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (verification_id, url, title, domain, src.get('snippet', ''), stance))

        saved_sources.append({
            "title": title,
            "url": url,
            "domain": domain,
            "stance": stance
        })

    conn.commit()
    conn.close()

    return jsonify({
        "id": verification_id,
        "claim": clean_claim,
        "assessment": ai_result.get('assessment', 'UNVERIFIED'),
        "confidence": ai_result.get('confidence', 0),
        "summary": ai_result.get('summary', ''),
        "explanation": ai_result.get('explanation', ''),
        "sources": saved_sources
    })


@app.route('/history')
def history_page():
    return render_template('history.html')


@app.route('/api/history', methods=['GET'])
def get_history():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, original_input, assessment, confidence FROM verifications ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()

    return jsonify([{"id": row['id'], "claim": row['original_input'], "assessment": row['assessment'], "confidence": row['confidence']} for row in rows])


@app.route('/api/history/<int:item_id>', methods=['GET'])
def get_history_detail(item_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM verifications WHERE id = ?", (item_id,))
    record = cursor.fetchone()
    if not record:
        conn.close()
        return jsonify({"error": "Not found"}), 404

    cursor.execute("SELECT * FROM verification_sources WHERE verification_id = ?", (item_id,))
    sources = cursor.fetchall()
    conn.close()

    return jsonify({
        "id": record['id'],
        "claim": record['original_input'],
        "assessment": record['assessment'],
        "confidence": record['confidence'],
        "summary": record['summary'],
        "explanation": record['explanation'],
        "sources": [dict(s) for s in sources]
    })


@app.route('/api/history/<int:item_id>', methods=['DELETE'])
def delete_history(item_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM verification_sources WHERE verification_id = ?", (item_id,))
    cursor.execute("DELETE FROM verifications WHERE id = ?", (item_id,))
    conn.commit()
    conn.close()
    return jsonify({"status": "success"})


if __name__ == '__main__':
    app.run(debug=True, port=5000)