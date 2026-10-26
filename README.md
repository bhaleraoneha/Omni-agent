# Omni Agent: multi-modal + voice + content pipeline (free Groq API)

A Flask app that extends a tool-using agent with:

| Requirement | Where |
|---|---|
| Multi-modal input (image understanding) | `groq_client.describe_image`, image button in Chat and Full demo |
| Voice interface (speech-to-text + text-to-speech) | Whisper on Groq (`/api/transcribe`) for input, browser speech synthesis for output |
| Automated content pipeline | `pipeline.py`: topic -> outline -> blog -> summary -> social post, one click |
| Agent tool-use + evaluation harness | `agent.py` (calculator, time, text stats, notes), `evals.py` (5 test cases, "Evals" tab) |
| Demo of all three together | "Full demo" tab: image + spoken request -> agent -> content pipeline |

## 1. Get a free Groq key
1. Sign up at https://console.groq.com (free, no card).
2. API Keys -> Create API Key -> copy it.

## 2. Run locally
```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key      # Windows PowerShell: $env:GROQ_API_KEY="your_key"
python app.py                     # open http://localhost:7860
python evals.py                   # optional: run the eval harness in the terminal
```
The microphone works on `localhost` and on any HTTPS site (so it works on the live URL).

## 3. Make it live (free)

### Option A: Render (easiest)
1. Push this folder to a new GitHub repo.
2. https://render.com -> New -> Blueprint (or Web Service) -> pick the repo. `render.yaml` is picked up automatically.
3. Add environment variable `GROQ_API_KEY`.
4. Deploy. You get `https://omni-agent-xxxx.onrender.com`. Free instances sleep after idle time, so open it once before the reviewer does.

### Option B: Hugging Face Spaces (no sleeping worries, Docker)
1. https://huggingface.co/new-space -> SDK: **Docker** -> Blank.
2. Upload all files of this folder (the Dockerfile uses port 7860, which Spaces expects).
3. Settings -> Variables and secrets -> new secret `GROQ_API_KEY`.
4. The Space URL is your live URL.

## 4. Adapting to your existing AI-C2 agent
`agent.py` contains a ready-made tool-calling loop. If your AI-C2 agent from the earlier tasks has its own tools or evals, add them to `TOOLS` / `TOOL_SCHEMAS` in `agent.py` and to `CASES` in `evals.py`; the rest of the app works unchanged.

## Models (change via env vars if Groq retires one)
`GROQ_TEXT_MODEL`, `GROQ_VISION_MODEL`, `GROQ_STT_MODEL`. See https://console.groq.com/docs/models.

## Notes
- Free tier is rate limited; the client retries on HTTP 429 automatically.
- Chat history lives in the browser; notes live in server memory (reset on restart).
