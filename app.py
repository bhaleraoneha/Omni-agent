import os

from flask import Flask, jsonify, request, send_from_directory

import agent
import evals
import groq_client as g
import pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024  # 12 MB uploads


def api(fn):
    def wrapper():
        try:
            return jsonify(fn())
        except g.GroqError as e:
            return jsonify({"error": str(e)}), 502
        except Exception as e:  # keep the demo friendly
            return jsonify({"error": f"{type(e).__name__}: {e}"}), 500
    wrapper.__name__ = fn.__name__
    return wrapper


@app.get("/")
def index():
        return send_from_directory(os.path.join(BASE_DIR, "public"), "index.html")


@app.get("/health")
def health():
    return {"ok": True, "key_configured": bool(os.getenv("GROQ_API_KEY"))}


@app.get("/api/models")
@api
def models():
    """Shows which Groq models were auto-selected (handy for debugging)."""
    return {k: g.model_for(k) for k in ("text", "vision", "stt")} | {"available": g.available_models()}


@app.post("/api/chat")
@api
def chat():
    d = request.get_json(force=True)
    return agent.run(d.get("messages", []), image_data_url=d.get("image"))


@app.post("/api/transcribe")
@api
def transcribe():
    f = request.files.get("audio")
    if not f:
        raise ValueError("no audio uploaded")
    data = f.read()
    if len(data) < 800:
        raise ValueError("audio too short - hold the mic a little longer")
    return {"text": g.transcribe(f.filename or "audio.webm", data, f.mimetype or "audio/webm")}


@app.post("/api/pipeline")
@api
def run_pipeline():
    d = request.get_json(force=True)
    topic = (d.get("topic") or "").strip()
    if not topic:
        raise ValueError("topic is required")
    return pipeline.run(topic, d.get("tone") or "friendly and clear", d.get("context") or "")


@app.post("/api/demo")
@api
def demo():
    """All three capabilities in one flow: image (+ optional voice transcript) -> agent -> pipeline."""
    d = request.get_json(force=True)
    image, spoken = d.get("image"), (d.get("spoken") or "").strip()
    if not image:
        raise ValueError("image is required")
    ask = spoken or "What is in this image? Give me a content idea based on it."
    desc = g.describe_image(image, "Describe this image in detail for a content writer.")
    reply = agent.run([{"role": "user", "content": ask + "\n\n[IMAGE DESCRIPTION]\n" + desc}])
    content = pipeline.run(spoken or "the subject of this image", "engaging", desc)
    return {"description": desc, "agent": reply, "content": content}


@app.post("/api/evals")
@api
def run_evals():
    return evals.run_all()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 7860)), debug=False)
