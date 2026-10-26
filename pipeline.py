"""Automated content pipeline: topic -> outline -> draft -> summary + social post."""
import groq_client as g


def _ask(system, user, max_tokens=1400):
    return g.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens, temperature=0.6,
    )["content"].strip()


def run(topic, tone="friendly and clear", source_text=""):
    """One call, zero manual steps. `source_text` can be an image description or transcript."""
    ctx = f"\nUse this extra context:\n{source_text}\n" if source_text else ""
    outline = _ask("You are an editor. Reply with a 4-6 bullet outline only.",
                   f"Topic: {topic}\nTone: {tone}{ctx}", 400)
    blog = _ask("You are a blog writer. Write a ~300 word blog post in markdown with a title.",
                f"Topic: {topic}\nTone: {tone}\nFollow this outline:\n{outline}{ctx}", 1000)
    summary = _ask("Summarise in exactly 2 sentences.", blog, 200)
    social = _ask("Write one engaging social media post under 280 characters with 2-3 hashtags.",
                  f"Topic: {topic}\nBlog:\n{blog}", 200)
    return {"topic": topic, "outline": outline, "blog": blog, "summary": summary, "social": social}
