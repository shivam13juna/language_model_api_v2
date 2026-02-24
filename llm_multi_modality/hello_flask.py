import os
from dotenv import load_dotenv
from flask import Flask, render_template, request
from openai import AuthenticationError, OpenAI, RateLimitError
import truststore

# Configuration
load_dotenv(
    "/Users/shivam13juna/Documents/scaler/iitr_classes/jan_2026/language_model_api_v2/openai_key.env"
)
truststore.inject_into_ssl()

api_key = os.getenv("OPENAI_API_KEY")
app = Flask(__name__)
client = OpenAI(api_key=api_key)

# Shared conversation state (reset on server restart)
messages = [
    {
        "role": "system",
        "content": "You are a friendly Python tutor who explains concepts clearly.",
    }
]


def safe_chat_call(messages):
    """Call OpenAI API with error handling."""
    try:
        return client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    except (AuthenticationError, RateLimitError, Exception):
        return None


@app.route("/", methods=["GET", "POST"])
def index():
    """Handle chat requests."""
    reply = ""
    usage_info = ""

    if request.method == "POST":
        user_text = request.form["user_input"]
        messages.append({"role": "user", "content": user_text})

        resp = safe_chat_call(messages)
        if resp:
            reply = resp.choices[0].message.content
            messages.append({"role": "assistant", "content": reply})
            usage = resp.usage
            usage_info = f"Prompt: {usage.prompt_tokens}, Completion: {usage.completion_tokens}, Total: {usage.total_tokens}"

    return render_template("index.html", reply=reply, usage=usage_info)


# Run with: flask run --app hello_flask.py --debug
