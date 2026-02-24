import os
from dotenv import load_dotenv
import os
import base64
from flask import Flask, render_template, request, redirect, url_for
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

SYSTEM_PROMPT = "You are a friendly assistant who does its best to help!"
# Track conversation via previous_response_id (reset on server restart)
last_response_id = None


def safe_responses_call(user_input, previous_response_id=None):
    """Call OpenAI Responses API with error handling."""
    try:
        return client.responses.create(
            model="gpt-5-nano",
            instructions=SYSTEM_PROMPT,
            input=user_input,
            previous_response_id=previous_response_id,
        )
    except (AuthenticationError, RateLimitError, Exception) as e:
        print(f"API Error: {e}")
        return None

@app.route("/", methods=["GET", "POST"])
def index():
    """Handle chat requests."""
    global last_response_id
    reply = ""
    usage_info = ""

    if request.method == "POST":
        user_text = request.form.get("user_input", "")
        image_file = request.files.get("image")

        # Build multimodal input
        content = []
        if user_text:
            content.append({"type": "input_text", "text": user_text})
        if image_file and image_file.filename:
            img_b64 = base64.b64encode(image_file.read()).decode("utf-8")
            mime = image_file.content_type or "image/png"
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:{mime};base64,{img_b64}",
                }
            )

        if content:
            user_msg = [{"role": "user", "content": content}]
            resp = safe_responses_call(user_msg, last_response_id)
            if resp:
                last_response_id = resp.id
                reply = resp.output_text
                usage = resp.usage
                usage_info = (
                    f"Input: {usage.input_tokens}, "
                    f"Output: {usage.output_tokens}, "
                    f"Total: {usage.total_tokens}"
                )

    return render_template("index.html", reply=reply, usage=usage_info)

@app.route("/reset", methods=["GET", "POST"])
def reset():
    """Reset conversation history."""
    global last_response_id
    last_response_id = None
    return redirect(url_for("index"))



# Run with: flask --app hello_flask.py run
