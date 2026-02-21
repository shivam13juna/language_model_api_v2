
## 1) Create an API key (Dashboard)

1. Sign in to the **OpenAI API Dashboard**.
2. Go to **Settings** → **(Organization or Project)** → **API keys**.
3. Click **Create new secret key**.
4. Copy it somewhere safe **immediately** (treat it like a password). ([OpenAI Developers][1])

Security note: don’t ship API keys in browser/mobile apps; load them server-side via env vars or a secrets manager. ([OpenAI Developers][2])

> Don’t confuse this with **Admin API keys** (those are only for org admin endpoints and won’t work for normal model calls). ([OpenAI Developers][3])

## 2) Put the key in an environment variable

macOS/Linux:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

Windows (PowerShell):

```powershell
setx OPENAI_API_KEY "your_api_key_here"
```

The official SDKs will read `OPENAI_API_KEY` automatically. ([OpenAI Developers][1])

## 3) Call `gpt-5-nano` (Python, Responses API)

Install:

```bash
pip install openai
```

Example (`example.py`):

```python
from openai import OpenAI

client = OpenAI()

resp = client.responses.create(
    model="gpt-5-nano",
    input="Summarize: LLMs are neural networks trained on text."
)

print(resp.output_text)
```

This is the same pattern as the official quickstart (just swapping the model). ([OpenAI Developers][1])

`gpt-5-nano` is the “fastest, most cost-efficient” GPT-5 variant, and it’s available as the alias **`gpt-5-nano`** (you can also pin a snapshot like `gpt-5-nano-2025-08-07` for stability). ([OpenAI Developers][4])

OpenAI recommends using the **Responses API** with GPT-5 family models. ([OpenAI Developers][5])

## 4) Quick test with curl

```bash
curl https://api.openai.com/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-5-nano",
    "input": "Say hello in one sentence."
  }'
```

Auth is standard **Bearer**. ([OpenAI Developers][2])

If you’re in multiple orgs/projects, you can specify headers:

```bash
-H "OpenAI-Organization: $ORGANIZATION_ID" \
-H "OpenAI-Project: $PROJECT_ID"
```

([OpenAI Developers][2])

## If you get quota / 429 errors

You may need to set up billing / prepaid credits for API usage. ([help.openai.com][6])

[1]: https://developers.openai.com/api/docs/quickstart/ "Developer quickstart | OpenAI API"
[2]: https://developers.openai.com/api/reference/overview/ "API Overview | OpenAI API Reference"
[3]: https://developers.openai.com/api/reference/administration/overview/ "Administration Overview | OpenAI API Reference"
[4]: https://developers.openai.com/api/docs/models/gpt-5-nano "GPT-5 nano Model | OpenAI API"
[5]: https://developers.openai.com/cookbook/examples/gpt-5/gpt-5_new_params_and_tools/ "GPT-5 New Params and Tools"
[6]: https://help.openai.com/en/articles/8264644-how-can-i-set-up-prepaid-billing?utm_source=chatgpt.com "How can I set up prepaid billing?"
