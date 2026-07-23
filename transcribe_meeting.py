def generate_summary(transcript_text: str) -> str:
    """
    Sends the transcript to Cloudflare Workers AI REST API.
    """
    account_id = os.environ.get("CF_ACCOUNT_ID")
    api_token = os.environ.get("CF_API_TOKEN")

    if not account_id or not api_token:
        return "[Error: CF_ACCOUNT_ID or CF_API_TOKEN environment variables not set]"

    # Updated to an active model in Cloudflare's catalog
    model = "@cf/meta/llama-3.2-3b-instruct"
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"

    system_prompt = "You are an AI executive assistant. Summarize meeting transcripts accurately and concisely."
    user_prompt = (
        "Summarize the following meeting transcript.\n\n"
        "Formatting guidelines:\n"
        "1. Executive Summary (2-3 sentences)\n"
        "2. Key Discussion Points (bullet points)\n"
        "3. Decisions Made\n"
        "4. Action Items & Next Steps\n\n"
        f"TRANSCRIPT:\n{transcript_text}"
    )

    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 1024
    }

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_token}",
                "Content-Type": "application/json"
            }
        )
        with urllib.request.urlopen(req, timeout=45) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            if res_data.get("success"):
                return res_data.get("result", {}).get("response", "Error: Empty response.")
            else:
                errors = res_data.get("errors", [])
                return f"[Cloudflare API Error: {errors}]"

    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        return f"[HTTP Error {e.code}: {error_body}]"
    except Exception as e:
        return f"[Error generating summary: {e}]"
