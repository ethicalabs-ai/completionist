import requests


def get_completion(
    prompt: str,
    model_name: str,
    api_url: str,
    system_prompt: str = None,
    hf_api_token: str = None,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> str:
    """
    Sends a prompt to an LLM API endpoint to get a text completion.
    Args:
        prompt: The text prompt to send.
        model_name: The name of the model to use for generation.
        api_url: The URL of the API endpoint.
        system_prompt: An optional system prompt to prepend to the user prompt.
        hf_api_token: An optional Hugging Face API token for authentication.
        max_tokens: The maximum number of tokens to generate.
        temperature: The sampling temperature.
        top_p: The nucleus sampling probability.

    Returns:
        The generated text completion.
    """
    headers = {"Content-Type": "application/json"}

    if (
        "huggingface.cloud" in api_url or "api-inference.huggingface.co" in api_url
    ) and hf_api_token:
        headers["Authorization"] = f"Bearer {hf_api_token}"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
    }

    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        response_json = response.json()
        completion_text = (
            response_json.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        return completion_text
    except requests.exceptions.RequestException as e:
        print(f"Error getting completion for prompt: '{prompt[:50]}...': {e}")
        return ""