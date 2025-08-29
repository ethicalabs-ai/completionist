import json
import traceback
from typing import Optional, Union

from pydantic import BaseModel
from openai import OpenAI as OpenAIClient
import outlines


def process_json(raw_response: str):
    if raw_response.strip().startswith("```json"):
        cleaned_response = (
            raw_response.strip().removeprefix("```json").removesuffix("```")
        )
    else:
        cleaned_response = raw_response

    return json.loads(cleaned_response)


def get_completion(
    prompt: str,
    model_name: str,
    api_url: str,
    system_prompt: Optional[str] = None,
    hf_api_token: Optional[str] = None,
    openai_api_token: Optional[str] = None,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.95,
    pydantic_schema: Optional[BaseModel] = None,
) -> Union[str, BaseModel]:
    """
    Sends a prompt to an LLM API to get a completion.

    If a Pydantic schema is provided, it uses the 'outlines' library for
    structured, schema-enforced generation. Otherwise, it performs a
    standard text completion request.

    Args:
        prompt: The text prompt to send.
        model_name: The name of the model to use for generation.
        api_url: The URL of the API endpoint.
        system_prompt: An optional system prompt.
        hf_api_token: An optional Hugging Face API token.
        max_tokens: The maximum number of tokens to generate.
        temperature: The sampling temperature.
        top_p: The nucleus sampling probability.
        pydantic_schema: An optional Pydantic BaseModel to enforce structured output.

    Returns:
        If a schema is provided, returns a Pydantic object.
        Otherwise, returns the generated text as a string.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    api_token = openai_api_token
    is_hf_url = (
        "huggingface.cloud" in api_url or "api-inference.huggingface.co" in api_url
    )
    if is_hf_url:
        if not hf_api_token:
            raise TypeError(
                "An hugging face token is required to perform this request."
            )
        else:
            api_token = hf_api_token

    try:
        # For outlines, the client needs the base URL, not the full endpoint
        client = OpenAIClient(base_url=api_url, api_key=api_token or "dummy")

        if pydantic_schema:
            generator = outlines.models.openai.OpenAI(
                client=client, model_name=model_name
            )
            chat_prompt = outlines.inputs.Chat(messages)
            result = generator(
                chat_prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            raw_json = process_json(result)
            return pydantic_schema(**raw_json)
        else:
            result = client.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            return result.choices[0].text.strip()
    except Exception:
        print(
            f"Error during structured generation for prompt: '{prompt[:50]}...': {traceback.format_exc()}"
        )
        return None
