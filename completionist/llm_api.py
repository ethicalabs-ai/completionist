import traceback
from typing import Optional, Union, Dict

from pydantic import BaseModel
from openai import OpenAI as OpenAIClient
from outlines import OpenAI as OutlinesOpenAI
from outlines import inputs as outlines_inputs
import httpx


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
    reasoning_effort: Optional[str] = None,
    reasoning: Optional[str] = None,
) -> Union[Dict[str, Optional[str]], BaseModel]:
    """
    Sends a prompt to an LLM API to get a completion.

    If a Pydantic schema is provided, it uses the 'outlines' library for
    structured, schema-enforced generation via the native generate() API.
    Otherwise, it performs a standard text completion request.

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
        Otherwise, returns a dict with 'content' and 'reasoning_content' keys.
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
        client = OpenAIClient(
            base_url=api_url,
            api_key=api_token or "dummy",
            http_client=httpx.Client(
                limits=httpx.Limits(max_connections=1000, max_keepalive_connections=100)
            ),
        )

        if pydantic_schema:
            generator = OutlinesOpenAI(client=client, model_name=model_name)
            chat_prompt = outlines_inputs.Chat(messages)
            generate_kwargs = {}
            if reasoning_effort is not None:
                generate_kwargs["reasoning_effort"] = reasoning_effort
            if reasoning is not None:
                generate_kwargs["extra_body"] = {"reasoning": reasoning}
            return generator.generate(
                chat_prompt,
                output_type=pydantic_schema,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                **generate_kwargs,
            )
        else:
            extra_kwargs = {}
            extra_body = {}
            if reasoning_effort is not None:
                extra_kwargs["reasoning_effort"] = reasoning_effort
            if reasoning is not None:
                extra_body["reasoning"] = reasoning
            result = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                extra_body=extra_body if extra_body else None,
                **extra_kwargs,
            )
            return {
                "content": result.choices[0].message.content,
                "reasoning_content": getattr(
                    result.choices[0].message, "reasoning_content", None
                ),
            }
    except Exception:
        print(
            f"Error during structured generation for prompt: '{prompt[:50]}...': {traceback.format_exc()}"
        )
        return None
