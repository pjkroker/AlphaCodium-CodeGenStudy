from litellm import acompletion, AsyncOpenAI

_vllm_client = None

import litellm



def _get_vllm_client(api_base: str, api_key: str):
    global _vllm_client
    if _vllm_client is None:
        _vllm_client = AsyncOpenAI(
            base_url=api_base,
            api_key=api_key,
        )
    return _vllm_client


async def acompletion_compat(
    *,
    model: str,
    messages: list,
    api_base: str | None = None,
    api_key: str | None = None,
    **kwargs,
):
    """
    Drop-in replacement for litellm.acompletion
    Supports vLLM transparently.
    """

    # ---------- vLLM ----------
    if model.startswith("hosted_vllm/") or model.startswith("vllm/"):
        if api_base is None:
            raise ValueError("api_base must be set for vLLM")

        client = _get_vllm_client(
            api_base=api_base,
            api_key=api_key or "dummy",
        )

        resp = await client.chat.completions.create(
            model=model.split("/", 1)[1],
            messages=messages,
            **kwargs,
        )

        # normalize to LiteLLM-like dict
        return {
            "choices": [
                {
                    "message": {
                        "content": resp.choices[0].message.content
                    },
                    "finish_reason": resp.choices[0].finish_reason,
                }
            ]
        }

    # ---------- everything else ----------
    return await acompletion(
        model=model,
        messages=messages,
        **kwargs,
    )
