import asyncio
import copy
import json
import logging
from typing import Any

import aiohttp
import llama_cpp

from .llama_cpp import LlamaCppTokenizer
from .model import Model


class LlamaCppServer(Model):
    def init(self) -> None:
        self.model_load_args = {
            'n_ctx': 4096,
            **(self.model_load_args or {}),
        }

        _model_load_args = copy.deepcopy(self.model_load_args)
        self.model_load_args = {}
        for arg in _model_load_args:
            if arg not in {
                'n_ctx', 'n_gqa', 'n_gpu_layers', 'n_threads',
                'flash_attn',
            }:
                logging.warning(
                    'model load arg is not supported by LLamaCpp: %s', arg,
                )
                continue
            self.model_load_args[arg] = _model_load_args[arg]

        self.model = llama_cpp.Llama(
            model_path=self.model_or_path,
            **self.model_load_args,
        )
        self.init_tokenizer('', **{**self.model_load_args, **_model_load_args})

    def init_tokenizer(self, model_or_path: str, **kwargs: Any) -> None:
        self._tokenizer = LlamaCppTokenizer(self.model, **kwargs)

    async def query_llama_server(
            self, prompts: list[str], **model_kwargs: Any,
    ) -> list[str]:
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=10),
        ) as session:
            url = model_kwargs.pop('api_url')
            requests = [
                self.make_request(
                    session, url, {
                        **model_kwargs,
                        'prompt': prompt,
                    },
                ) for prompt in prompts
            ]
            return await asyncio.gather(*requests)

    @staticmethod
    async def make_request(
        session: aiohttp.ClientSession, url: str, data: dict[str, Any],
    ) -> str:
        request_body = json.dumps(data)
        async with session.post(
            url,
            headers={
                'Content-type': 'application/json',
            },
            data=request_body,
        ) as resp:
            resp.raise_for_status()
            payload = await resp.json()
            return str(payload['content'])

    def _process(self, inputs: list[str], **model_kwargs: Any) -> list[str]:
        return asyncio.run(self.query_llama_server(inputs, **model_kwargs))

    def process(self, inputs: list[str], **model_kwargs: Any) -> list[str]:
        responses = self._process(inputs, **model_kwargs)
        logging.debug(
            'prompt/response:\n>>>%s\n<<<%s',
            inputs[0],
            responses[0],
        )
        return responses
