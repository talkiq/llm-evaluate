import logging
import os
import warnings
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import backoff

from .model import Model
from .model import Tokenizer
from .model import TruncationSide

with warnings.catch_warnings():
    # TODO: remove when gcp updates aiplatform to use importlib
    warnings.simplefilter('ignore', category=DeprecationWarning)
    from google.cloud import aiplatform
    from vertexai.preview import generative_models
    from vertexai.preview.generative_models import GenerationConfig
    from vertexai.preview.generative_models import GenerativeModel
    from vertexai.preview.language_models import TextGenerationModel

# Trying to access gemini response text will result in the error below if
# safety issues are detected by the model: AttributeError: Content has no
# parts.
GEMINI_SAFETY_CONFIG = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH:
    generative_models.HarmBlockThreshold.BLOCK_NONE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT:
    generative_models.HarmBlockThreshold.BLOCK_NONE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT:
    generative_models.HarmBlockThreshold.BLOCK_NONE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT:
    generative_models.HarmBlockThreshold.BLOCK_NONE,
    generative_models.HarmCategory.HARM_CATEGORY_UNSPECIFIED:
    generative_models.HarmBlockThreshold.BLOCK_NONE,
}


class GCPTokenizer(Tokenizer):

    def __init__(self, _model: str, **kwargs: str):
        self._tokenizer = lambda x: x
        self.max_tokens = kwargs.pop('model_max_length', 4096)
        self.truncation_side = TruncationSide(
            kwargs.pop('truncation_side', TruncationSide.RIGHT.value))

    def tokenize(self, inputs: List[str], **_kwargs: Any) -> List[List[str]]:
        """Tokenize input strings to their token ids."""
        return inputs

    def decode(self, outputs: Any) -> List[str]:
        """Decode and detokenize the model output."""
        return outputs

    def get_num_tokens(self, input_: str) -> int:
        """Get the count of tokens in the input string."""
        return len(input_)

    def __call__(self, inputs: List[str], **kwds: Any) -> List[List[str]]:
        return self.tokenize(inputs)


class GCPModel(Model):
    def __init__(self,
                 model_or_path: str,
                 prompt_start_placeholder: Optional[str] = None,
                 prompt_end_placeholder: Optional[str] = None,
                 tokenizer_args: Optional[Dict[str, str]] = None,
                 model_load_args: Optional[Dict[str, str]] = None,
                 model_inference_args: Optional[Dict[str, str]] = None,
                 **kwargs: Any,
                 ) -> None:
        super().__init__(
            model_or_path=model_or_path,
            prompt_start_placeholder=prompt_start_placeholder,
            prompt_end_placeholder=prompt_end_placeholder,
            tokenizer_args=tokenizer_args,
            model_load_args=model_load_args,
            model_inference_args=model_inference_args,
            **kwargs,
        )
        self.gcp_model: Optional[GenerativeModel | TextGenerationModel] = None

    def init(self) -> None:
        self.truncation_side = self.model_load_args.get(
            'truncation_side', TruncationSide.RIGHT.value)
        self.max_input_tokens = self.model_load_args.get(
            'max_input_tokens', 8192)
        self.model_load_args = {
            'max_output_tokens': self.model_load_args.get('max_new_tokens', 128),
            'top_k': self.model_load_args.get(
                'top_k', 1) or self.model_load_args.get('num_beams', 1),
            'temperature': self.model_load_args.get('temperature', 0.),
        }
        self.model_name = self.model_or_path
        self.init_tokenizer(self.model_or_path)

    def _init_model(self) -> None:
        # this is a working approach to get auth done on vm,
        # not sure if this is the best way to do so.
        if not self.gcp_model:
            aiplatform.init(project=os.environ.get('GCLOUD_PROJECT'))
            if 'gemini' in self.model_or_path:
                self.gcp_model = GenerativeModel(self.model_name)
            else:
                self.gcp_model = TextGenerationModel.from_pretrained(
                    self.model_name)

    @classmethod
    def truncate(cls, query: str, max_length: int,
                 truncation_side: str) -> str:
        # TODO: Max length based on token limit, this is crude
        overflow = len(query.split()) - max_length
        if overflow > 0:
            if truncation_side == 'right':
                query = ' '.join(query.split()[:-overflow])
            else:
                query = ' '.join(query.split()[overflow:])

            logging.warning(
                'token limit reached, truncating from %s.', truncation_side)
        return query

    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_time=180,
    )
    def query_model(self, query: str) -> str:
        query = self.truncate(
            query, self.max_input_tokens, self.truncation_side)
        if 'gemini' in self.model_or_path:
            generation_config = GenerationConfig(top_p=0.,
                                                 **self.model_load_args,
                                                 )
            response = self.gcp_model.generate_content(contents=query,
                                                       generation_config=generation_config,
                                                       safety_settings=GEMINI_SAFETY_CONFIG,
                                                       )
            # dealing with recitation issue: https://github.com/google/generative-ai-docs/issues/257
            # The recitation stop reason is used whenever the model begins to recite training data,
            # especially copyrighted material.
            # In such cases, the model does not produce response text.
            try:
                return response.text
            except (AttributeError, ValueError):
                return ''
        else:
            response = self.gcp_model.predict(prompt=query,
                                              top_p=0.,
                                              **self.model_load_args,
                                              )
            return response.text

    def process(self, inputs: List[str], **model_kwargs: Any) -> List[str]:
        self._init_model()
        responses = [self.query_model(input_) for input_ in inputs]
        logging.debug('prompt/response:\n>>>%s\n<<<%s',
                      inputs[0], responses[0])
        return responses

    def init_tokenizer(self, model_or_path: str, **kwargs: Any) -> None:
        self._tokenizer = GCPTokenizer(model_or_path)
