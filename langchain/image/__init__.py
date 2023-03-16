"""Wrappers on top of large language models APIs."""
from typing import Dict, Type

from langchain.image.base import BaseImageModel
from langchain.image.huggingface_hub import HuggingFaceHubImageGeneration
# from langchain.image.openai import OpenAI  # not azure yet


__all__ = [
    # "OpenAI",
    "HuggingFaceHub"
]

type_to_cls_dict: Dict[str, Type[BaseImageModel]] = {
    "huggingface_hub": HuggingFaceHubImageGeneration,
    # "openai": OpenAI,
}