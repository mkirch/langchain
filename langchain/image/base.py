"""Base interface for image models to expose."""
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import yaml
from pydantic import BaseModel, Extra, Field, validator

import langchain
from langchain.callbacks import get_callback_manager
from langchain.callbacks.base import BaseCallbackManager
from langchain.schema import ImageModelGeneration, ImageModelResult


def _get_verbosity() -> bool:
    return langchain.verbose


def get_prompts(
    params: Dict[str, Any], prompts: List[str]
) -> Tuple[Dict[int, List], str, List[int], List[str]]:
    """Get prompts that are already cached."""
    image_model_string = str(sorted([(k, v) for k, v in params.items()]))
    missing_prompts = []
    missing_prompt_idxs = []
    existing_prompts = {}
    for i, prompt in enumerate(prompts):
        if langchain.image_model_cache is not None:
            cache_val = langchain.image_model_cache.lookup(prompt, image_model_string)
            if isinstance(cache_val, list):
                existing_prompts[i] = cache_val
            else:
                missing_prompts.append(prompt)
                missing_prompt_idxs.append(i)
    return existing_prompts, image_model_string, missing_prompt_idxs, missing_prompts

def update_cache(
    existing_prompts: Dict[int, List],
    image_string: str,
    missing_prompt_idxs: List[int],
    new_results: ImageModelResult,
    prompts: List[str],
) -> Optional[dict]:
    """Update the cache and get the ImageModel output."""
    for i, result in enumerate(new_results.generations):
        existing_prompts[missing_prompt_idxs[i]] = result
        prompt = prompts[missing_prompt_idxs[i]]
        if langchain.image_model_cache is not None:
            langchain.image_model_cache.update(prompt, image_string, result)
    image_model_output = new_results.image_output
    return image_model_output


class BaseImageModel(BaseModel, ABC):
    """ImageModel wrapper should take in a prompt and return a string."""

    cache: Optional[bool] = None
    verbose: bool = Field(default_factory=_get_verbosity)
    """Whether to print out response text."""
    callback_manager: BaseCallbackManager = Field(default_factory=get_callback_manager)

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @validator("callback_manager", pre=True, always=True)
    def set_callback_manager(
        cls, callback_manager: Optional[BaseCallbackManager]
    ) -> BaseCallbackManager:
        """If callback manager is None, set it.

        This allows users to pass in None as callback manager, which is a nice UX.
        """
        return callback_manager or get_callback_manager()

    @validator("verbose", pre=True, always=True)
    def set_verbose(cls, verbose: Optional[bool]) -> bool:
        """If verbose is None, set it.

        This allows users to pass in None as verbose to access the global setting.
        """
        if verbose is None:
            return _get_verbosity()
        else:
            return verbose

    @abstractmethod
    def _generate(
        self, prompts: List[str]
    ) -> ImageModelResult:
        """Run the ImageModel on the given prompts."""

    @abstractmethod
    async def _agenerate(
        self, prompts: List[str]
    ) -> ImageModelResult:
        """Run the ImageModel on the given prompts."""

    def generate(
        self, prompts: List[str]
    ) -> ImageModelResult:
        """Run the ImageModel on the given prompt and input."""
        # If string is passed in directly no errors will be raised but outputs will
        # not make sense.
        if not isinstance(prompts, list):
            raise ValueError(
                "Argument 'prompts' is expected to be of type List[str], received"
                f" argument of type {type(prompts)}."
            )
        disregard_cache = self.cache is not None and not self.cache
        if langchain.image_model_cache is None or disregard_cache:
            # This happens when langchain.cache is None, but self.cache is True
            if self.cache is not None and self.cache:
                raise ValueError(
                    "Asked to cache, but no cache found at `langchain.cache`."
                )
            self.callback_manager.on_image_model_start(
                {"name": self.__class__.__name__}, prompts, verbose=self.verbose
            )
            try:
                output = self._generate(prompts)
            except (KeyboardInterrupt, Exception) as e:
                self.callback_manager.on_image_model_error(e, verbose=self.verbose)
                raise e
            self.callback_manager.on_image_model_end(output, verbose=self.verbose)
            return output
        params = self.dict()
        (
            existing_prompts,
            image_model_string,
            missing_prompt_idxs,
            missing_prompts,
        ) = get_prompts(params, prompts)
        if len(missing_prompts) > 0:
            self.callback_manager.on_image_model_start(
                {"name": self.__class__.__name__}, missing_prompts, verbose=self.verbose
            )
            try:
                new_results = self._generate(missing_prompts)
            except (KeyboardInterrupt, Exception) as e:
                self.callback_manager.on_image_model_error(e, verbose=self.verbose)
                raise e
            self.callback_manager.on_image_model_end(new_results, verbose=self.verbose)
            image_model_output = update_cache(
                existing_prompts, image_model_string, missing_prompt_idxs, new_results, prompts
            )
        else:
            image_model_output = {}
        generations = [existing_prompts[i] for i in range(len(prompts))]
        return ImageModelResult(generations=generations, image_model_output=image_model_output)

    async def agenerate(
        self, prompts: List[str]
    ) -> ImageModelResult:
        """Run the ImageModel on the given prompt and input."""
        disregard_cache = self.cache is not None and not self.cache
        if langchain.image_model_cache is None or disregard_cache:
            # This happens when langchain.cache is None, but self.cache is True
            if self.cache is not None and self.cache:
                raise ValueError(
                    "Asked to cache, but no cache found at `langchain.cache`."
                )
            if self.callback_manager.is_async:
                await self.callback_manager.on_image_model_start(
                    {"name": self.__class__.__name__}, prompts, verbose=self.verbose
                )
            else:
                self.callback_manager.on_image_model_start(
                    {"name": self.__class__.__name__}, prompts, verbose=self.verbose
                )
            try:
                output = await self._agenerate(prompts)
            except (KeyboardInterrupt, Exception) as e:
                if self.callback_manager.is_async:
                    await self.callback_manager.on_image_model_error(e, verbose=self.verbose)
                else:
                    self.callback_manager.on_image_model_error(e, verbose=self.verbose)
                raise e
            if self.callback_manager.is_async:
                await self.callback_manager.on_image_model_end(output, verbose=self.verbose)
            else:
                self.callback_manager.on_image_model_end(output, verbose=self.verbose)
            return output
        params = self.dict()
        (
            existing_prompts,
            image_model_string,
            missing_prompt_idxs,
            missing_prompts,
        ) = get_prompts(params, prompts)
        if len(missing_prompts) > 0:
            if self.callback_manager.is_async:
                await self.callback_manager.on_image_model_start(
                    {"name": self.__class__.__name__},
                    missing_prompts,
                    verbose=self.verbose,
                )
            else:
                self.callback_manager.on_image_model_start(
                    {"name": self.__class__.__name__},
                    missing_prompts,
                    verbose=self.verbose,
                )
            try:
                new_results = await self._agenerate(missing_prompts)
            except (KeyboardInterrupt, Exception) as e:
                if self.callback_manager.is_async:
                    await self.callback_manager.on_image_model_error(e, verbose=self.verbose)
                else:
                    self.callback_manager.on_image_model_error(e, verbose=self.verbose)
                raise e
            if self.callback_manager.is_async:
                await self.callback_manager.on_image_model_end(
                    new_results, verbose=self.verbose
                )
            else:
                self.callback_manager.on_image_model_end(new_results, verbose=self.verbose)
            image_model_output = update_cache(
                existing_prompts, image_model_string, missing_prompt_idxs, new_results, prompts
            )
        else:
            image_model_output = {}
        generations = [existing_prompts[i] for i in range(len(prompts))]
        return ImageModelResult(generations=generations, image_model_output=image_model_output)

    def __call__(self, prompt: str) -> str:
        """Check Cache and run the ImageModel on the given prompt and input."""
        return self.generate([prompt]).generations[0][0].text

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}

    def __str__(self) -> str:
        """Get a string representation of the object for printing."""
        cls_name = f"\033[1m{self.__class__.__name__}\033[0m"
        return f"{cls_name}\nParams: {self._identifying_params}"

    @property
    @abstractmethod
    def _image_model_type(self) -> str:
        """Return type of image model."""

    def dict(self, **kwargs: Any) -> Dict:
        """Return a dictionary of the ImageModel."""
        starter_dict = dict(self._identifying_params)
        starter_dict["_type"] = self._image_model_type
        return starter_dict

    def save(self, file_path: Union[Path, str]) -> None:
        """Save the ImageModel.

        Args:
            file_path: Path to file to save the ImageModel to.

        Example:
        .. code-block:: python

            image_model.save(file_path="path/image_model.yaml")
        """
        # Convert file to Path object.
        if isinstance(file_path, str):
            save_path = Path(file_path)
        else:
            save_path = file_path

        directory_path = save_path.parent
        directory_path.mkdir(parents=True, exist_ok=True)

        # Fetch dictionary to save
        prompt_dict = self.dict()

        if save_path.suffix == ".json":
            with open(file_path, "w") as f:
                json.dump(prompt_dict, f, indent=4)
        elif save_path.suffix == ".yaml":
            with open(file_path, "w") as f:
                yaml.dump(prompt_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"{save_path} must be json or yaml")


class ImageModel(BaseModel):
    """ImageModel class that expect subclasses to implement a simpler call method.

    The purpose of this class is to expose a simpler interface for working
    with ImageModels, rather than expect the user to implement the full _generate method.
    """

    @abstractmethod
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Run the ImageModel on the given prompt and input."""

    def _generate(
        self, prompts: List[str], stop: Optional[List[str]] = None
    ) -> ImageModelResult:
        """Run the ImageModel on the given prompt and input."""
        # TODO: add caching here.
        generations = []
        for prompt in prompts:
            text = self._call(prompt)
            generations.append([ImageModelGeneration(text=text)])
        return ImageModelResult(generations=generations)

    async def _agenerate(
        self, prompts: List[str], stop: Optional[List[str]] = None
    ) -> ImageModelResult:
        """Run the ImageModel on the given prompt and input."""
        raise NotImplementedError("Async generation not implemented for this LLM.")
