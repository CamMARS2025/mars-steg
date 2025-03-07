
"""
Module Name: base_model.py
------------------------


This module contains the implementation of the base classes for models in the mars-steg framework to load the models for training and evaluating an LLM for elicitng steganography.

Classes:
    - BaseModel

Examples:
    Cf. mars_steg.README.md

"""

from mars_steg.config import ModelConfig
from typing import Dict, List, Optional
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from abc import ABCMeta, abstractmethod
import torch
from trl import AutoModelForCausalLMWithValueHead
from mars_steg.config import GenerationConfig
import warnings
try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None
    warnings.warn('Reintroduce: BitsAndBytesConfig. Import hasn\'t been successful.')


Conversation = List[List[Dict[str, str]]]

class BaseModel(metaclass=ABCMeta):
    """
    A base class for a model that integrates loading and managing a transformer-based model 
    (with support for LoRA) and handles tokenization and generation.

    Attributes
    ----------
    model_config : ModelConfig
        Configuration for the model, including model name and precision settings.
    model_name : str
        Name of the model being loaded.
    tokenizer : AutoTokenizer
        Tokenizer used to convert text to tokens for the model.
    lora : bool
        Whether the model uses LoRA (Low-Rank Adaptation).
    precission_mode : str
        Mode used for model precision (e.g., '4bits', '8bits').
    model_save_path : str
        Path to save the model.
    output_delimitation_tokens : Dict[str, str]
        Tokens used to delimit model outputs.
    generation_config : GenerationConfig
        Configuration for controlling generation behavior.
    device : Optional[str]
        The device ('cpu' or 'cuda') where the model will run.
    model : Optional[AutoModelForCausalLM]
        The model object, either loaded from a checkpoint or provided.
    
    Methods
    -------
    load_model() -> AutoModelForCausalLM
        Loads the model based on configuration (including LoRA if applicable).
    log_loading()
        Logs the loading process of the model.
    duplicate(whitebox_model: Optional[AutoModelForCausalLM]) -> BaseModel
        Creates a copy of the current model instance.
    get_message(user_string: str, system_prompt: str) -> Dict
        Abstract method to retrieve a message from the user/system.
    transform_conversation(conversations_list: Conversation)
        Abstract method to transform a list of conversations.
    batchize_conversation(user_prompts: Conversation, system_prompt: Optional[str] = None) -> List[List[Dict]]
        Creates a batch of messages based on the user prompts and system prompt.
    full_generate(conversations_list: Conversation) -> List[str]
        Generates responses for a list of conversations using the model.
    """
    def __init__(self, model_config: ModelConfig, tokenizer: AutoTokenizer, generation_config: GenerationConfig, output_delimitation_tokens: Dict[str, str], device: Optional[str] = None, model=None, precision_override: Optional[str]=None):
        """
        Initializes the base model with the provided configurations and model.
        
        Parameters
        ----------
        model_config : ModelConfig
            Configuration for the model.
        tokenizer : AutoTokenizer
            Tokenizer for the model.
        generation_config : GenerationConfig
            Configuration for generation behavior.
        output_delimitation_tokens : Dict[str, str]
            Tokens used for delimiting outputs.
        device : Optional[str]
            Device where the model should be loaded ('cpu' or 'cuda').
        model : Optional[AutoModelForCausalLM]
            Pre-loaded model to use, otherwise the model will be loaded using the provided configuration.
        precision_override: Optional[str]
            Override precision in specific case (problem in duplicating from ref model when quantized)
        """

        self.model_config = model_config
        self.model_name = model_config.model_name
        self.tokenizer = tokenizer
        self.lora = model_config.lora
        if precision_override is None:
            self.precission_mode = model_config.load_precision_mode
        else:
            self.precission_mode = precision_override
        self.model_save_path = model_config.model_save_path
        self.output_delimitation_tokens = output_delimitation_tokens
        self.generation_config = generation_config
        self.device = device
        if model is not None:
            self.model = model
        else:
            self.model = self.load_model(device)

        # self.model.gradient_checkpointing_enable()

    def load_model(self, device: str = None):
        """
        Loads the model based on configuration.

        This function handles precision mode and optional LoRA configuration, 
        loads the model using either a regular or LoRA setup, and moves it to 
        the device. 

        Parameters
        ----------
        device : Optional[str]
            Device where the model should be loaded ('cpu' or 'cuda').

        Returns
        -------
        AutoModelForCausalLM
            The loaded model.
        """
        self.log_loading()
        quantization_config = {
            "4bits": BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.float16,  bnb_4bit_quant_type="nf4"),
            "8bits": BitsAndBytesConfig(load_in_8bit=True)
        }.get(self.precission_mode, None)
        print(device)

        warnings.warn('Reintroduce: model.resize_token_embeddings(len(self.tokenizer)) ?')
        if self.lora:
            lora_config = LoraConfig(
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "v_proj"],
                 **self.model_config.lora_params
            )
            model = AutoModelForCausalLMWithValueHead.from_pretrained(
                self.model_name,
                peft_config=lora_config,
                torch_dtype=torch.bfloat16,
                quantization_config=quantization_config,
            )

        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                quantization_config=quantization_config,
            )
        if device is not None:
            model.to(device)
        
        for name, param in model.named_parameters():
            if param.device != torch.device(device):
                print(f"Warning: Layer {name} is on {param.device}, expected {device}")
                param.data = param.data.to(device)  # Move manually if needed

        return model
    
    def log_loading(self):
        """
        Logs the loading process of the model to the console.
        """
        print("------------------")
        print(f"Model loaded in {self.precission_mode} mode")
        print("------------------")
    
    def duplicate(self, whitebox_model: Optional[AutoModelForCausalLM], device, precision_override: Optional[str]):
        """
        Creates a duplicate of the current model, sharing the same configuration 
        but with a new model instance. The model is set to evaluation mode and 
        gradients are disabled.

        Parameters
        ----------
        whitebox_model : Optional[AutoModelForCausalLM]
            A pre-loaded model to use for the duplicate instance.
        precision_override: Optional[str]
            Override precision in specific case (problem in duplicating from ref model when quantized)

        Returns
        -------
        BaseModel
            A new instance of the model with the same configuration.
        """
        # Create a new instance of the same class
        new_instance = self.__class__(
            model_config=self.model_config,
            tokenizer=self.tokenizer,
            output_delimitation_tokens = self.output_delimitation_tokens,
            generation_config = self.generation_config,
            model = whitebox_model,
            device = device,
            precision_override = precision_override
        )

        new_instance.model.eval()
        for param in new_instance.model.parameters():
            param.requires_grad = False
            
        return new_instance

    @abstractmethod
    def get_message(self, user_string: str, system_prompt: str) -> Dict:
        """
        Abstract method to retrieve a message from the user/system.

        Parameters
        ----------
        user_string : str
            The string input from the user.
        system_prompt : str
            The system prompt to guide the conversation.

        Returns
        -------
        Dict
            A dictionary containing the message.
        """
        pass

    @abstractmethod
    def transform_conversation(self, conversations_list: Conversation, prompt_thinking_helper: Optional[str] = None) -> Conversation:
        """
        Abstract method to retrieve a message from the user/system.

        Parameters
        ----------
        user_string : str
            The string input from the user.
        system_prompt : str
            The system prompt to guide the conversation.
        prompt_thinking_helper: str
            A prompt that helps models to generate the way we want

        Returns
        -------
        Dict
            A dictionary containing the message.
        """
        pass

    @abstractmethod
    def tokenize(self, conversations_list: Conversation | List[List[str]]) -> List[Dict[str, torch.TensorType]]:
        pass

    def batchize_conversation(
        self, user_prompts: Conversation, system_prompt: Optional[str] = None):
        """
        Creates a batch of messages for multiple conversations. Optionally includes a system prompt.

        Parameters
        ----------
        user_prompts : Conversation
            A list of user prompts.
        system_prompt : Optional[str]
            A system prompt to prepend to each conversation (default is None).

        Returns
        -------
        List[List[Dict]]
            A list of batches, each containing dictionaries for system and user messages.
        """
        if system_prompt is None:
            batch_messages = [
                [
                    self.get_message(
                        "user", user_prompt
                    )  # {"role": "user", "content": user_prompt}
                ]
                for user_prompt in user_prompts
            ]

        else:
            batch_messages = [
                [
                    self.get_message(
                        "system", system_prompt
                    ),  # {"role": "system", "content": system_prompt},
                    self.get_message(
                        "user", user_prompt
                    ),  # {"role": "user", "content": user_prompt}
                ]
                for user_prompt in user_prompts
            ]
        return batch_messages

    def full_generate(self, 
                      conversations_list: Conversation | List[List[str]], 
                      is_neural_assessor: bool = False, 
                      is_neural_overseer: bool = False, 
                      neural_assessor_thinking_helper: str = None,
                      neural_overseer_thinking_helper: str = None):
        """
        Generates responses for a list of conversations using the model. The inputs are tokenized 
        and processed before generating the outputs. The responses are then decoded back to text.

        Parameters
        ----------
        conversations_list : Conversation
            The list of conversations to process and generate responses for.

        Returns
        -------
        List[str]
            A list of decoded responses from the model.
        """

        if is_neural_overseer or is_neural_assessor:
            if neural_overseer_thinking_helper is None and is_neural_overseer:
                warnings.warn("No neural_overseer_thinking_helper provide, the model might produce useless tokens that could lead it to hallucinate")
                transformed_conversations = [self.transform_conversation(conversation) for conversation in conversations_list]
            elif neural_assessor_thinking_helper is None and is_neural_assessor:
                warnings.warn("No neural_assessor_thinking_helper provide, the model might produce useless tokens that could lead it to hallucinate")
                transformed_conversations = [self.transform_conversation(conversation) for conversation in conversations_list]
            else:
                transformed_conversations = [
                    self.transform_conversation(conversation, neural_overseer_thinking_helper) if is_neural_overseer 
                    else self.transform_conversation(conversation,neural_assessor_thinking_helper)  
                    for conversation in conversations_list
                    ]
        else:
            transformed_conversations = [self.transform_conversation(conversation) for conversation in conversations_list]
        inputs = self.tokenize(conversations_list=transformed_conversations)
        inputs = {key: tensor.to(self.device) for key, tensor in inputs.items()}

        generation_kwargs = self.generation_config.to_generation_dict()
        {k: v for k, v in self.generation_config.to_dict().items() if k not in ['return_prompt', 'generate_ref_response', 'batch_size']}

        # Run the reference model
        outputs = self.model.generate(**inputs, **generation_kwargs)

        # These always work, so can diretly set
        decoded_responses = [
            # TODO: self.tokenizer.decode(r.squeeze()[prompt_length:]) for prompt_length, r in zip(prompt_lengths, outputs)
            self.tokenizer.decode(r.squeeze()) for r in outputs
        ]
        return decoded_responses
