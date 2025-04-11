import torch as t
import numpy as np
import random
from torch import nn
import torch
from transformers import AutoModelForCausalLM
from typing import Optional, Dict, List, Tuple
from torch.optim import AdamW
import logging
import traceback
from transformers import AutoTokenizer

from tqdm import tqdm
from mars_steg.config import ModelConfig, OptimizerConfig, PromptConfig, GenerationConfig
from mars_steg.utils.answer_extraction import extract_cots, extracts_no_cots, extract_floatingpoint_overseer_or_assessor_answer
from mars_steg.utils.exceptions import LLMTranscriptExtractionError
from mars_steg.utils.score import CoTGapSummary
from mars_steg.model import BaseModel, GPT2Model, Llama_31_8B_Instruct, Llama_32_1B_Instruct, DeepSeek_R1_1pt5B
from trl.models.modeling_base import PreTrainedModelWrapper, LAYER_PATTERNS
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

from torch.utils.data import DataLoader

from mars_steg.utils.prompt_data import BatchPromptData

from mars_steg import task
from mars_steg import language
from mars_steg.task import TorchDatasetTask
from mars_steg.language import LanguageAspect

from trl import create_reference_model, PPOTrainer

from accelerate import Accelerator
accelerate = Accelerator()
DEVICE =  accelerate.device

def get_dataloaders_and_ref_model(
    dataset_class_name: str,
    penalisation_class_name: str,
    dataset_class_kwargs: Optional[Dict],
    prompt_config: PromptConfig,
    penalisation_class_kwargs: Optional[Dict],
    train_proportion: float,
    validation_proportion: float,
    batch_size: int,
    training_model: Optional[BaseModel] = None,  
    *_,
    override_create_ref_model: bool = False,
    number_shared_layers_for_ref_model: Optional[int] = None,
    neural_overseer_model: Optional[BaseModel] = None,
    neural_assessor_model: Optional[BaseModel] = None,
    device_map: Optional[Dict] = None,
    test_train_split_kwargs: Optional[Dict] = {}
):
    """

    A getter for the Task subclasses to be used here; handles splitting into train, val, test. 
    Also returns torch dataloaders for all three. 

    Parameters
    ----------
    dataset_class_name : str
        The name of the Task subclass used here (defined name, NOT name attribute)
    penalisation_class_name : str
        The name of the LanguageAspect subclass used here
    dataset_class_kwargs : Optional[Dict]
        Kwargs to configure the dataset class being used; these should NOT include (i) language_aspect, or (ii) prompt_config, which will be set by other args.  
    prompt_config : PromptConfig
        PromptConfig to configure the prompt generation in the task subclass. May also be used to configure prompts in LanguageAspect, e.g if this is a NeuralOverseer. 
    penalisation_class_kwargs : Optional[Dict]
        Kwargs to configure the dataset class being used; these should NOT include prompt_config, which will be set automatically. 
    train_proportion : float
        Proportion of dataset to use for train (float in range [0-1])
    validation_proportion : float
        Proportion of dataset to use for validation (float in range [0-1])
    batch_size : int
        Batch size for torch dataloaders
    training_model : Optional[BaseModel], optional
        By default None. Need to pass this if if `override_create_ref_model` is True or if the neural overseer or task requires it
    override_create_ref_model : bool, optional
        By default False. If True, forces creation of a reference model. 
    number_shared_layers_for_ref_model : Optional[int], optional
        Required if reference model created, by default None
    neural_overseer_model : Optional[BaseModel], optional
        If model for neural overseer has already been created, pass here. Otherwise the ref_model will be used.
    neural_assessor_model : Optional[BaseModel], optional
        If model for neural assessor has already been created, pass here. Otherwise the ref_model will be used.
    device: Optional[str] | Optional[Dict]
        Indicates to which kernel is assigned each model. Can be a single device if using accelerate


    Returns
    -------
    TorchDatasetTask
        A TorchDatasetTask instance that can be used to iterate through a training dataset 
    TorchDatasetTask
        Likewise, validation set
    TorchDatasetTask
        Likewise, test set
    torch.utils.data.DataLoader
        The actual torch dataloader for the training set
    torch.utils.data.DataLoader
        Likewise, validation set 
    torch.utils.data.DataLoader
        Likewise, test set
    trl.PreTrainedModelWrapper
        The reference model

    Notes
    -----
    Including ref model creation here because it is needed for the task,
    as well as potentially for the training!

    Create the reference model if `override_create_ref_model` is True
    or if the neural overseer or task requires it

    """

    dataset_class_kwargs = dataset_class_kwargs or dict()
    penalisation_class_kwargs = penalisation_class_kwargs or dict()
    language_aspect_class: LanguageAspect = getattr(language, penalisation_class_name)(**penalisation_class_kwargs, prompt_config=prompt_config)
    task_class: TorchDatasetTask = getattr(task, dataset_class_name)(language_aspect = language_aspect_class, prompt_config = prompt_config, batch_size = batch_size, **dataset_class_kwargs)

    need_reference_model = (
        override_create_ref_model or 
        (language_aspect_class.uses_local_neural_overseer and neural_overseer_model is None) or
        (task_class.uses_local_neural_assessor and neural_assessor_model is None)
    )


    if language_aspect_class.uses_local_neural_overseer and need_reference_model:
        print("----------------------------")
        print("Using local overseer")
        print(f"device {device_map['overseer']}")
        print("----------------------------")
        ref_model_class: BaseModel = training_model.duplicate(None, device_map["overseer"], precision_override="full")
        reference_model = ref_model_class.model

    elif task_class.uses_local_neural_assessor and need_reference_model:
        ref_model_class: BaseModel = training_model.duplicate(None, device_map["assessor"], precision_override="full")
        reference_model = ref_model_class.model
        
    elif not language_aspect_class.uses_local_neural_overseer and need_reference_model and not is_deepspeed_zero3_enabled():
        reference_model = create_reference_model(
            training_model.model, num_shared_layers=number_shared_layers_for_ref_model,
            pattern = 'pretrained_model.base_model.model.model.layers.{layer}'  # XXX: this works for peft LoRA, not tried for other things
        )


    elif need_reference_model and is_deepspeed_zero3_enabled():
        reference_model = create_reference_model_under_deepspeed_R3(
                model_name=training_model.model_name,
                num_shared_layers=number_shared_layers_for_ref_model,
                device= DEVICE
                # pattern = 'pretrained_model.base_model.model.model.layers.{layer}'  # XXX: this works for peft LoRA, not tried for other things #XXX: We will need to reintroduce it later on because we are not using it for now (26/02/2025).
            )

    else:
        reference_model = None

    if language_aspect_class.uses_local_neural_overseer:
        if neural_overseer_model is None:
            # XXX: 2025.02.15 FIND A WAY TO MOVE QUANTISED MODELS (WHICH MAY OR MAY NOT BE IN PEFT) OVER TO ANOTHER DEVICE // 
            neural_overseer: BaseModel = ref_model_class
        else:
            neural_overseer: AutoModelForCausalLM = neural_overseer_model.eval()
            raise NotImplementedError("The line below will fail because we are passing an AutoModelForCausalLM while the code is mainly working with BaseModel. We need to implement something to make it a BaseModel")
        language_aspect_class.recruit_neural_overseer(neural_overseer)    

    if task_class.uses_local_neural_assessor:
        if neural_assessor_model is None:
            neural_assessor: BaseModel = ref_model_class
        else:
            neural_assessor: AutoModelForCausalLM = neural_assessor_model.eval()
            raise NotImplementedError("The line below will fail because we are passing an AutoModelForCausalLM while the code is mainly working with BaseModel. We need to implement something to make it a BaseModel")
        task_class.recruit_neural_assessor(neural_assessor)   

    train_dataset, val_dataset, test_dataset = task_class.test_train_split(
        train_proportion=train_proportion,
        validation_proportion=validation_proportion,
        **test_train_split_kwargs
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=task_class.prompt_data_collate_fn)
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=task_class.prompt_data_collate_fn)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=task_class.prompt_data_collate_fn)

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, reference_model



def get_tokenizer(
    model_name: str
):

    """
    Get tokenizer and set pad token to be same as end-of-sequence token

    Parameters
    ----------
    model_name : str
        The model name - get string from huggingface

    Returns
    -------
    AutoTokenizer
        Configured tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_optimizer(
    optimizer_config: OptimizerConfig,
    model: nn.Module
) -> t.optim.Optimizer:
    
    """
    Returns an instance of torch.optim.Adam configured to work with submitted model/with settings from optimizer_config

    Parameters
    ----------
    optimizer_config : OptimizerConfig
        Optimisation config
    model : nn.Module
        torch model that the optimizer will be used with

    Returns
    -------
    torch.optim.Adam
        Configured optimizer
    """

    #TODO ADD AS WELL THE WARM UP SCHEDULER AND THE MAX GRAD NORM USING :
    """
    from torch.optim import AdamW

    optimizer = AdamW(
        model.parameters(),
        lr=5.0e-5,                # learning_rate
        weight_decay=0.001,       # weight_decay
        eps=1e-8,                 # adam_epsilon
        betas=(0.9, 0.999)        # Default AdamW betas
    )

    # Gradient clipping (max_grad_norm)
    import torch.nn.utils as nn_utils

    # Inside training loop, before optimizer.step():
    nn_utils.clip_grad_norm_(model.parameters(), 1.0)  # max_grad_norm

    # Learning rate scheduler with warmup_steps
    from transformers import get_scheduler

    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=500,     # warmup_steps
        num_training_steps=total_training_steps  # You need to define this
    )

    """
    if optimizer_config.optimizer_name == "adam":
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=optimizer_config.learning_rate,
            weight_decay=optimizer_config.weight_decay,
            eps= optimizer_config.adam_epsilon,
            fused=True,
        )
        return optimizer


def get_model(
    model_config: ModelConfig, 
    tokenizer: AutoTokenizer, 
    output_delimitation_tokens: Dict[str, str], 
    generation_config=GenerationConfig,
    device: Optional[str] = None,
    precision_override: Optional[str]=None
) -> BaseModel:
    
    """
    The getter for instances of our own model class, which wraps pretrained models. 

    Parameters
    ----------
    model_config : ModelConfig
        An instance of ModelConfig, which carries configuration details for the model
    tokenizer : AutoTokenizer
        tokenizer, use get_tokenizer() to load
    output_delimitation_tokens : Dict[str, str]
        A dict containing the tokens which will be used to delineate parts of the model's responses. 
        For the main model being trained, which emits a scratchpad and final output, the keys should be:
        - "start_output_token"
        - "end_output_token"
        - "start_scratchpad_token"
        - "end_scratchpad_token"
        And the values should be an appropriately chosen token, e.g <SCRATCHPAD>, </SCRATCHPAD> etc.
    generation_config : GenerationConfig
        Contains the keyword arguments that will be fed to self.model.generate() internally (some edits are made on the fly)
    device: str = None
        The device for the model to sit on (use torch keywords, eg 'cuda', 'cuda:0')
    precision_override: Optional[str]
        override precision if needed
    
    
    Returns
    -------
    BaseModel
        An instance of a subclass of BaseModel (our wrapper for pretrained models)

    Raises
    ------
    ValueError
        Throws an error if no-one has defined a wrapper for your model, or they've forgotten to add it to this function
    """


    model_name = model_config.model_name
    print(f'GETTING: {model_name}')
    if model_name == "gpt2":
        return GPT2Model(model_config=model_config, tokenizer=tokenizer, output_delimitation_tokens=output_delimitation_tokens, generation_config=generation_config, device=device, precision_override=precision_override)
    elif model_name == 'meta-llama/Llama-3.1-8B-Instruct':
        return Llama_31_8B_Instruct(model_config=model_config, tokenizer=tokenizer, output_delimitation_tokens=output_delimitation_tokens, generation_config=generation_config, device=device, precision_override=precision_override)
    elif model_name == 'meta-llama/Llama-3.2-1B-Instruct':
        return Llama_32_1B_Instruct(model_config=model_config, tokenizer=tokenizer, output_delimitation_tokens=output_delimitation_tokens, generation_config=generation_config, device=device, precision_override=precision_override)
    elif model_name == 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B':
        return DeepSeek_R1_1pt5B(model_config=model_config, tokenizer=tokenizer, output_delimitation_tokens=output_delimitation_tokens, generation_config=generation_config, device=device, precision_override=precision_override)
    elif "deepseek-ai" in model_name:
        return DeepSeek_R1_1pt5B(model_config=model_config, tokenizer=tokenizer, output_delimitation_tokens=output_delimitation_tokens, generation_config=generation_config, device=device, precision_override=precision_override)


    else:
        raise ValueError(model_name)


def get_device() -> str:

    """
    Getter for device (torch keyword)

    Returns
    -------
    str
        Appropriate torch device keyword e.g 'cuda'
    """


    if t.backends.mps.is_available():
        device = "mps"
    else:
        device = "cuda" if t.cuda.is_available() else "cpu"

    # if t.cuda.is_available() and t.cuda.device_count() > 1:
    # raise ValueError("Please specify which GPU to use with CUDA_VISIBLE_DEVICES")

    return device

def get_device_map(force_cpu = False) -> Dict[str, str]:
    """
    Allocates devices for models based on available GPUs.
    Main model always gets its own GPU (GPU 0).
    Other models are distributed across remaining GPUs optimally.

    Parameters
    ----------

    force_cpu : bool
        Forces models to go on cpu
    
    Returns
    -------
        dict: Model name to device mapping
    """
    # TODO: Update this so it uses available GPUs in a more optimal way
    if force_cpu:
        device = "cpu"
    else:
        device = get_device()

    if device in ["cpu", "mps"]:
        return {
        'main_model': f'{device}',  # Main model always gets GPU 0
        'ref_model': f'{device}',
        'overseer': f'{device}',   # These two should always be on the same device!
        'assessor': f'{device}'
    }
    
    num_gpus = t.cuda.device_count()
    print(f"Found {num_gpus} GPUs")

    if num_gpus < 1:
        raise ValueError("Must have at least 1 GPU")
    
    # Initialize device mapping
    device_map = {
        'main_model': 'cuda:0',  # Main model always gets GPU 0
    }
    
    if num_gpus == 1:
        # Everything goes on the same GPU
        device_map.update({
            'ref_model': 'cuda:0',
            'overseer': 'cuda:0',   # These two should always be on the same device!
            'assessor': 'cuda:0'
        })
    
    elif num_gpus == 2:
        # Main model on GPU 0, everything else on GPU 1
        device_map.update({
            'ref_model': 'cuda:1',
            'overseer': 'cuda:1',
            'assessor': 'cuda:1'
        })
    
    elif num_gpus == 3:
        # Main model on GPU 0, ref model on GPU 1, both overseers on GPU 2
        device_map.update({
            'ref_model': 'cuda:1',
            'overseer': 'cuda:2',
            'assessor': 'cuda:2'
        })
    
    else:  # num_gpus >= 4
        # Each model gets its own GPU
        device_map.update({
            'ref_model': 'cuda:1',
            'overseer': 'cuda:2',
            'assessor': 'cuda:3'
        })
    
    return device_map


def set_seed(seed: int = 42) -> None:

    """
    Sets random seeds for all libraries using RNG: numpy, random, torch

    Parameters
    ----------
    seed : int
        Integer seed
    """

    np.random.seed(seed)
    random.seed(seed)
    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)
        t.backends.cudnn.deterministic = True
        t.backends.cudnn.benchmark = False

def get_rewards_and_training_datas(
        batch_prompt_datas: BatchPromptData, 
        query_tensors: List[t.TensorType], 
        transcript_responses: List[t.TensorType], 
        train_dataset: TorchDatasetTask,
        skipping_failed_parsing_examples: Optional[bool] = True,
        t_weight: float = 1.0, 
        l_weight: float = 1.0,
        tokenizer: Optional[AutoTokenizer] = None,
        ) -> Tuple[List]:
    
    """
    Used after the model being trained in RL has produced a response, to mark this response using Task.reward_from_transcript(). 

    batch_prompt_datas come in imbued with answers extracted from the neural overseer and assessor already
    Any that failed should have been pruned from this argument anyway, so no need to double check here! 

    However: Just because the extraction worked does not mean that the score assignment will
    We do another round of pruning here if the extracted answers in either NOS or NA are not in the right format

    Parameters
    ----------
    batch_prompt_datas : BatchPromptData
        An object containing a list of PromptData objects; each PromptData contains all data associated with an individual training example
    query_tensors : List[t.TensorType]
        The original list of tensors used to generate the trained model's responses
    transcript_responses : List[t.TensorType]
        The original list of tensors generated in response to the inputs
    train_dataset : TorchDatasetTask
        The Task subclass describing the task the model is solving; in this case specifically that for the train dataset. 
    t_weight : float, optional
        The weight assigned to the task score (default is 1.0).
    l_weight : float, optional
        The weight assigned to the language score (default is 1.0).
    skipping_failed_parsing_examples: bool
        Skip examples that failed when parsing them for task score, language score, answer extraction, CoT extraction etc


    Returns
    -------
    List[t.TensorType]
        composite_reward_list - a list of the rewards from the overall reward function, already converted into torch tensors.
    List[float]
        task_score_list - a list of the task element of the reward (emitted for tracking purposes)
    List[float]
        language_score_list  - a list of the language element of the reward (emitted for tracking purposes)
    List[t.TensorType]
        train_query_tensors - Same as input param query_tensors, but with any that didn't work in .reward_from_transcript() dropped
    List[t.TensorType]
        train_transcript_responses - Same as input param transcript_responses, but with any that didn't work in .reward_from_transcript() dropped

    """

    
    composite_reward_list, task_score_list, language_score_list, penalisation_tensor_list = [], [], [], []
    language_score_list_no_failed, task_score_list_no_failed, composite_reward_list_no_failed = [], [], []
    train_query_tensors = []
    train_transcript_responses = []

    for i, prompt_data in tqdm(enumerate(batch_prompt_datas)):

        # Original model extraction failed - penalise if required
        if None in [
            prompt_data.extracted_cot,
            prompt_data.extracted_final_answer_with_cot
        ]:
            if skipping_failed_parsing_examples:
                continue
            else:
                composite_reward = -1.0
                task_score = 0.0
                language_score = 0.0
                if tokenizer:
                    penalisation_tensor = torch.zeros_like(query_tensors[i], dtype=torch.bool).to(torch.bfloat16)


        # Original model extraction succeeded!
        else:
            try:
                if tokenizer:
                    composite_reward, task_score, language_score, penalisation_tensor = train_dataset.reward_from_transcript(
                        prompt_data=prompt_data, 
                        skipping_failed_parsing_examples=skipping_failed_parsing_examples, 
                        t_weight=t_weight, 
                        l_weight= l_weight,
                        tokenizer=tokenizer
                    )
                    
                else:

                    composite_reward, task_score, language_score, _ = train_dataset.reward_from_transcript(
                        prompt_data=prompt_data, 
                        skipping_failed_parsing_examples=skipping_failed_parsing_examples, 
                        t_weight=t_weight, 
                        l_weight= l_weight,
                        
                    )
            # Extraction failed at overseer/assessor stage - do not penalise model
            except LLMTranscriptExtractionError as e:
                print(f"LLMTranscriptExtractionError occurred: {str(e)}")
                traceback.print_exc()
                continue
        if tokenizer:
            penalisation_tensor_list.append(penalisation_tensor)
        
        composite_reward_list.append(t.tensor(composite_reward))
        task_score_list.append(task_score)
        language_score_list.append(language_score)
        train_query_tensors.append(query_tensors[i])
        train_transcript_responses.append(transcript_responses[i])
        if not None in [
            prompt_data.extracted_cot,
            prompt_data.extracted_final_answer_with_cot
        ]:
            language_score_list_no_failed.append(language_score)
            task_score_list_no_failed.append(task_score)
            composite_reward_list_no_failed.append(composite_reward)


        prompt_data.task_score = task_score
        prompt_data.language_score = language_score
        prompt_data.composite_reward = composite_reward

    return composite_reward_list, task_score_list, language_score_list, train_query_tensors, train_transcript_responses, penalisation_tensor_list, task_score_list_no_failed, language_score_list_no_failed, composite_reward_list_no_failed


def evaluate_cot_gap_summary(
        ppo_trainer : PPOTrainer, 
        tokenizer : AutoTokenizer, 
        dataloader: DataLoader,
        system_prompt: str,
        model_class: BaseModel, 
        device: str, 
        generation_kwargs: Dict,
        output_delimitation_tokens: Dict[str, str],
        skipping_failed_parsing_examples: bool
    ) -> CoTGapSummary:
    
    """
    A function for evaluating the current CoT gap on the task; i.e, the diffence in task success with and without use of a Chain of Thought. 

    Parameters
    ----------
    ppo_trainer : PPOTrainer
        An already-initialised instance of trl's PPOTrainer, which will be used to generate responses, with and without CoT, for the whole dataset of task examples in the submitted dataloader. 
    tokenizer : AutoTokenizer
        transformers AutoTokenizer instance, used to encode and task prompts and decode responses
    dataloader : DataLoader
        torch dataloader, which yields BatchPromptDatas of task examples
    system_prompt : str
        System prompt used to generate responses from ppo_trainer 
    model_class : BaseModel
        The torch model wrapper being used; this function needs to access the 'batchize_conversation' method. 
    device : str
        torch string representation of device being used, e.g 'cuda:0'
    generation_kwargs : Dict
        The additional args submitted to ppo_trainer.generate() when generating
    output_delimitation_tokens : Dict[str, str]
        A dict containing the tokens which will be used to delineate parts of the model's responses. 
        For the main model being trained, which emits a scratchpad and final output, the keys should be:
        - "start_output_token"
        - "end_output_token"
        - "start_scratchpad_token"
        - "end_scratchpad_token"
        And the values should be an appropriately chosen token, e.g <SCRATCHPAD>, </SCRATCHPAD> etc.
    skipping_failed_parsing_examples: bool
        Skip examples that failed when parsing them for task score, language score, answer extraction, CoT extraction etc

    Returns
    -------
    CoTGapSummary
        An object storing summary statistics about the CoT gap. 

    """


    # Total examples provided:
    sum_examples = 0

    # Examples answered in the correct format:
    sum_extracted_with_cot = 0
    sum_extracted_without_cot = 0

    # Of those answered of the correct format...
    sum_reward_with_cot = 0
    sum_reward_without_cot = 0
    
    with t.no_grad():

        for batch_prompt_datas in tqdm(dataloader):

            batch_messages_without_cot = model_class.batchize_conversation(
                system_prompt, batch_prompt_datas.no_cot_prompts
            )

            batch_messages_with_cot = model_class.batchize_conversation(
                system_prompt, batch_prompt_datas.cot_prompts
            )

            batch_length = len(batch_messages_with_cot)
            batch_messages = batch_messages_with_cot + batch_messages_without_cot
            inputs = tokenizer.apply_chat_template(
                batch_messages,
                return_tensors="pt",
                add_generation_prompt=True,
                return_dict=True,
                padding=True,
                truncation=True,
            )

            # Move the tokenized inputs to the same device the model is on (GPU/CPU)
            inputs = {key: tensor.to(device) for key, tensor in inputs.items()}

            # Translate it into ppo.generate format which List[torch.tensors]
            query_tensors = []
            for input in inputs["input_ids"]:
                query_tensors.append(input)

            # Generate the with-CoT transcript (no-CoT transcript not needed during training)
            transcript_responses = ppo_trainer.generate(
                query_tensors, **generation_kwargs,
            )
            decoded_responses = [
                tokenizer.decode(r.squeeze()) for r in transcript_responses
            ]

            batch_prompt_datas.cot_transcripts = decoded_responses[:batch_length]
            batch_prompt_datas.no_cot_transcripts = decoded_responses[batch_length:]
            extracted_batch_prompt_datas_with_cot = extract_cots(batch_prompt_datas, output_delimitation_tokens, skipping_failed_parsing_examples)
            extracted_batch_prompt_datas_with_cot_and_without_cot = extracts_no_cots(extracted_batch_prompt_datas_with_cot, output_delimitation_tokens, skipping_failed_parsing_examples)
            extracted_batch_prompt_datas_with_na_with_cot = dataloader.dataset.neural_assessor_from_batch_prompt_data(extracted_batch_prompt_datas_with_cot_and_without_cot, from_cot_prompt = True)
            results_with_cot = [extract_floatingpoint_overseer_or_assessor_answer(prompt_data.extracted_assessor_answer) for prompt_data in extracted_batch_prompt_datas_with_na_with_cot]
            extracted_batch_prompt_datas_with_na_without_cot = dataloader.dataset.neural_assessor_from_batch_prompt_data(batch_prompt_datas, from_cot_prompt = False)
            results_without_cot = [extract_floatingpoint_overseer_or_assessor_answer(prompt_data.extracted_assessor_answer) for prompt_data in extracted_batch_prompt_datas_with_na_without_cot]

            cot_gap = CoTGapSummary(
                sum_examples = len(batch_prompt_datas),
                sum_extracted_with_cot = len(results_with_cot) ,
                sum_extracted_without_cot = len(results_without_cot),
                sum_reward_with_cot = sum(results_with_cot),
                sum_reward_without_cot = sum(results_without_cot),
            )

            print(cot_gap.summary)

    # cot_gap = (sum_reward_with_cot - sum_reward_without_cot)
    # cot_ratio = sum_reward_with_cot / (sum_reward_without_cot + t.finfo(t.float32).eps)

    return CoTGapSummary(
        sum_examples,
        sum_extracted_with_cot,
        sum_extracted_without_cot,
        sum_reward_with_cot,
        sum_reward_without_cot,
    )


def create_reference_model_under_deepspeed_R3(
    model_name: str, num_shared_layers: Optional[int] = None, pattern: Optional[str] = None, device: Optional[torch.device] = None
) -> PreTrainedModelWrapper:
    """
    Creates a static reference copy of a model underdeepseed ZERO Stage 3. Note that model will be in `.eval()` mode.

    Parameters
    ----------
        model_name: str
            The model to be copied.
        num_shared_layers: Optional[int] 
            The number of initial layers that are shared between both models and kept frozen.
        pattern: Optional[str]
            The shared layers are selected with a string pattern (e.g. "transformer.h.{layer}" for GPT2) and if a custom pattern is necessary it can be passed here.
        device: Optional[torch.device]
            The device on which to put the ref model

    Returns
    -------
        ref_model: PreTrainedModelWrapper
    """
    logging.info("Recruiting ref model under deepseed ZERO Stage 3")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32).to(device)
    parameter_names = [n for n, _ in ref_model.named_parameters()]

    # if no layers are shared, return copy of model
    if num_shared_layers is None:
        for param_name in parameter_names:
            param = ref_model.get_parameter(param_name)
            param.requires_grad = False
        return ref_model.eval()
    else:
        raise NotImplementedError("We need to think with accelerate framework, go on each GPU, and identify which one requires grad = False or not. OR have a function before that does the require grad m false (in load model for instance in BaseModel)")
        # identify layer name pattern
        if pattern is not None:
            pattern = pattern.format(layer=num_shared_layers)
        else:
            for pattern_candidate in LAYER_PATTERNS:
                pattern_candidate = pattern_candidate.format(layer=num_shared_layers)
                if any(pattern_candidate in name for name in parameter_names):
                    pattern = pattern_candidate
                    break

        if pattern is None:
            raise ValueError("Layer pattern could not be matched.")

        # divide parameters in shared and unshared parameter lists
        shared_param_list = []
        unshared_param_list = []

        shared_parameter = True
        for name, _param in model.named_parameters():
            if pattern in name:
                shared_parameter = False
            if shared_parameter:
                shared_param_list.append(name)
            else:
                unshared_param_list.append(name)

        # create reference of the original parameter if they are shared
        for param_name in shared_param_list:
            param = model.get_parameter(param_name)
            param.requires_grad = False

            _ref_param = ref_model.get_parameter(param_name)

        # for all other parameters just make sure they don't use gradients
        for param_name in unshared_param_list:
            param = ref_model.get_parameter(param_name)
            param.requires_grad = False

        if pattern is not None and len(unshared_param_list) == 0:
            logging.warning("Pattern passed or found, but no layers matched in the model. Check for a typo.")

        logging.info("Ref model recruited")
        return ref_model.eval()



 