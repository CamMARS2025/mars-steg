from __future__ import annotations


from typing import Dict, List
import torch

from mars_steg.model import BaseModel

from mars_steg.utils.answer_extraction import extract_neural_assessor_answers

from mars_steg.utils.prompt_data import BatchPromptData
from mars_steg.config import PromptConfig



class ProblemSolutionAssessor:
    """
    A neural assessor for math tasks
    """
    
    def __init__(
        self, 
        model: BaseModel,
        prompt_config: PromptConfig
    ):
        self.whitebox_model = model
        self.whitebox_model.model.eval()

        self.prompt_config = prompt_config
        
        # TODO: move to config
        self.user_prompt = prompt_config.neural_assessor_user_prompt_template
        self.system_prompt = prompt_config.neural_assessor_system_prompt

    def convert_batch_prompt_into_conversation_template(
        self,
        batch_prompt_data: BatchPromptData,
        true_answers: List[str],
        with_cot: bool
        ) -> List[List[Dict[str, str]]]:

        assessor_prompts = []

        for pd, ta in zip(batch_prompt_data, true_answers):
            
            assessor_prompts.append(
                self.user_prompt.format(
                    # agent_prompt = pd.cot_prompt if with_cot else pd.no_cot_prompt,
                    llm_answer = pd.extracted_final_answer_with_cot if with_cot else pd.extracted_final_answer_without_cot, 
                    true_answer = ta
                )
            )

        batch_prompt_data.assessor_prompts = assessor_prompts

        return self.whitebox_model.batchize_conversation(assessor_prompts, self.system_prompt)

    @torch.no_grad()
    def get_assessor_generated_answers(
        self,
        batch_prompt_data: BatchPromptData,
        true_answers: List[str],
        with_cot: bool,
        neural_assessor_thinking_helper: str = None
    ) -> BatchPromptData:
        """
        Imbue all members of batch_prompt_data with assessor_transcripts in place,
            but only return the ones for which extraction of extracted_assessor_answer was successful

        Unsuccessful ones will still be available due to inplace operation
        """
        if len(batch_prompt_data) == 0:
            return batch_prompt_data

        # Convert batch_prompt_data into a conversation template
        conversations_list = self.convert_batch_prompt_into_conversation_template(
            batch_prompt_data,
            true_answers=true_answers,
            with_cot=with_cot
        )

        decoded_responses = self.whitebox_model.full_generate(
            conversations_list, 
            is_neural_assessor = True, 
            neural_assessor_thinking_helper = neural_assessor_thinking_helper)

        batch_prompt_data.assessor_transcripts = decoded_responses

        # TODO: pass output_delimitation_tokens to here!
        pruned_batch_prompt_data = extract_neural_assessor_answers(batch_prompt_data, self.whitebox_model.output_delimitation_tokens)
        
        return pruned_batch_prompt_data




    




