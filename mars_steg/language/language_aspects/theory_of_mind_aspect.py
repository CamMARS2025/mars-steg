from typing import List, Optional
import yaml
import re
import torch
import random
from mars_steg.language.base_language_aspect import LanguageAspect
from transformers import AutoTokenizer



class ToMTokenBanTask(LanguageAspect):

    """
    temporal_reward_penalisation_offset is used to offset the effect of kl regularisation on the length of CoT
    """

    compatible_tasks = {"TheoryOfMindTask"}

    uses_local_neural_overseer = False

    def __init__(self, chance_penalisation: float, nouns_path: str, nouns_penalisation_type: str, max_freq=50, temporal_reward_penalisation_offset: Optional[float] = None, **kwargs):
        super().__init__()
        self.chance_penalisation = chance_penalisation
        self.nouns_path = nouns_path
        self.nouns_penalisation_type = nouns_penalisation_type
        with open(nouns_path, 'r') as f:
            nouns = yaml.safe_load(f)
        self.penalise_substrings = set(nouns[nouns_penalisation_type])

        self.max_freq = max_freq
        if temporal_reward_penalisation_offset is not None:
            self.temporal_reward_penalisation_offset = torch.tensor(temporal_reward_penalisation_offset).to(torch.bfloat16)
        else:
            self.temporal_reward_penalisation_offset = None

    def limit_penalise_substrings(self, new_penalise_substrings: List[str]):
        new_instance = ToMTokenBanTask(
            chance_penalisation=self.chance_penalisation,
            nouns_path=self.nouns_path,
            nouns_penalisation_type=self.nouns_penalisation_type,
            max_freq=self.max_freq,
            temporal_reward_penalisation_offset = self.temporal_reward_penalisation_offset
        )
        new_penalise_substrings = set(new_penalise_substrings)
        assert new_instance.penalise_substrings.issuperset(new_penalise_substrings), \
            "Trying to override ToMTokenBanTask.penalise_substrings with unseen before penalisation strings!"
        new_instance.penalise_substrings = new_penalise_substrings

        return new_instance

    def create_penalisation_tensor(self, token_ids:torch.Tensor, lists_of_tokens_to_penalize: List[torch.Tensor])-> torch.Tensor:

        token_ids = torch.tensor(token_ids).to("cpu")
        penalisation_tensor = torch.zeros_like(token_ids, dtype=torch.bool)
        lists_of_tokens_to_penalize = [t.to("cpu") for t in lists_of_tokens_to_penalize]
        for list_of_tokens_to_penalize in lists_of_tokens_to_penalize:

            if random.random() > self.chance_penalisation:
                # Only penalize with probability chance_penalised
                continue

            # Get sequence length
            seq_len = list_of_tokens_to_penalize.shape[0]

            # Use unfold to create a rolling window view (efficiently without extra memory copies)
            windows = token_ids.unfold(0, seq_len, 1)  # Shape: [num_windows, seq_len]

            # Compare each window with the target sequence
            matches = (windows == list_of_tokens_to_penalize).all(dim=1)
            start_indices = matches.nonzero(as_tuple=True)[0]

            # Create a mask for penalization
            penalized_one_sequence_token_tensor = torch.zeros_like(token_ids, dtype=torch.bool)

            # Penalize entire matched windows
            for idx in start_indices:
                penalized_one_sequence_token_tensor[idx: idx + seq_len] = True  # Set mask for all elements in the window
            penalisation_tensor+=penalized_one_sequence_token_tensor
        return penalisation_tensor

    def get_language_score(self, prompt_data):

        if self.chance_penalisation == 1.0:
            raise ValueError("chance_penalise cannot be 1.0")

        # Count equals signs in cot
        cot = prompt_data.extracted_cot

        # TODO: 2025.03.29 - make string matches work with multiple strings
        n_string_matches = 0
        for penalise_substring in self.penalise_substrings:
            n_string_matches += cot.count(penalise_substring)

        # We want to drive down the number of equals signs
        # We will punish based on use up to a maximum number of times
        n_string_matches = min(n_string_matches, self.max_freq)
        language_penality = float(n_string_matches / self.max_freq)
        return  1 - language_penality

    def get_language_score_temporal_reward(self, prompt_data, tokenizer: Optional[AutoTokenizer] = None):

        # Count equals signs in cot

        if self.chance_penalisation == 1.0:
            raise ValueError("chance_penalise cannot be 1.0")
        
        cot = prompt_data.extracted_cot

        if tokenizer is not None:
            
            assert self.temporal_reward_penalisation_offset is not None

            tokenized = tokenizer(cot)
            tokenized_cot = tokenized['input_ids']
            matches_token = [re.findall(keyword, cot) for keyword in self.penalise_substrings]
            list_word_penalise = []
            for words_to_penalise in matches_token:
                list_word_penalise+=words_to_penalise
            list_word_penalise = list(set(list_word_penalise))

            special_tokens = list(tokenizer.all_special_ids)
            token_to_penalise = [tokenizer(" "+ word)['input_ids'] for word in list_word_penalise]
            token_to_penalise = [torch.tensor([token for token in list_token if token not in special_tokens]).to("cuda") for list_token in token_to_penalise]

            penalisation_tensor = self.create_penalisation_tensor(
                token_ids=tokenized_cot, 
                lists_of_tokens_to_penalize=token_to_penalise
            ).to(torch.bfloat16) - self.temporal_reward_penalisation_offset # Offset removed = small reward for each token
        
        n_string_matches = 0
        for penalise_substring in self.penalise_substrings:
            n_string_matches += cot.count(penalise_substring)

        n_string_matches = min(n_string_matches, self.max_freq)
        language_penality = float(n_string_matches / self.max_freq)

        if tokenizer is not None:
            return 1 - language_penality, penalisation_tensor
        else:
            return 1 - language_penality


    