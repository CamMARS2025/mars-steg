from mars_steg.model.base_model import BaseModel
from typing import Dict


class GPT2Model(BaseModel):

    def get_message(self, user_string: str, system_prompt: str) -> Dict:
        return {"role": user_string, "content": f"{system_prompt}"}
