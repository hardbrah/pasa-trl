# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import shutil

from accelerate import PartialState
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    AutoConfig,
    Qwen2ForSequenceClassification,
    Qwen2Model,
)

from trl import ModelConfig, PPOConfig, PPOTrainer, ScriptArguments
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from custom_agent.agent_dataset import AgentDataset
from typing import Optional
import torch.nn as nn
import torch
from peft import PeftModel


class FixZero3CheckpointPPOTrainer(PPOTrainer):

    def save_model(
        self, output_dir: Optional[str] = None, _internal_call: bool = False
    ):
        backup_model = self.model
        self.model = self.model.policy  # save only the policy

        Trainer.save_model(self, output_dir, _internal_call)

        self.model = backup_model

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if self.is_deepspeed_enabled:
            state_dict = {
                name.removeprefix("policy."): param
                for name, param in state_dict.items()
                if name.startswith("policy.")
            }

        super()._save(output_dir, state_dict)


class CustomQwen2ForSequenceClassification(Qwen2ForSequenceClassification):
    def __init__(self, lora_path, config) -> None:
        super().__init__(config)
        self.num_labels = 1
        # model = Qwen2Model(config).to(dtype=torch.bfloat16)
        self.model = PeftModel.from_pretrained(
            self.model, lora_path, is_trainable=True, torch_dtype=torch.bfloat16
        )
        self.score = nn.Sequential(
            nn.Linear(config.hidden_size, 384, bias=False),
            nn.Linear(384, self.num_labels, bias=False),
        )

        self.post_init()


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_into_dataclasses()
    shutil.rmtree(training_args.output_dir, ignore_errors=True)

    # tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        padding_side="left",
        trust_remote_code=model_config.trust_remote_code,
    )
    train_dataset = AgentDataset(script_args.dataset_name, tokenizer)
    assert (
        train_dataset[0]["input_ids"][-1] != tokenizer.eos_token_id
    ), "The last token should not be an EOS token"

    # models
    config = AutoConfig.from_pretrained(
        training_args.reward_model_path,
        num_labels=1,
        trust_remote_code=model_config.trust_remote_code,
    )
    value_model = CustomQwen2ForSequenceClassification(
        lora_path=training_args.lora_path, config=config
    )
    for m in value_model.score.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=0.01)
    ref_policy = None
    policy_base = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch.bfloat16,
    )
    policy = PeftModel.from_pretrained(
        policy_base,
        adapter_name=training_args.model_adapter_name,
        torch_dtype=torch.bfloat16,
        model_id=training_args.lora_path,
        is_trainable=True,
    )
    policy.load_adapter(
        training_args.lora_path,
        adapter_name=training_args.ref_adapter_name,
        is_trainable=False,
        torch_dtype=torch.bfloat16,
    )
    policy = policy.to(dtype=torch.bfloat16)
    value_model = value_model.to(dtype=torch.bfloat16)
    if ref_policy is not None:
        ref_policy = ref_policy.to(dtype=torch.bfloat16)

    trainer = FixZero3CheckpointPPOTrainer(
        config=training_args,
        processing_class=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        value_model=value_model,
        train_dataset=train_dataset,
        paper_db=training_args.paper_db,
        paper_id=training_args.paper_id,
    )
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
