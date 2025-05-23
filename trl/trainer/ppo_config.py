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

import os
from dataclasses import dataclass

from ..trainer.utils import OnPolicyConfig


@dataclass
class PPOConfig(OnPolicyConfig):
    r"""
    Configuration class for the [`PPOTrainer`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        exp_name (`str`, *optional*, defaults to `os.path.basename(__file__)[:-3]`):
            Name of this experiment.
        reward_model_path (`str`, *optional*, defaults to `"EleutherAI/pythia-160m"`):
            Path to the reward model.
        num_ppo_epochs (`int`, *optional*, defaults to `4`):
            Number of epochs to train.
        whiten_rewards (`bool`, *optional*, defaults to `False`):
            Whether to whiten the rewards.
        kl_coef (`float`, *optional*, defaults to `0.05`):
            KL coefficient.
        cliprange (`float`, *optional*, defaults to `0.2`):
            Clip range.
        vf_coef (`float`, *optional*, defaults to `0.1`):
            Value function coefficient.
        cliprange_value (`float`, *optional*, defaults to `0.2`):
            Clip range for the value function.
        gamma (`float`, *optional*, defaults to `1.0`):
            Discount factor.
        lam (`float`, *optional*, defaults to `0.95`):
            Lambda value for GAE.
    """

    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    reward_model_path: str = "EleutherAI/pythia-160m"
    num_ppo_epochs: int = 4
    whiten_rewards: bool = False
    kl_coef: float = 0.05
    cliprange: float = 0.2
    vf_coef: float = 0.1
    cliprange_value: float = 0.2
    gamma: float = 1.0
    lam: float = 0.95
    gamma1: float = (
        0.1  # discount factor for the estimated returns of the newly searched paper
    )
    rounds: int = 2  # rollout rounds
    use_vm: bool = True  # use value model to estimate the returns of the new papers
    use_selector: bool = True
    paper_db: str = ""
    paper_id: str = ""
    search_select_score: float = 0.5
    expand_select_score: float = 0.5
    search_cost: float = 0.2
    expand_cost: float = 0.2
    max_papers: int = 5
    value_step: int = (
        5  # To avoid wasting computing resources, it is best for value_step to be a factor of max_papers
    )
    value_max_length: int = (
        992  # Set the maximum length of the value model input to avoid OOM.
    )
    warm_up_step: int = 50
    lora_path: str = ""
    model_adapter_name: str = "default"
    ref_adapter_name: str = ""
