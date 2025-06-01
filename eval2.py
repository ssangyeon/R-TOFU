import csv
import json
import os
import shutil
import warnings


import hydra
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from utils import get_model_identifiers_from_yaml
from test import rouge_answer_score
from test_cot import rouge_cot_forget_score, evaluate_with_gpt, compute_cosine_similarity_score

warnings.filterwarnings('ignore')


@hydra.main(version_base=None, config_path="config", config_name="forget")
def main(cfg):
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}

    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]

    task_list = os.getenv('TASK_LIST').split(',')
    task_list = [int(i) for i in task_list]
    cfg.save_dir = os.path.join(cfg.save_dir, os.getenv('TASK_LIST').replace(',', '-'))

    unlearn_times = task_list.index(cfg.task_id) + 1
    curr_save_dir = os.path.join(cfg.save_dir, f"unlearn_times_{unlearn_times}")
    curr_data_path = os.path.join(curr_save_dir, "task_data")
    if cfg.eval_unlearn_step == 0:
        curr_checkpoint_dir = cfg.model_path
    elif cfg.eval_unlearn_step == 1:            ##base model
        curr_checkpoint_dir = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    else:
        if not os.path.exists(curr_checkpoint_dir):
            print(f'{curr_checkpoint_dir} does not exist.')
            exit()

    curr_eval_dir = os.path.join(curr_save_dir, f'eval_results-{cfg.eval_unlearn_step}')
    if os.path.exists(os.path.join(curr_eval_dir, 'aggregate_stat.csv')):
        print(f'{curr_eval_dir} already evaluated.')
        exit()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(model_id)
    if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
        config.rope_scaling.setdefault("type", "linear")
    if cfg.use_LoRA:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            config=config,
            attn_implementation='flash_attention_2',
            torch_dtype=torch.bfloat16,
            device_map=device_map
        )
        model = PeftModel.from_pretrained(model, curr_checkpoint_dir)
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            curr_checkpoint_dir,
            config=config,
            attn_implementation='flash_attention_2',
            torch_dtype=torch.bfloat16,
            device_map=device_map
        )
    model = model.eval()


    rouge_answer_score(cfg, unlearn_times, model, tokenizer)
    rouge_cot_forget_score(cfg, unlearn_times, model, tokenizer)
    evaluate_with_gpt(cfg,unlearn_times)
    compute_cosine_similarity_score(cfg,unlearn_times)

    if unlearn_times == len(task_list) and not cfg.save_checkpoint:
        if (os.path.exists(curr_checkpoint_dir)) and (cfg.eval_unlearn_step != 0):
            shutil.rmtree(curr_checkpoint_dir)



if __name__ == "__main__":
    main()