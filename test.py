import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import logging
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from tqdm.contrib import tzip
from typing import List, Dict
from rouge_score import rouge_scorer


def apply_think_strategy(prompt, strategy="DefaultCoT"):
    """Modify prompt based on Think strategy."""
    if strategy == "ZeroThink":
        return f"<｜User｜>{prompt}<｜Assistant｜><think>\n\n</think>\n\n"
    elif strategy == "LessThink":
        return f"<｜User｜>{prompt}<｜Assistant｜><think>\nOkay, the user asked this, I can answer it without thinking much.\n</think>\n\n"
    else:
        raise ValueError("Invalid Think strategy. Choose from ['ZeroThink', 'LessThink'].")

def generate_response(prompt, model, tokenizer, strategy="DefaultCoT", max_tokens=2048):
    """Generate response with a specific Think strategy"""
    modified_prompt = apply_think_strategy(prompt, strategy=strategy)

    inputs = tokenizer(modified_prompt, return_tensors="pt", padding=True, truncation=True, return_attention_mask=True)
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")

    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_tokens,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
        temperature=1.0,
    )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    if "<think>" in output_text:
        cot_start = output_text.find("<think>")
        cot_end = output_text.find("</think>") if "</think>" in output_text else len(output_text)
        cot = output_text[cot_start:cot_end].strip()
        answer = output_text[cot_end+len("</think>"):].strip() if "</think>" in output_text else "No explicit answer found."
    else:
        cot = "No explicit COT found."
        answer = output_text.strip()

    return cot, answer



def rouge_answer_score(cfg, unlearn_times, model, tokenizer):

    st_model = SentenceTransformer("all-MiniLM-L6-v2", device=torch.device('cuda'))

    input_file = f"data/tofu/{cfg.split}.json"
    curr_save_dir = os.path.join(cfg.save_dir, f"unlearn_times_{unlearn_times}")
    os.makedirs(curr_save_dir, exist_ok=True)

    think_strategies = ["ZeroThink", "LessThink"]

    for strategy in think_strategies:
        curr_eval_dir = os.path.join(curr_save_dir, f"eval_results-{cfg.eval_unlearn_step}")
        os.makedirs(curr_eval_dir, exist_ok=True)
        
        output_file = os.path.join(curr_eval_dir, f'{strategy}_answer_rouge_score.json')

        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
        rougeL_recall_scores = []
        cosine_sims = []

        with open(input_file, "r", encoding="utf-8") as f, open(output_file, "w", encoding="utf-8") as out_f:
            for line in f:
                entry = json.loads(line)
                
                if entry["task_id"] != "1":
                    continue

                question = entry["question"]
                answer = entry["answer"]
                
                _, response = generate_response(question, model, tokenizer, strategy=strategy)
                
                
                scores = scorer.score(answer, response)
                rougeL_recall = scores["rougeL"].recall
                rougeL_recall_scores.append(rougeL_recall)

                answer_emb = st_model.encode(answer, convert_to_tensor=True)
                response_emb = st_model.encode(response, convert_to_tensor=True)
                cosine_sim = F.cosine_similarity(answer_emb, response_emb, dim=0).item()
                cosine_sims.append(cosine_sim)

                result = {
                    "rougeL_recall": rougeL_recall,
                    "cosine_sim": cosine_sim,
                    "answer": answer,
                    "response": response,
                    "strategy": strategy
                }
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()

            total_count = len(rougeL_recall_scores)
            if total_count > 0:
                avg_rougeL_recall = sum(rougeL_recall_scores) / total_count
                avg_cosine = sum(cosine_sims) / total_count
            else:
                avg_rougeL_recall = 0.0
                avg_cosine = 0.0

            avg_result = {
                "average_rougeL_recall": avg_rougeL_recall,
                "average_cosine_sim": avg_cosine,
                "total_entries": total_count,
                "strategy": strategy
            }
            out_f.write(json.dumps(avg_result, ensure_ascii=False) + "\n")
