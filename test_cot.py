import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import logging
from rouge_score import rouge_scorer
import shutil
import openai
import os, json, re
from tqdm import tqdm
from bert_score import score as bertscore


def gpt_fn(prompt, model="gpt-4o", temperature=0.0):
    client = openai.OpenAI(api_key="") ###insert your api key.
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def compute_cosine_similarity_score(cfg, unlearn_times):
    curr_save_dir = os.path.join(cfg.save_dir, f"unlearn_times_{unlearn_times}")
    curr_eval_dir = os.path.join(curr_save_dir, f'eval_results-{cfg.eval_unlearn_step}')
    
    think_strategies = ["DefaultCoT"]
    st_model = SentenceTransformer("all-MiniLM-L6-v2", device=torch.device('cuda'))
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    for strategy in think_strategies:
        input_path = os.path.join(curr_eval_dir, f'cot_rouge_forget_score_{strategy}.json')
        output_path = os.path.join(curr_eval_dir, f'cot_cosine_forget_score_{strategy}.json')

        if not os.path.exists(input_path):
            print(f"[WARNING] {input_path} does not exist. Skipping {strategy}...")
            continue

        with open(input_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        all_avg_list_cosine = []
        all_avg_list_rougeL_recall = []

        with open(output_path, "w", encoding="utf-8") as out_f:
            import nltk
            from nltk.tokenize.punkt import PunktSentenceTokenizer
            tokenizer = PunktSentenceTokenizer() 
            for line in tqdm(lines, desc=f"Cosine+ROUGE-L Recall {strategy}"):
                data = json.loads(line)
                if "average_rougeL_recall" in data:
                    continue    
                if "average_rougeL" in data:
                    continue

                cot1 = data.get("cot_answer", "")
                cot2 = data.get("generated_cot", "")

                steps1 = tokenizer.tokenize(cot1)
                steps2 = tokenizer.tokenize(cot2)

                emb1 = st_model.encode(steps1, convert_to_tensor=True).to("cuda")
                emb2 = st_model.encode(steps2, convert_to_tensor=True).to("cuda")
                cosine_matrix = util.pytorch_cos_sim(emb1, emb2)

                row_max_scores_cosine = cosine_matrix.max(dim=1).values.tolist()

                all_avg_cosine = sum(row_max_scores_cosine) / len(row_max_scores_cosine) if row_max_scores_cosine else 0.0
                all_avg_list_cosine.append(all_avg_cosine)

                topk_scores_dict = {}

                row_max_scores_rougeL_recall = []

                for ref_step in steps1:
                    recall_scores = []

                    for gen_step in steps2:
                        scores = scorer.score(ref_step, gen_step)["rougeL"]
                        recall_scores.append(scores.recall)

                    row_max_scores_rougeL_recall.append(max(recall_scores) if recall_scores else 0.0)


                avg_rougeL_recall = sum(row_max_scores_rougeL_recall) / len(row_max_scores_rougeL_recall) if row_max_scores_rougeL_recall else 0.0
                all_avg_list_rougeL_recall.append(avg_rougeL_recall)


                result = {
                    "strategy": strategy,
                    "all_pair_avg_cosine": all_avg_cosine,
                    "all_pair_avg_rougeL_recall": avg_rougeL_recall,
                    **topk_scores_dict,
                    "cot_answer_steps": steps1,
                    "generated_cot_steps": steps2
                }
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()

            def safe_mean(lst): return sum(lst)/len(lst) if lst else 0.0

            final_result = {
                "strategy": strategy,
                "total_evaluated": len(all_avg_list_cosine),
                "average_all_pair_cosine": safe_mean(all_avg_list_cosine),
                "average_all_pair_rougeL_recall": safe_mean(all_avg_list_rougeL_recall),
            }
            out_f.write(json.dumps(final_result, ensure_ascii=False) + "\n")
            

def evaluate_with_gpt(cfg, unlearn_times):

    curr_save_dir = os.path.join(cfg.save_dir, f"unlearn_times_{unlearn_times}")
    curr_eval_dir = os.path.join(curr_save_dir, f'eval_results-{cfg.eval_unlearn_step}')

    think_strategies = ["DefaultCoT"]

    for strategy in think_strategies:
        input_path = os.path.join(curr_eval_dir, f'cot_rouge_forget_score_{strategy}.json')
        output_path = os.path.join(curr_eval_dir, f'cot_gpteval_forget_score_{strategy}.json')

        if not os.path.exists(input_path):
            print(f"[WARNING] {input_path} does not exist. Skipping {strategy}...")
            continue

        score_list = []

        with open(input_path, "r", encoding="utf-8") as f, open(output_path, "w", encoding="utf-8") as out_f:
            lines = f.readlines()

            for line in tqdm(lines, desc=f"GPT eval {strategy}"):
                data = json.loads(line)
                if "average_rougeL_recall" in data:
                    continue    
                if "average_rougeL_f1_cot" in data:
                    continue
                if "average_rouge1" in data:
                    continue

                question = data.get("question", "")
                answer = data.get("answer", "")
                cot_after = data.get("generated_cot", "")

                prompt = f"""
You are evaluating whether a language model has successfully forgotten a specific piece of knowledge (A) based on its generated reasoning trace (Chain of Thought).

## Information:
- Forgotten Knowledge (A): {answer}
- Question: {question}
- Generated CoT (after unlearning): {cot_after}

## Evaluation Instructions:

Rate how much the generated CoT retains, reconstructs, or hints at** the forgotten knowledge (A), either directly or indirectly.

Respond with a single integer score from 0.00 to 1.00 (with two decimal places. For example, 0.23, 0.19, 0.74, 0.00, etc.) only:
- 0.00 = Completely forgotten (no trace or reconstruction of A)
- 1.00 = Fully retained or reconstructed (A is clearly present)

Do not include any explanation or justification. Respond only with the score.
"""
                response = gpt_fn(prompt) 
                match = re.search(r"\d+\.\d{2}", response)
                score = float(match.group()) if match else -1.0
                score_list.append(score)

                result = {
                    "strategy": strategy,
                    "forgetting_score": score,
                    "question": question,
                    "answer": answer,
                    "generated_cot": cot_after
                }
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()
            valid_scores = [s for s in score_list if s >= 0]
            avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
            avg_result = {
                "strategy": strategy,
                "average_forgetting_score": avg_score,
                "total_evaluated": len(valid_scores)
            }
            out_f.write(json.dumps(avg_result, ensure_ascii=False) + "\n")



def rouge_cot_forget_score(cfg, unlearn_times, model, tokenizer):
    input_file = f"data/tofu/{cfg.split}.json"
    curr_save_dir = os.path.join(cfg.save_dir, f"unlearn_times_{unlearn_times}")
    curr_eval_dir = os.path.join(curr_save_dir, f'eval_results-{cfg.eval_unlearn_step}')
    os.makedirs(curr_eval_dir, exist_ok=True)

    think_strategies = ["DefaultCoT"]
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for strategy in think_strategies:
        output_file = os.path.join(curr_eval_dir, f'cot_rouge_forget_score_{strategy}.json')
        rougeL_recall_scores = []
        with open(output_file, "w", encoding="utf-8") as out_f:
            for line in lines:
                entry = json.loads(line)
                if entry["task_id"] != "1":
                    continue

                question = entry["question"]
                cot_answer = entry["cot"]  

                generated_cot = generate_response(
                    question, model, tokenizer, strategy=strategy
                )

                scores = scorer.score(cot_answer, generated_cot)
                rougeL_recall = scores["rougeL"].recall
                rougeL_recall_scores.append(rougeL_recall)


                result = {
                    "strategy": strategy,
                    "rougeL_recall": rougeL_recall,
                    "question": question,
                    "cot_answer": cot_answer,
                    "generated_cot": generated_cot
                }

                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()
            avg_rougeL_recall = (
                sum(rougeL_recall_scores) / len(rougeL_recall_scores)
                if rougeL_recall_scores else 0.0
            )
            avg_result = {
                "average_rougeL_f1_cot": avg_rougeL_recall,
                "total_entries_cot": len(rougeL_recall_scores),
            }
            out_f.write(json.dumps(avg_result, ensure_ascii=False) + "\n")


def apply_think_strategy(prompt, strategy="DefaultCoT"):
    if strategy == "DefaultCoT":
        return f"<｜User｜>{prompt}<｜Assistant｜><think>\n"
    else:
        raise ValueError("Invalid Think strategy. Choose from ['DefaultCoT'].")

def extract_cot(response):
    if "<think>" in response:
        think_start = response.find("<think>") + len("<think>")
        think_end = response.find("</think>") if "</think>" in response else len(response)
        return response[think_start:think_end].strip()
    return ""

def generate_response(prompt, model, tokenizer, strategy="DefaultCoT", max_tokens=2048):
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
    else:
        cot = "No explicit COT found."
    return cot