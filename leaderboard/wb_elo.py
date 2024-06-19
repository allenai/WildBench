from leaderboard import data_utils
import json 
import random
from collections import defaultdict
from tqdm import tqdm   
import fire 
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import numpy as np
import os
from datetime import datetime

if data_utils.eval_results is None:
    data_utils.load_eval_results()

# eval_results, score_eval_results = load_eval_results()

all_scores_by_id_model = {}
all_outputs_by_id_model = {}

def load_scores():
    global all_scores_by_id_model
    if data_utils.score_eval_results is None:
        data_utils.load_eval_results()
    for model_name, model_eval_data in data_utils.score_eval_results.items():
        for item in model_eval_data:
            session_id = item["session_id"]
            score = item["score"]
            if session_id not in all_scores_by_id_model:
                all_scores_by_id_model[session_id] = {}
                all_outputs_by_id_model[session_id] = {}
            all_scores_by_id_model[session_id][model_name] = int(score)
            all_outputs_by_id_model[session_id][model_name] = item["model_output"].strip()
    return 

def get_all_votes(margin=2, tie_margin=1):
    global all_scores_by_id_model 
    votes = []
    for session_id, scores_by_model in all_scores_by_id_model.items():
        for model_1, score_1 in scores_by_model.items():
            for model_2, score_2 in scores_by_model.items():
                if model_1 == model_2:
                    continue
                vote_item = {}
                vote_item["session_id"] = session_id
                vote_item["model_1"] = model_1
                vote_item["model_2"] = model_2
                vote_item["score_1"] = score_1
                vote_item["score_2"] = score_2
                # decide the empty and truncated 
                model_1_output =  all_outputs_by_id_model[session_id][model_1]
                model_2_output =  all_outputs_by_id_model[session_id][model_2]
                if len(model_1_output) == 0 or len(model_2_output) == 0:
                    continue
                if model_1_output.endswith("... (truncated)") or model_2_output.endswith("... (truncated)"):
                    continue
                if score_1 > score_2 and score_1 - score_2 >= margin:
                    vote_item["winner"] = model_1
                elif score_2 > score_1 and score_2 - score_1 >= margin:
                    vote_item["winner"] = model_2
                else:
                    if abs(score_1 - score_2) <= tie_margin:
                        vote_item["winner"] = "tie"
                    else:
                        continue
                votes.append(vote_item)
    return votes 
 
def compute_single_round(votes, K, init_elos, dynamic):
    elo = init_elos.copy() if init_elos is not None else {}
    sample_votes = [random.choice(votes) for _ in range(len(votes))]

    # Initialize Elo ratings
    for vote in sample_votes:
        if vote["model_1"] not in elo:
            elo[vote["model_1"]] = 1000
        if vote["model_2"] not in elo:
            elo[vote["model_2"]] = 1000

    vote_update_cnt = defaultdict(int)
    # Calculate Elo ratings for the bootstrap sample
    for vote in sample_votes:
        model_1 = vote["model_1"]
        model_2 = vote["model_2"]
        if model_1 in init_elos and model_2 in init_elos:
            continue

        elo_1 = elo[model_1]
        elo_2 = elo[model_2]

        expected_1 = 1 / (1 + 10 ** ((elo_2 - elo_1) / 400))
        expected_2 = 1 / (1 + 10 ** ((elo_1 - elo_2) / 400))

        if vote["winner"] == model_1:
            score_1 = 1
            score_2 = 0
        elif vote["winner"] == model_2:
            score_1 = 0
            score_2 = 1
        else:
            score_1 = 0.5
            score_2 = 0.5

        if model_1 not in init_elos:
            elo[model_1] += K * (score_1 - expected_1)
        else:
            if dynamic:
                elo[model_1] += K * (score_1 - expected_1)
                if vote_update_cnt[model_1] % 5 == 0:
                    elo[model_1] = (elo[model_1] + init_elos[model_1]) / 2

        if model_2 not in init_elos:
            elo[model_2] += K * (score_2 - expected_2)
        else:
            if dynamic:
                elo[model_2] += K * (score_2 - expected_2)
                if vote_update_cnt[model_2] % 5 == 0:
                    elo[model_2] = (elo[model_2] + init_elos[model_2]) / 2

        vote_update_cnt[model_1] += 1
        vote_update_cnt[model_2] += 1

    return elo

def compute_elo_based_on_votes(votes, K=4, num_rounds=1000, init_elos=None, dynamic=False, num_processes=None):
    """
    Compute Elo rating based on votes with bootstrapping method using multiprocessing.
    """
    elo_cumulative = defaultdict(list)
    num_models = defaultdict(int)

    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(compute_single_round, votes, K, init_elos, dynamic) for _ in range(num_rounds)]
        for future in tqdm(as_completed(futures), total=num_rounds, desc="Computing WB-Elo"):
            elo = future.result()
            for model, rating in elo.items():
                elo_cumulative[model].append(rating)
                num_models[model] += 1

    elo_avg = {model: sum(ratings) / num_models[model] for model, ratings in elo_cumulative.items()}
    elo_std = {model: (sum((rating - elo_avg[model]) ** 2 for rating in ratings) / num_models[model]) ** 0.5 for model, ratings in elo_cumulative.items()}
    elo_ci_lower = {}
    elo_ci_upper = {}
    for model, ratings in elo_cumulative.items():
        ci_lower = np.percentile(ratings, 2.5)
        ci_upper = np.percentile(ratings, 97.5)
        elo_ci_lower[model] = ci_lower
        elo_ci_upper[model] = ci_upper

    elo_ci = {model: (elo_ci_lower[model], elo_ci_upper[model]) for model in elo_avg.keys()}
    elo_median = {model: np.median(ratings) for model, ratings in elo_cumulative.items()}
    return elo_avg, elo_std, elo_median, elo_ci

def load_init_elo(filepath = "leaderboard/data_dir/all_stat.json", elo_key = "Arena Elo (hard-en) - latest"):
    init_elos = {} 
    with open(filepath, "r") as f:
        data = json.load(f)
        for model in data:
            model = model.replace("@together", "")
            elo = data[model].get(elo_key, "-")
            if elo != "-":
                init_elos[model] = float(elo)
    return init_elos


def compute_wb_elo(loo=-1, seed=42, margin=2, K=4, num_rounds=10, tie_margin=1, dynamic=False): 
    global all_scores_by_id_model

    random.seed(seed) 
    init_elos = load_init_elo() 

    if all_scores_by_id_model == {}:
        load_scores()
    
    
    print(f">>> Config: WB Elo with K={K} and num_rounds={num_rounds}; margin={margin}; loo={loo}; seed={seed}; init_elo={len(init_elos)} models; tie_margin={tie_margin}; dynamic={dynamic};")
 
    
    votes = get_all_votes(margin, tie_margin)
    print(f">>> Found {len(votes)} votes")
    # non-tie votes
    non_tie_votes = [item for item in votes if item["winner"] != "tie"]
    print(f">>> Found {len(non_tie_votes)} non-tie votes")

    not_useful_votes = []
    for v in votes:
        if v["model_1"] in init_elos and v["model_2"] in init_elos:
            not_useful_votes.append(v)
    # print(f">>> Found {len(not_useful_votes)} votes that are not useful for WB Elo")

    elo_avg, elo_std, elo_median, elo_ci = compute_elo_based_on_votes(votes, K=K, num_rounds=num_rounds, init_elos=init_elos, dynamic=dynamic)
    # rank by elo
    elo_stat = {k: {"avg": v, "std": elo_std[k], \
                     "median": elo_median[k], "ci": elo_ci[k],
                      "init_elo": init_elos.get(k, '-')} \
                for k, v in sorted(elo_avg.items(), key=lambda item: item[1], reverse=True)}
    print(f">>> WB Elo with K={K} and num_rounds={num_rounds}")
    # print(json.dumps(elo_stat, indent=4))
 
    elo_results = {
        "config": {
            "K": K,
            "num_rounds": num_rounds,
            "margin": margin,
            "tie_margin": tie_margin,
            "dynamic": dynamic,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
        },
        "elo_stat": elo_stat
    }
    with open(f"leaderboard/data_dir/wb_elo_results.json", "w") as f:
        json.dump(elo_results, f, indent=4)

if __name__ == "__main__":
    fire.Fire(compute_wb_elo)

"""
FOLDER="tmp_loo_exp_v10"
mkdir ${FOLDER}
margin=3
tie_margin=1
K=4
dynamic=True
python -m analysis_scripts.wb_elo --loo -1 --K $K --margin $margin --tie_margin $tie_margin --num_rounds 100 --dynamic $dynamic > ./${FOLDER}/wb_elo.txt &

for i in {0..37}
do
    python -m analysis_scripts.wb_elo --loo $i --K $K --margin $margin --tie_margin $tie_margin --num_rounds 5  --dynamic $dynamic > ./${FOLDER}/wb_elo_loo_$i.txt &
done
"""