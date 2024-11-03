import json
import os
from pathlib import Path
from tqdm import tqdm
from unified_utils import retry_handler, openai_chat_request

def load_results(result_file):
    """Load results from a JSON file."""
    with open(result_file, 'r') as f:
        return json.load(f)

def load_template(template_file):
    """Load evaluation template from a file."""
    with open(template_file, 'r') as f:
        return f.read()

def generate_pairwise_prompts(model_a_results, model_b_results, template):
    """Generate evaluation prompts for pairwise comparison."""
    prompts = []
    for a, b in zip(model_a_results, model_b_results):
        # Verify matching session IDs
        assert a['session_id'] == b['session_id'], f"Session ID mismatch: {a['session_id']} != {b['session_id']}"
        
        # Extract required fields
        history = a.get('chat_history', '')
        last_query = a.get('last_query', '')
        output_a = a['output'][0] if isinstance(a['output'], list) else a['output']
        output_b = b['output'][0] if isinstance(b['output'], list) else b['output']
        
        # Randomly decide which output goes to A and B
        if random.random() < 0.5:
            candidate_A = output_a
            candidate_B = output_b
            assignment = {'A': a['generator'], 'B': b['generator']}
        else:
            candidate_A = output_b
            candidate_B = output_a
            assignment = {'A': b['generator'], 'B': a['generator']}
        
        # Format prompt using template
        prompt = template
        prompt = prompt.replace("{$user_query}", last_query)
        prompt = prompt.replace("{$candidate_A}", candidate_A)
        prompt = prompt.replace("{$candidate_B}", candidate_B)
        
        prompts.append({
            'session_id': a['session_id'],
            'prompt': prompt,
            'assignment': assignment
        })
    # print(json.dumps(prompts, indent=2))
    # exit()
    return prompts

@retry_handler(retry_limit=3)
def evaluate_pair(prompt, args):
    """Evaluate a single pair using OpenAI API."""
    response = openai_chat_request(
        model=args.model,
        prompt=prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        json_mode=True
    )
    return response[0]

def main(args):
    # Load results from both models
    model_a_results = load_results(os.path.join(args.result_dir, f"{args.model_a}.json"))
    model_b_results = load_results(os.path.join(args.result_dir, f"{args.model_b}.json"))
    
    # Load evaluation template
    template = load_template(args.eval_template)
    
    # Generate evaluation prompts
    eval_prompts = generate_pairwise_prompts(model_a_results, model_b_results, template)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.eval_output_file), exist_ok=True)
    
    # Evaluate pairs
    results = []
    for item in tqdm(eval_prompts, desc="Evaluating pairs"):
        result = {
            'session_id': item['session_id'],
            'prompt': item['prompt'],
            'assignment': item['assignment'],
            'result': 'N/A'
        }
        
        try:
            eval_response = evaluate_pair(item['prompt'], args)
            result['result'] = eval_response
            result['parsed_result'] = json.loads(eval_response)
            # Add winner and degree based on choice
            choice = parsed_result['choice']
            if choice in ['A++', 'A+']:
                result['winner'] = result['assignment']['A']
                result['degree'] = '++' if choice == 'A++' else '+'
            elif choice in ['B++', 'B+']:
                result['winner'] = result['assignment']['B']
                result['degree'] = '++' if choice == 'B++' else '+'
            else:  # A=B
                result['winner'] = 'tie'
                result['degree'] = '='
        except Exception as e:
            result['error'] = str(e)
        
        results.append(result)
        
        # Save intermediate results
        if len(results) % args.save_interval == 0:
            with open(args.eval_output_file, 'w') as f:
                json.dump(results, f, indent=2)
    
    # Save final results
    with open(args.eval_output_file, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    import argparse
    import random  # Add this import
    
    parser = argparse.ArgumentParser() 
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--model_a", type=str, required=True)
    parser.add_argument("--model_b", type=str, required=True)
    parser.add_argument("--eval_template", type=str, required=True)
    parser.add_argument("--eval_output_file", type=str, required=True)
    parser.add_argument("--model", type=str, default="o1-mini-2024-09-12")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)  # Add seed argument
    
    args = parser.parse_args()
    random.seed(args.seed)  # Set random seed
    main(args)

"""
A=gpt-4o-mini-2024-07-18
B=o1-mini-2024-09-12
J=gpt-4o-mini-2024-07-18

python src/pairwise_eval_clean.py \
    --result_dir result_dirs/wild_bench_v2-hard/ \
    --model_a ${A} \
    --model_b ${B} \
    --eval_template evaluation/eval_template.pairwise.v3.md \
    --eval_output_file eval_results/v2-hard.1103/A=${A}.B=${B}.J=${J}.json \
    --model ${J}

# o1-mini-2024-09-12
"""