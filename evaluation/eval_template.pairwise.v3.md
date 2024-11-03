# Instruction 

As an expert evaluator, your role is to assess the quality of responses produced by two AI models. You will be given a user query along with two AI-generated responses (Response A and Response B). Begin by thoroughly reviewing the user query and the conversation history to understand the task, then evaluate the quality of the responses according to the guidelines provided below.

# Conversation between User and AI

## User Query
<|begin_of_query|>

{$user_query}

<|end_of_query|>

## Response A
<|begin_of_response_A|>

{$candidate_A}

<|end_of_response_A|>

## Response B
<|begin_of_response_B|>

{$candidate_B}

<|end_of_response_B|>

# Evaluation    

## Rules 

You should compare the above two responses based on the context and rules provided below.
Please focus on the following key aspects when evaluating:

1. Correctness
   - Accuracy of information provided
   - Freedom from factual errors
   - Logical consistency

2. Helpfulness
   - Direct addressing of the user's needs
   - Practical usefulness of the solution
   - Completeness of the answer

Note: Do not prioritize stylistic differences (e.g., tone, verbosity) or response length unless they significantly impact understanding.

There are five choices to give your final assessment: ["A++", "A+", "A=B", "B+", "B++"], which correspond to the following meanings:
    - `A++`: Response A is much better than Response B.
    - `A+`: Response A is only slightly better than Response B.
    - `A=B`: Response A and B are of the same quality. Please use this choice sparingly.
    - `B+`: Response B is only slightly better than Response A.
    - `B++`: Response B is much better than Response A.


## Output Format 

Please provide your evaluation results in the following json format by filling in the placeholders in []:
```json
{
    "analysis_A": "[Your analysis for response A]",
    "analysis_B": "[Your analysis for response B]",
    "reason_preference": "[Your reasons for preferring one response over the other]",
    "choice": "[A++ or A+ or A=B or B+ or B++]"
}
```