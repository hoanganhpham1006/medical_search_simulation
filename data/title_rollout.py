import pandas as pd
from pandarallel import pandarallel
from openai import OpenAI
import os
import re
from dotenv import load_dotenv
from datasets import load_dataset

import json
import re

def extract_json_from_response(llm_response):
    """
    Extract JSON from LLM response that may contain markdown code blocks or other text.
    
    Args:
        llm_response (str): The raw response from the LLM
        
    Returns:
        dict: Parsed JSON data or None if extraction fails
    """
    try:
        # Method 1: Try to parse the entire response as JSON
        return json.loads(llm_response.strip())
    except json.JSONDecodeError:
        pass
    
    # Method 2: Look for JSON in code blocks (```json ... ```)
    json_pattern = r'```json\s*(.*?)\s*```'
    match = re.search(json_pattern, llm_response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    
    # Method 3: Look for JSON in regular code blocks (``` ... ```)
    code_pattern = r'```\s*(.*?)\s*```'
    match = re.search(code_pattern, llm_response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    
    # Method 4: Look for JSON object pattern { ... }
    json_obj_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_obj_pattern, llm_response)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    # Method 5: Last resort - try to find title and summary separately
    try:
        title_match = re.search(r'"title":\s*"([^"]+)"', llm_response)
        summary_match = re.search(r'"summary":\s*"([^"]+)"', llm_response)
        
        if title_match and summary_match:
            return {
                "title": title_match.group(1),
                "summary": summary_match.group(1)
            }
    except:
        pass
    
    return None

# Initialize pandarallel
pandarallel.initialize(progress_bar=True, nb_workers=8)

load_dotenv()

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:30000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
model_name = "Qwen/Qwen3-0.6B"



def get_openai_response(passage_texts):
    content = passage_texts[0]
    USER_PROMPT = """
/no_think Analyze ANY input and return EXACTLY this JSON structure:
```json
{
  "summary": "3-5 sentences summarizing key findings, methodology, and clinical significance.",
  "title": "1-2 sentences title, capturing the essence of the document",
}
```

# Requirements
- Title: Clear, specific, one sentence
- Summary: Exactly 3 sentences covering main findings, methods, and implications
- Use accurate medical terminology
- Always only return valid JSON, even if the content is not well-formed

# Content Handling
- For gibberish: 
```json
{
  "title": "Content is gibberish",
  "summary": "Content is gibberish"
}
```
- Never exceed sentence limits - truncate if needed

# Your input
""".strip() + "\n" + content + "\n# Your output\n" 
    count = 0
    while True:
        count += 1
        if count > 3:
            print("Exceeded maximum retries")
            extracted_response = {"title": "No Title", "summary": "No Summary"}
            break
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": USER_PROMPT},
                ],
                temperature=0.6,
                top_p=0.8,
                max_tokens=1024,
            )
            generated = response.choices[0].message.content
            # print(generated)
            extracted_response = extract_json_from_response(generated)
            # print(extracted_response)
            if extracted_response is not None:
                if "summary" in extracted_response and "title" in extracted_response:
                    break

        except Exception as e:
            print(f"Error: {e}")
    
    return extracted_response.get("title", "No Title")



def main():
    # Read the CSV file
    dataset = load_dataset("hoanganhpham/eai_taxonomy_med_part_3", split="train")
    # dataset = dataset.select(range(65752, 65760)) # DEBUG
    # df = pd.DataFrame(dataset)
    # Shuffle the dataframe
    # df = df.sample(frac=1).reset_index(drop=True)
    print(f"Total rows: {len(dataset)}")
    batch_size = 8192
    # for j in range(272 * 8096, len(dataset), batch_size):
    for j in range(len(dataset) - batch_size, 0, -batch_size):
        print(f"Processing from row {j}")
        _df = pd.DataFrame(dataset.select(range(j, j + batch_size)))
        del _df['paper_title']
        _df["paper_title"] = _df['passage_text'].parallel_apply(lambda x: get_openai_response(x))
        _df = _df[["paper_title"]]
        try:
            filename = f"/mnt/sharefs/tuenv/eai/title_outputs_3/output_{j // batch_size}.parquet"
            _df.to_parquet(filename, index=False)
        except Exception as e:
            print(f"Error while saving parquet: {e}")
            continue

    # DEBUG
    # df['passage_text'].apply(lambda x: get_openai_response(x))



if __name__ == "__main__":
    main()
