# transaction_labeler.py
import json
import os
import argparse
import datetime
from typing import List, Dict, Any, Optional, Tuple, Set
import subprocess

# Configuration
CONFIG = {
    "models": {
        "gemma3": {
            "context_window": 128000,
            "command": "llm -m mlx-community/gemma-3-12b-it-4bit generate",
            "token_buffer": 1000,  # Reserve tokens for prompt and new batch
        },
        "mistral": {
            "context_window": 8192,
            "command": "llm -m mistral generate",
            "token_buffer": 1000,
        },
        "anthropic": {
            "context_window": 100000,
            "command": "llm -m claude generate",
            "token_buffer": 2000,
        },
        "openai": {
            "context_window": 16384,
            "command": "llm -m gpt-4 generate",
            "token_buffer": 1500,
        }
    },
    "categories": [
        "Food", "Grocery", "Shopping", "Travel", "Automotive", 
        "Home", "Payment", "Health", "Services", "Fees", 
        "Child", "Entertainment", "Other"
    ],
    "base_dir": "tools_data/auto_cat",
    "batch_dir": "batches",
    "corrections_dir": "corrections",
    "results_dir": "results",
    "history_file": "history.json"
}

def ensure_directories():
    """Create necessary directories if they don't exist."""
    for directory in [
        CONFIG["base_dir"],
        os.path.join(CONFIG["base_dir"], CONFIG["batch_dir"]),
        os.path.join(CONFIG["base_dir"], CONFIG["corrections_dir"]),
        os.path.join(CONFIG["base_dir"], CONFIG["results_dir"])
    ]:
        if not os.path.exists(directory):
            os.makedirs(directory)

def load_transactions(file_path: str) -> List[Dict[str, Any]]:
    """Load transactions from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def estimate_tokens(text: str) -> int:
    """Estimate token count for a string."""
    # Simple estimation: 4 chars per token on average
    return len(text) // 4

def estimate_transaction_tokens(transaction: Dict[str, Any]) -> int:
    """Estimate token count for a transaction."""
    # Convert transaction to string and estimate tokens
    return estimate_tokens(json.dumps(transaction))

def get_processed_transaction_ids() -> Set[str]:
    """Get IDs of all transactions that have been processed before."""
    processed_ids = set()
    
    # Check results directory
    results_dir = os.path.join(CONFIG["base_dir"], CONFIG["results_dir"])
    if os.path.exists(results_dir):
        for file_name in os.listdir(results_dir):
            if file_name.endswith('.json'):
                try:
                    with open(os.path.join(results_dir, file_name), 'r') as f:
                        transactions = json.load(f)
                        for transaction in transactions:
                            if "id" in transaction and "model_guess" in transaction:
                                processed_ids.add(transaction["id"])
                except Exception as e:
                    print(f"Warning: Could not process file {file_name}: {e}")
    
    # Check corrections directory
    corrections_dir = os.path.join(CONFIG["base_dir"], CONFIG["corrections_dir"])
    if os.path.exists(corrections_dir):
        for file_name in os.listdir(corrections_dir):
            if file_name.endswith('.json'):
                try:
                    with open(os.path.join(corrections_dir, file_name), 'r') as f:
                        transactions = json.load(f)
                        for transaction in transactions:
                            if "id" in transaction:
                                processed_ids.add(transaction["id"])
                except Exception as e:
                    print(f"Warning: Could not process file {file_name}: {e}")
    
    return processed_ids

def load_examples_from_results(limit_tokens: int) -> List[Dict[str, Any]]:
    """Load examples from previous results if corrections folder is empty."""
    results_dir = os.path.join(CONFIG["base_dir"], CONFIG["results_dir"])
    if not os.path.exists(results_dir):
        return []
    
    result_files = sorted(
        [f for f in os.listdir(results_dir) if f.endswith('.json')],
        reverse=True  # Most recent first
    )
    
    example_transactions = []
    total_tokens = 0
    
    for file_name in result_files:
        file_path = os.path.join(results_dir, file_name)
        with open(file_path, 'r') as f:
            transactions = json.load(f)
        
        for transaction in transactions:
            if "model_guess" in transaction:
                # Create a copy with "corrected_category" field for consistency
                example_transaction = transaction.copy()
                example_transaction["corrected_category"] = transaction["model_guess"]
                
                token_count = estimate_transaction_tokens(example_transaction)
                if total_tokens + token_count <= limit_tokens:
                    example_transactions.append(example_transaction)
                    total_tokens += token_count
                else:
                    # We've reached the token limit
                    return example_transactions
    
    return example_transactions

def load_corrected_transactions(limit_tokens: int) -> List[Dict[str, Any]]:
    """Load corrected transactions, most recent first, up to token limit."""
    corrections_dir = os.path.join(CONFIG["base_dir"], CONFIG["corrections_dir"])
    if not os.path.exists(corrections_dir) or not os.listdir(corrections_dir):
        print("No corrections found. Using previous results as examples.")
        return load_examples_from_results(limit_tokens)
    
    correction_files = sorted(
        [f for f in os.listdir(corrections_dir) if f.endswith('.json')],
        reverse=True  # Most recent first
    )
    
    corrected_transactions = []
    total_tokens = 0
    
    for file_name in correction_files:
        file_path = os.path.join(corrections_dir, file_name)
        with open(file_path, 'r') as f:
            transactions = json.load(f)
        
        for transaction in transactions:
            if "corrected_category" in transaction:
                token_count = estimate_transaction_tokens(transaction)
                if total_tokens + token_count <= limit_tokens:
                    corrected_transactions.append(transaction)
                    total_tokens += token_count
                else:
                    # We've reached the token limit
                    return corrected_transactions
    
    return corrected_transactions

def format_transaction_for_prompt(transaction: Dict[str, Any]) -> str:
    """Format a transaction for inclusion in the prompt, including all fields."""
    result = f"ID: {transaction['id']}\n"
    
    # Add all fields except 'id', 'model_guess', and 'corrected_category'
    skip_fields = {'id', 'model_guess', 'corrected_category'}
    
    for key, value in transaction.items():
        if key not in skip_fields and value:  # Skip empty values
            # Capitalize field name and format value
            formatted_key = key.replace('_', ' ').title()
            result += f"{formatted_key}: {value}\n"
    
    return result

def format_prompt(example_transactions: List[Dict[str, Any]], new_transactions: List[Dict[str, Any]]) -> str:
    """Create the prompt for the LLM."""
    categories_str = ", ".join(CONFIG["categories"])
    
    prompt = f"""You are a transaction categorization expert. 
Your task is to categorize financial transactions into one of the following categories:
{categories_str}

For each transaction, examine the description and other details to determine the most appropriate category.
"""
    
    # Add examples if available
    if example_transactions:
        prompt += "\nHere are some examples of correctly categorized transactions:\n"
        
        for example in example_transactions:
            prompt += f"""
Transaction Example:
{format_transaction_for_prompt(example)}
Correct Category: {example["corrected_category"]}

"""
    
    prompt += """
Now, categorize the following new transactions. For each transaction, respond with ONLY the transaction ID and the category. 
Format your response as JSON with transaction ID as key and category as value:

{
  "transaction_id_1": "Category",
  "transaction_id_2": "Category",
  ...
}

Here are the transactions to categorize:

"""
    
    for transaction in new_transactions:
        prompt += f"""
{format_transaction_for_prompt(transaction)}
"""
    
    return prompt

def call_llm(prompt: str, model: str) -> str:
    """Call the LLM using the llm CLI tool and pass prompt via stdin."""
    model_config = CONFIG["models"][model]
    command = model_config["command"]
    
    # Call LLM using the command and pass prompt via stdin
    process = subprocess.Popen(
        command,
        shell=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    stdout, stderr = process.communicate(input=prompt)
    
    if process.returncode != 0:
        raise Exception(f"LLM call failed: {stderr}")
    
    return stdout

def parse_llm_response(response: str) -> Dict[str, str]:
    """Parse the LLM response to extract transaction ID to category mappings."""
    try:
        # Extract JSON from response (it might contain other text)
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start == -1 or json_end == 0:
            raise ValueError("No JSON found in response")
        
        json_str = response[json_start:json_end]
        return json.loads(json_str)
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        print(f"Response was: {response}")
        # Return empty dict on failure
        return {}

def apply_predictions(transactions: List[Dict[str, Any]], predictions: Dict[str, str]) -> List[Dict[str, Any]]:
    """Apply model predictions to the transactions."""
    result = []
    for transaction in transactions:
        transaction_copy = transaction.copy()
        transaction_id = transaction_copy["id"]
        if transaction_id in predictions:
            transaction_copy["model_guess"] = predictions[transaction_id]
        else:
            transaction_copy["model_guess"] = "Other"  # Default if no prediction
        result.append(transaction_copy)
    return result

def save_results(transactions: List[Dict[str, Any]], batch_name: str) -> str:
    """Save labeled transactions to the results directory."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"{batch_name}_{timestamp}.json"
    result_path = os.path.join(CONFIG["base_dir"], CONFIG["results_dir"], result_file)
    
    with open(result_path, 'w') as f:
        json.dump(transactions, f, indent=2)
    
    return result_path

def load_history() -> Dict[str, Any]:
    """Load processing history from the history file."""
    history_path = os.path.join(CONFIG["base_dir"], CONFIG["history_file"])
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load history file: {e}")
            return initialize_history()
    else:
        return initialize_history()

def initialize_history() -> Dict[str, Any]:
    """Create a new history object with default values."""
    return {
        "runs": [],
        "transaction_count": 0,
        "last_updated": "",
        "models_used": {},
        "category_distribution": {category: 0 for category in CONFIG["categories"]},
        "file_registry": {
            "batches": [],
            "results": [],
            "corrections": []
        }
    }

def update_history(
    history: Dict[str, Any],
    batch_file: str,
    result_file: str,
    model: str,
    transactions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Update history with information from current run."""
    # Get current timestamp
    timestamp = datetime.datetime.now().isoformat()
    
    # Extract filename from paths
    batch_filename = os.path.basename(batch_file)
    result_filename = os.path.basename(result_file)
    
    # Create run record
    run_record = {
        "timestamp": timestamp,
        "batch_file": batch_filename,
        "result_file": result_filename,
        "model": model,
        "transaction_count": len(transactions),
        "category_distribution": {}
    }
    
    # Update category distribution
    for transaction in transactions:
        if "model_guess" in transaction:
            category = transaction["model_guess"]
            if category in run_record["category_distribution"]:
                run_record["category_distribution"][category] += 1
            else:
                run_record["category_distribution"][category] = 1
                
            if category in history["category_distribution"]:
                history["category_distribution"][category] += 1
            else:
                history["category_distribution"][category] = 1
    
    # Update runs list
    history["runs"].append(run_record)
    
    # Update global counters
    history["transaction_count"] += len(transactions)
    history["last_updated"] = timestamp
    
    # Update models used
    if model in history["models_used"]:
        history["models_used"][model] += 1
    else:
        history["models_used"][model] = 1
    
    # Update file registry
    if batch_filename not in history["file_registry"]["batches"]:
        history["file_registry"]["batches"].append(batch_filename)
    
    if result_filename not in history["file_registry"]["results"]:
        history["file_registry"]["results"].append(result_filename)
    
    return history

def save_history(history: Dict[str, Any]) -> None:
    """Save history to the history file."""
    history_path = os.path.join(CONFIG["base_dir"], CONFIG["history_file"])
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Transaction Labeler")
    parser.add_argument("batch_file", help="JSON file with transactions to label")
    parser.add_argument("--model", default="gemma3", choices=CONFIG["models"].keys(),
                        help="LLM model to use for labeling")
    parser.add_argument("--skip-processed", action="store_true",
                        help="Skip transactions that have been processed before")
    parser.add_argument("--min-examples", type=int, default=5,
                        help="Minimum number of examples to include in prompt")
    args = parser.parse_args()
    
    # Ensure directories exist
    ensure_directories()
    
    # Load history
    history = load_history()
    
    # Extract batch name from file path
    batch_name = os.path.splitext(os.path.basename(args.batch_file))[0]
    
    # Load new transactions
    new_transactions = load_transactions(args.batch_file)
    print(f"Loaded {len(new_transactions)} transactions from {args.batch_file}")
    
    # Filter out already processed transactions if requested
    if args.skip_processed:
        processed_ids = get_processed_transaction_ids()
        original_count = len(new_transactions)
        new_transactions = [t for t in new_transactions if t["id"] not in processed_ids]
        skipped_count = original_count - len(new_transactions)
        print(f"Skipped {skipped_count} already processed transactions")
    
    # If no transactions left to process, exit
    if not new_transactions:
        print("No transactions to process. Exiting.")
        return
    
    # Calculate available token budget for examples
    model_config = CONFIG["models"][args.model]
    context_window = model_config["context_window"]
    token_buffer = model_config["token_buffer"]
    
    # Estimate tokens for new transactions
    new_batch_tokens = sum(estimate_transaction_tokens(t) for t in new_transactions)
    
    # Calculate available tokens for examples
    available_tokens = context_window - token_buffer - new_batch_tokens
    if available_tokens <= 0:
        print("Warning: Not enough context window for examples")
        available_tokens = 0
    
    # Load examples (either from corrections or previous results if no corrections exist)
    examples = load_corrected_transactions(available_tokens)
    
    # If we don't have minimum examples, try to add some from results
    if len(examples) < args.min_examples and available_tokens > 0:
        print(f"Only found {len(examples)} examples, trying to add more from results")
        examples_from_results = load_examples_from_results(available_tokens - sum(estimate_transaction_tokens(e) for e in examples))
        # Remove duplicates by ID
        example_ids = {e["id"] for e in examples}
        additional_examples = [e for e in examples_from_results if e["id"] not in example_ids]
        
        # Add as many as will fit
        for example in additional_examples:
            if len(examples) >= args.min_examples:
                break
            examples.append(example)
    
    print(f"Using {len(examples)} examples for context")
    
    # Create prompt
    prompt = format_prompt(examples, new_transactions)
    
    # Call LLM
    print(f"Calling {args.model} for predictions...")
    llm_response = call_llm(prompt, args.model)
    
    # Parse response
    predictions = parse_llm_response(llm_response)
    print(f"Received predictions for {len(predictions)} transactions")
    
    # Apply predictions
    labeled_transactions = apply_predictions(new_transactions, predictions)
    
    # Save results
    result_file = save_results(labeled_transactions, batch_name)
    print(f"Results saved to {result_file}")
    
    # Update and save history
    history = update_history(history, args.batch_file, result_file, args.model, labeled_transactions)
    save_history(history)
    print(f"History updated ({history['transaction_count']} total transactions processed)")
    
    print("\nNext steps:")
    print("1. Review the results file and correct any wrong categories")
    print("2. Save the corrected file in the corrections directory")
    print(f"   - {os.path.join(CONFIG['base_dir'], CONFIG['corrections_dir'])}")
    print("3. The corrections will be used as examples for future runs")

if __name__ == "__main__":
    main()