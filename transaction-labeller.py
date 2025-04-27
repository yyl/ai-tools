import sqlite3
import argparse
import subprocess
import random
import json
import os
import datetime
import tqdm
import time
from typing import List, Dict, Tuple, Optional

def create_prompt_template(examples: List[Dict], transaction: Dict, max_tokens: int) -> Tuple[str, int]:
    """
    Create a prompt for the LLM with examples and the current transaction.
    Also calculates token count approximation.
    
    Args:
        examples: List of example transactions with labels
        transaction: Current transaction to be labeled
        max_tokens: Maximum tokens allowed
        
    Returns:
        prompt text and estimated token count
    """
    # Approximate token count (rough estimate: 4 chars = 1 token)
    def estimate_tokens(text: str) -> int:
        return len(text) // 4
    
    # Basic prompt structure
    prompt = """You are a financial transaction categorizer. Based on the transaction details provided, assign the most appropriate category label.
Please reply with ONLY the category label and nothing else. Assign only one category from this list: ["Food", "Grocery", "Shopping", "Travel", "Automotive", 
"Home", "Payment", "Health", "Services", "Fees", "Entertainment", "Other"]

Here are some examples of properly categorized transactions:

"""
    
    # Add examples one by one, checking token count
    token_count = estimate_tokens(prompt)
    examples_added = 0
    
    for example in examples:
        example_text = f"Date: {example['date']}\nAmount: {example['amount']}\nDescription: {example['description']}\nLabel: {example['label']}\n\n"
        example_tokens = estimate_tokens(example_text)
        
        # If adding this example would exceed limit, break
        if token_count + example_tokens > max_tokens * 0.7:  # Use only 70% of tokens for examples
            break
            
        prompt += example_text
        token_count += example_tokens
        examples_added += 1
    
    # Add the transaction to be labeled
    transaction_text = f"""
Now, please categorize this transaction:

Date: {transaction['date']}
Amount: {transaction['amount']}
Description: {transaction['description']}

The category label should be:"""

    token_count += estimate_tokens(transaction_text)
    prompt += transaction_text
    
    return prompt, token_count

def get_model_token_limit(model: str) -> int:
    """Get the context window size for the specified model"""
    # Define token limits for common models
    model_limits = {
        # "gpt-3.5-turbo": 4096,
        # "gpt-4": 8192,
        # "gpt-4-turbo": 128000,
        # "claude-3-opus": 200000,
        # "claude-3-sonnet": 100000,
        # "claude-3-haiku": 200000,
        # "mistral-small": 8192,
        # "mistral-medium": 32768,
        # "mistral-large": 32768,
        # "llama3": 8192,
        # "llama3-70b": 8192,
        "mlx-community/gemma-3-12b-it-4bit": 128000,
        "default": 4096  # Default for unknown models
    }
    
    for model_name, limit in model_limits.items():
        if model_name in model.lower():
            return limit
    
    return model_limits["default"]

def invoke_llm(prompt: str, model: str) -> str:
    """
    Invoke the LLM model using the LLM CLI tool
    
    Args:
        prompt: The prepared prompt text
        model: Model identifier
        
    Returns:
        The LLM's response (label)
    """
    try:
        # Call LLM CLI tool with prompt
        result = subprocess.run(
            ["llm", "prompt", "-m", model, prompt],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Return the trimmed output
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error calling LLM: {e}")
        print(f"Error output: {e.stderr}")
        return ""
    except Exception as e:
        print(f"Unexpected error invoking LLM: {e}")
        return ""

def update_transactions_with_labels(db_path: str, model: str, num_examples: int = 10, batch_size: int = 2) -> None:
    """
    Update transactions table with labels based on category mappings and LLM predictions.
    
    Args:
        db_path: Path to SQLite database file
        model: LLM model to use
        num_examples: Number of examples to use for few-shot learning
        batch_size: Number of transactions to process in one batch
    """
    # Get DB directory for log file location
    db_dir = os.path.dirname(os.path.abspath(db_path))
    log_file = os.path.join(db_dir, "labeling_history.jsonl")
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # This enables column access by name
    cursor = conn.cursor()
    
    # Check and create necessary columns
    cursor.execute("PRAGMA table_info(transactions)")
    columns = [info[1] for info in cursor.fetchall()]
    
    # Add missing columns if needed
    if 'label' not in columns:
        cursor.execute("ALTER TABLE transactions ADD COLUMN label TEXT")
        print("Added 'label' column to transactions table")
    
    if 'by_llm' not in columns:
        cursor.execute("ALTER TABLE transactions ADD COLUMN by_llm BOOLEAN DEFAULT FALSE")
        print("Added 'by_llm' column to transactions table")
    
    # Process using category mappings first
    print("Step 1: Applying category mappings...")
    
    # Get all category mappings
    cursor.execute("SELECT category, mapped_category FROM category_mappings")
    category_mappings = {row[0]: row[1] for row in cursor.fetchall()}
    print(f"Loaded {len(category_mappings)} category mappings")
    
    # Update transactions with mappings
    cursor.execute("""
        UPDATE transactions 
        SET label = CASE 
                WHEN category = '' OR category IS NULL THEN ''
                ELSE (SELECT mapped_category FROM category_mappings WHERE category_mappings.category = transactions.category)
            END
        WHERE (label IS NULL OR label = '') 
        AND category IS NOT NULL 
        AND category != '' 
        AND EXISTS (SELECT 1 FROM category_mappings WHERE category_mappings.category = transactions.category)
    """)
    
    mapping_updates = cursor.rowcount
    conn.commit()
    print(f"Updated {mapping_updates} transaction labels using category mappings")
    
    # Log mapping results
    if mapping_updates > 0:
        timestamp = datetime.datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "action": "category_mapping",
            "count": mapping_updates,
            "mappings": dict(category_mappings)
        }
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    # If model is provided, use LLM to label remaining transactions
    if model:
        print(f"\nStep 2: Using LLM ({model}) to label remaining transactions...")
        
        # Get token limit for the model
        token_limit = get_model_token_limit(model)
        print(f"Model token limit: {token_limit}")
        
        # Get labeled examples for few-shot learning
        cursor.execute("""
            SELECT id, date, amount, description, label 
            FROM transactions 
            WHERE label IS NOT NULL 
            AND label != '' 
            AND by_llm = FALSE
            ORDER BY RANDOM() 
            LIMIT ?
        """, (num_examples,))
        
        examples = [dict(row) for row in cursor.fetchall()]
        print(f"Loaded {len(examples)} examples for few-shot learning")
        
        # Get transactions that need labeling
        cursor.execute("""
            SELECT id, date, amount, description 
            FROM transactions 
            WHERE (label IS NULL OR label = '') 
            ORDER BY date
        """)
        
        unlabeled_transactions = [dict(row) for row in cursor.fetchall()]
        print(f"Found {len(unlabeled_transactions)} transactions to label")
        
        if not unlabeled_transactions:
            print("No transactions need labeling. Exiting.")
            conn.close()
            return
        
        llm_updates = 0
        log_entries = []
        
        # Process transactions in batches with progress bar
        for i in tqdm.tqdm(range(0, len(unlabeled_transactions), batch_size), desc="Processing batches"):
            batch = unlabeled_transactions[i:i+batch_size]
            
            for transaction in batch:
                # Create prompt
                prompt, token_count = create_prompt_template(examples, transaction, token_limit)
                
                if token_count > token_limit:
                    print(f"Warning: Prompt exceeds token limit ({token_count} > {token_limit})")
                
                # Get prediction from LLM
                label = invoke_llm(prompt, model)
                
                if label:
                    # Update transaction
                    cursor.execute("""
                        UPDATE transactions 
                        SET label = ?, by_llm = TRUE 
                        WHERE id = ?
                    """, (label, transaction['id']))
                    
                    llm_updates += 1
                    
                    # Prepare log entry
                    log_entries.append({
                        "timestamp": datetime.datetime.now().isoformat(),
                        "action": "llm_labeling",
                        "model": model,
                        "transaction_id": transaction['id'],
                        "date": transaction['date'],
                        "amount": transaction['amount'],
                        "description": transaction['description'],
                        "assigned_label": label
                    })
            
            # Commit batch and write logs
            conn.commit()
            
            # Write log entries
            with open(log_file, 'a') as f:
                for entry in log_entries:
                    f.write(json.dumps(entry) + '\n')
            log_entries = []  # Clear log entries
            
            # Small delay to prevent overwhelming the LLM service
            time.sleep(0.2)
        
        print(f"Labeled {llm_updates} transactions using LLM")
        
        # Log summary
        summary_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "action": "llm_labeling_summary",
            "model": model,
            "total_labeled": llm_updates,
            "examples_used": len(examples)
        }
        with open(log_file, 'a') as f:
            f.write(json.dumps(summary_entry) + '\n')
    
    # Verify and show results
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN label IS NOT NULL AND label != '' THEN 1 ELSE 0 END) as labeled,
            SUM(CASE WHEN by_llm = TRUE THEN 1 ELSE 0 END) as by_llm
        FROM transactions
    """)
    stats = dict(cursor.fetchone())
    
    print("\nSummary:")
    print(f"Total transactions: {stats['total']}")
    print(f"Labeled transactions: {stats['labeled']} ({stats['labeled']/stats['total']*100:.1f}%)")
    print(f"Labeled by LLM: {stats['by_llm']}")
    
    # Sample of recently labeled transactions
    print("\nSample of recently labeled transactions:")
    cursor.execute("""
        SELECT id, date, amount, description, label, by_llm
        FROM transactions
        WHERE label IS NOT NULL AND label != ''
        ORDER BY RANDOM()
        LIMIT 5
    """)
    
    for row in cursor.fetchall():
        print(f"ID: {row['id']}, Date: {row['date']}, Amount: {row['amount']}")
        print(f"Description: {row['description']}")
        print(f"Label: {row['label']} ({'LLM' if row['by_llm'] else 'Mapping'})")
        print("-" * 50)
    
    conn.close()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Update transaction labels using category mappings and optionally LLM')
    parser.add_argument('db_path', help='Path to the SQLite database file')
    parser.add_argument('--model', '-m', 
        default="mlx-community/gemma-3-12b-it-4bit", 
        help='LLM model to use for labeling (default: "mlx-community/gemma-3-12b-it-4bit")')
    parser.add_argument('--examples', '-e', type=int, default=10, help='Number of examples to use for few-shot learning (default: 10)')
    parser.add_argument('--batch-size', '-b', type=int, default=2, help='Number of transactions to process in one batch (default: 2)')
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        update_transactions_with_labels(
            args.db_path,
            model=args.model,
            num_examples=args.examples,
            batch_size=args.batch_size
        )
        print("Process completed successfully!")
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()