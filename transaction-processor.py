import os
import csv
import json
import sys
import random
import pprint
import re
import uuid
import sqlite3
import argparse

def clean_text(text):
    """
    Clean text by replacing multiple whitespaces, tabs, and newlines with a single space.
    
    Args:
        text (str): Input text to clean
    
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return text
    return re.sub(r'\s+', ' ', str(text)).strip()

def extract_provider(filename):
    """
    Extract provider name by replacing '-' and '_' with spaces and taking the first word.
    
    Args:
        filename (str): Input filename
    
    Returns:
        str: Extracted provider name
    """
    # Replace '-' and '_' with spaces, then split and take first word
    cleaned_name = filename.replace('-', ' ').replace('_', ' ')
    return cleaned_name.split()[0]

def determine_provider_type(filename):
    """
    Determine provider type based on filename.
    
    Args:
        filename (str): Input filename
    
    Returns:
        str: Provider type ('banking' or 'credit_card')
    """
    banking_prefixes = ['Chase1515', 'Chase6992', 'Discover-checking', 'Discover-saving', 'WF']
    
    for prefix in banking_prefixes:
        if filename.startswith(prefix):
            return 'banking'
    
    return 'credit_card'

def process_csv_file(file_path, file_object_counts):
    """
    Process a single CSV file and return the extracted transactions.
    
    Args:
        file_path (str): Path to the CSV file
        file_object_counts (dict): Dictionary to track objects per file
    
    Returns:
        list: List of extracted transactions
    """
    filename = os.path.basename(file_path)
    transactions = []
    
    # Default column names for WF files
    default_headers = ['Date', 'Amount', 'Col1', 'Col2', 'Description']
    
    # Extract provider name and type
    provider = extract_provider(filename)
    provider_type = determine_provider_type(filename)
    
    # Process the CSV file
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as csvfile:
        # Determine if the CSV has headers
        first_line = csvfile.readline().strip()
        csvfile.seek(0)  # Reset file pointer
        
        # Prepare the reader
        if filename.startswith('WF'):
            # For WF files, use default headers
            csvreader = csv.reader(csvfile)
            headers = default_headers
        elif ',' not in first_line:
            # Keep original headers for non-WF files without comma
            csvreader = csv.reader(csvfile)
            headers = next(csvreader)
            csvfile.seek(0)  # Reset file pointer
        else:
            # Normal CSV with headers
            csvreader = csv.reader(csvfile)
            headers = next(csvreader)
        
        file_transactions = []
        for row in csvreader:
            # Skip empty rows
            if not row or all(cell == '' for cell in row):
                continue
                
            # Ensure row has enough elements, pad if needed
            row = row + [''] * (len(headers) - len(row))
            
            # Create transaction dictionary
            transaction = {}
            
            # Generate unique UUID
            transaction['id'] = str(uuid.uuid4())
            
            # Create blob string parts
            blob_parts = []
            
            for i, header in enumerate(headers):
                if i >= len(row):
                    continue
                    
                # Skip columns related to reference
                if header.lower() in ['reference', 'reference number']:
                    continue
                
                # Clean the value
                cleaned_value = clean_text(row[i])
                
                # Add to blob parts if value exists
                if cleaned_value:
                    blob_parts.append(f"{header}-{cleaned_value}")
                
                # Special handling for Debit/Credit columns
                if header.lower() in ['debit', 'credit']:
                    # Only add if not zero
                    if cleaned_value != '0' and cleaned_value:
                        transaction[header] = cleaned_value
                elif cleaned_value:
                    # Add other columns with non-empty values
                    transaction[header] = cleaned_value
            
            # Add provider information and type
            transaction['provider'] = provider
            transaction['provider_type'] = provider_type
            transaction['filename'] = filename
            
            # Add blob field - all fields concatenated
            transaction['blob'] = '|'.join(blob_parts)
            
            # Only add non-empty transactions
            if transaction and len(transaction) > 1:  # More than just the id
                file_transactions.append(transaction)
        
        # Update tracking
        file_object_counts[filename] = len(file_transactions)
        transactions.extend(file_transactions)
    
    return transactions

def traverse_and_process_directory(input_folder):
    """
    Recursively traverse directory and process all CSV files.
    
    Args:
        input_folder (str): Path to the input folder
        
    Returns:
        tuple: Total number of objects processed and list of transactions
    """
    all_transactions = []
    file_object_counts = {}
    
    # Walk through all directories and subdirectories
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.csv') or file.endswith('.CSV'):
                file_path = os.path.join(root, file)
                try:
                    transactions = process_csv_file(file_path, file_object_counts)
                    all_transactions.extend(transactions)
                    print(f"Processed {file}: {file_object_counts[file]} transactions")
                except Exception as e:
                    print(f"Error processing {file}: {e}")
    
    total_objects = sum(file_object_counts.values())
    return total_objects, all_transactions, file_object_counts

def map_keys(transactions):
    """
    Map transaction keys to standardized keys.
    
    Args:
        transactions (list): List of transaction dictionaries
        
    Returns:
        list: List of transactions with mapped keys
    """
    # Define key mappings
    key_mappings = {
        "date": ["date", "transaction date", "post date", "posting date"],
        "amount": ["amount", "credit", "debit", "value", "sum", "price", "cost", "payment", "total"],
        "description": ["description", "transaction description", "details", "memo", "narration", "note", "transaction", "info"],
        "category": ["category", "transaction category", "type", "transaction type", "classification", "group", "tag"]
    }
    
    # Generate 3 random indices for printing (changed from 5 to 3)
    random_indices = random.sample(range(len(transactions)), min(3, len(transactions)))
    
    # Transform transactions
    transformed_transactions = []
    for idx, transaction in enumerate(transactions):
        # Create a new transaction with only the mapped keys
        new_trans = {}
        
        # Keep 'id' if it exists
        if 'id' in transaction:
            new_trans['id'] = transaction['id']
        
        # Add filename field
        if 'filename' in transaction:
            new_trans['filename'] = transaction['filename']

        # Add provider field
        if 'provider' in transaction:
            new_trans['provider'] = transaction['provider']
            
        # Add provider_type field
        if 'provider_type' in transaction:
            new_trans['provider_type'] = transaction['provider_type']
            
        # Add blob field
        if 'blob' in transaction:
            new_trans['blob'] = transaction['blob']
        
        # Map keys based on the defined mappings
        for fixed_key, possible_keys in key_mappings.items():
            found = False
            matched_key = None
            
            # Look for matching keys in the transaction
            for possible_key in possible_keys:
                # Case-insensitive matching
                matching_keys = [k for k in transaction.keys() if k.lower() == possible_key.lower()]
                
                if matching_keys:
                    matched_key = matching_keys[0]
                    value = transaction[matched_key]
                    
                    # Remove dollar sign from amount if present
                    if fixed_key == "amount" and isinstance(value, str):
                        value = value.replace("$", "").strip()
                    
                    new_trans[fixed_key] = value
                    found = True
                    break
            
            # If no match found, add empty string
            if not found:
                new_trans[fixed_key] = ""
        
        transformed_transactions.append(new_trans)
    
    return transformed_transactions

def save_to_sqlite(transactions, db_file):
    """
    Save transactions to SQLite database.
    
    Args:
        transactions (list): List of transaction dictionaries
        db_file (str): Path to the SQLite database file
    """
    # Connect to SQLite database (create if it doesn't exist)
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # Create table if it doesn't exist
    if transactions:
        # Get all column names from the first transaction
        columns = list(transactions[0].keys())
        
        # Create column definitions, ensuring id is the primary key
        column_defs = []
        for col in columns:
            if col == 'id':
                column_defs.append(f"{col} TEXT PRIMARY KEY")
            else:
                column_defs.append(f"{col} TEXT")
        
        # Create the table
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS transactions (
            {", ".join(column_defs)}
        )
        """
        cursor.execute(create_table_sql)
        
        # Insert transactions
        print(f"\nSaving {len(transactions)} transactions to SQLite database...")
        
        # Prepare placeholders for the INSERT statement
        placeholders = ", ".join(["?" for _ in columns])
        
        # Prepare INSERT statement
        insert_sql = f"""
        INSERT OR REPLACE INTO transactions ({", ".join(columns)})
        VALUES ({placeholders})
        """
        
        # Insert transactions in batches to improve performance
        batch_size = 1000
        for i in range(0, len(transactions), batch_size):
            batch = transactions[i:i+batch_size]
            batch_values = [tuple(transaction.get(col, "") for col in columns) for transaction in batch]
            cursor.executemany(insert_sql, batch_values)
            conn.commit()
            print(f"  Saved batch {i//batch_size + 1}/{(len(transactions)-1)//batch_size + 1} ({min(i+batch_size, len(transactions))}/{len(transactions)} transactions)")
    
    # Commit and close connection
    conn.commit()
    conn.close()
    print(f"Database saved to: {db_file}")

def main(input_folder, db_file):
    try:
        # Create output directory for intermediate file (optional, can be removed if not needed)
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Step 1: Process all CSV files in the input folder recursively
        print(f"Processing CSV files in {input_folder}...")
        total_objects, all_transactions, file_object_counts = traverse_and_process_directory(input_folder)

        # Print processing statistics
        print(f"\nTotal objects processed: {total_objects}")
        print("File-wise object counts:")
        for file, count in file_object_counts.items():
            print(f"  {file}: {count} objects")
        
        # Step 2: Map keys to standardized format
        print("\nMapping transaction keys...")
        transformed_transactions = map_keys(all_transactions)
         
        # Step 3: Save transactions to SQLite database using the provided db_file path
        save_to_sqlite(transformed_transactions, db_file) # Use the db_file argument
        
        print(f"\nProcessing completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process transaction CSV files and save to SQLite.")
    parser.add_argument("input_folder", help="Path to the folder containing CSV files.")
    parser.add_argument("db_file", help="Path to the output SQLite database file.")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call main function with parsed arguments
    main(args.input_folder, args.db_file)