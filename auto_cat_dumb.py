#!/usr/bin/env python3
import os
import csv
import sqlite3
import uuid
import re
from pathlib import Path

def load_category_mapping(conn):
    """
    Load category mapping from the mapped_categories table in the database.
    
    Args:
        conn: SQLite database connection
        
    Returns:
        dict: Dictionary mapping category to mapped_category
    """
    cursor = conn.cursor()
    
    # Load the mapping from existing mapped_categories table
    cursor.execute("SELECT category, mapped_category FROM mapped_categories")
    mapping = {row[0]: row[1] for row in cursor.fetchall()}
    
    return mapping

def process_csv_files_to_sqlite(folder_path, db_file):
    """
    Process all CSV files in a folder and its subfolders, extracting key attributes,
    and storing the data in a SQLite database.
    
    Args:
        folder_path (str): Path to the folder containing CSV files
        db_file (str): Path to the SQLite database file
    """
    # Key attribute mappings
    key_mappings = {
        "date": ["date", "transaction date", "post date", "posting date"],
        "amount": ["amount", "credit", "debit", "value", "sum", "price", "cost", "payment", "total"],
        "description": ["description", "transaction description", "details", "memo", "narration", "note", "transaction", "info"],
        "category": ["category", "transaction category", "type", "classification", "group", "tag"]
    }
    
    # Connect to SQLite database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # Load category mapping from existing table
    category_map = load_category_mapping(conn)
    
    # Check if transactions table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='transactions'")
    if cursor.fetchone():
        # Check if mapped_category column exists
        cursor.execute("PRAGMA table_info(transactions)")
        columns = [info[1] for info in cursor.fetchall()]
        if "mapped_category" not in columns:
            cursor.execute("ALTER TABLE transactions ADD COLUMN mapped_category TEXT")
            conn.commit()
    else:
        # Create table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id TEXT PRIMARY KEY,
            date TEXT,
            amount TEXT,
            description TEXT,
            category TEXT,
            mapped_category TEXT,
            filename TEXT,
            blob TEXT
        )
        ''')
        conn.commit()
    
    # Statistics counters
    total_rows_processed = 0
    total_rows_inserted = 0
    
    # Find all CSV files in folder and subfolders
    csv_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.csv', '.CSV')):
                csv_files.append(os.path.join(root, file))
    
    # Process each CSV file
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        print(f"Processing {csv_file}...")
        
        try:
            with open(csv_file, 'r', encoding='utf-8', errors='replace') as f:
                # Try to determine the dialect
                sample = f.read(4096)
                f.seek(0)
                
                try:
                    dialect = csv.Sniffer().sniff(sample)
                    has_header = csv.Sniffer().has_header(sample)
                except:
                    # Fallback to default CSV dialect
                    dialect = csv.excel
                    has_header = True
                
                reader = csv.reader(f, dialect)
                headers = next(reader) if has_header else [f"col{i}" for i in range(1, len(next(reader)) + 1)]
                
                if not has_header:
                    # Reset file pointer to beginning if we had to create column names
                    f.seek(0)
                    reader = csv.reader(f, dialect)
                
                # Normalize headers (convert to lowercase for matching)
                headers_norm = [h.lower().strip() for h in headers]
                
                # Map headers to key attributes
                header_mapping = {}
                for key, possible_names in key_mappings.items():
                    for header_idx, header in enumerate(headers_norm):
                        if any(possible_name == header or 
                               possible_name in header or 
                               re.search(rf'\b{re.escape(possible_name)}\b', header) 
                               for possible_name in possible_names):
                            header_mapping[key] = header_idx
                            break
                
                # Process rows
                for row in reader:
                    if not any(row):  # Skip empty rows
                        continue
                    
                    total_rows_processed += 1
                    
                    # Extract key attributes
                    date = row[header_mapping.get('date', 0)] if 'date' in header_mapping and header_mapping['date'] < len(row) else ""
                    amount = row[header_mapping.get('amount', 0)] if 'amount' in header_mapping and header_mapping['amount'] < len(row) else ""
                    description = row[header_mapping.get('description', 0)] if 'description' in header_mapping and header_mapping['description'] < len(row) else ""
                    category = row[header_mapping.get('category', 0)] if 'category' in header_mapping and header_mapping['category'] < len(row) else ""
                    
                    # Look up mapped category
                    mapped_category = category_map.get(category, "")
                    
                    # Generate UUID
                    row_id = str(uuid.uuid4())
                    
                    # Create blob
                    blob_parts = []
                    for idx, val in enumerate(row):
                        if idx < len(headers):
                            blob_parts.append(f"{headers[idx]}-{val}")
                        else:
                            blob_parts.append(f"column{idx+1}-{val}")
                    blob = "|".join(blob_parts)
                    
                    # Insert row if it doesn't exist
                    cursor.execute(
                        '''
                        INSERT OR IGNORE INTO transactions 
                        (id, date, amount, description, category, mapped_category, filename, blob)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ''',
                        (row_id, date, amount, description, category, mapped_category, filename, blob)
                    )
                    
                    # Check if row was inserted
                    if cursor.rowcount > 0:
                        total_rows_inserted += 1
            
            conn.commit()
            
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
    
    conn.close()
    
    # Print statistics
    print("\nProcessing complete!")
    print(f"Total rows processed: {total_rows_processed}")
    print(f"Total rows inserted: {total_rows_inserted}")
    print(f"Rows skipped (already exist): {total_rows_processed - total_rows_inserted}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process CSV files into SQLite database')
    parser.add_argument('folder', help='Folder containing CSV files')
    parser.add_argument('db_file', help='SQLite database file')
    
    args = parser.parse_args()
    
    folder_path = args.folder
    db_file = args.db_file
    
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory")
        exit(1)
    
    process_csv_files_to_sqlite(folder_path, db_file)