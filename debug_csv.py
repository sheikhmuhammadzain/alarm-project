import pandas as pd
from pathlib import Path

# Test reading the CSV file to understand its structure
csv_path = Path(r"c:\Users\ibrahim laptops\Desktop\qbit\alarm project\VCM Events (Feb)\1.csv")

print("=== Testing CSV Structure ===")

# Try different encodings
encodings = ['windows-1252', 'utf-8', 'latin-1', 'cp1252']

for encoding in encodings:
    try:
        print(f"\n--- Testing encoding: {encoding} ---")
        
        # Read first 15 lines as text
        with open(csv_path, 'r', encoding=encoding, errors='replace') as f:
            lines = [f.readline().strip() for _ in range(15)]
        
        for i, line in enumerate(lines):
            if line:
                print(f"Line {i}: {line}")
        
        # Try to find header row
        for i, line in enumerate(lines):
            if 'event' in line.lower() and 'time' in line.lower():
                print(f"\nFound potential header at line {i}: {line}")
                
                # Try reading with this as header
                df = pd.read_csv(csv_path, encoding=encoding, header=i, nrows=5)
                print(f"Columns: {list(df.columns)}")
                print(f"First few rows:\n{df.head()}")
                break
        
        break  # If successful, stop trying other encodings
        
    except Exception as e:
        print(f"Failed with {encoding}: {e}")
        continue
