from analyst_core import load_data, generate_data_report
import os

# Define file path
file_path = "Instagram_Final_Data.xlsx"

if not os.path.exists(file_path):
    print(f"Error: File '{file_path}' not found.")
else:
    try:
        print(f"Loading '{file_path}'...")
        df = load_data(file_path)
        print("Data loaded successfully.")
        
        report = generate_data_report(df)
        print("\n" + "="*30)
        print("INITIAL DATA SCAN REPORT")
        print("="*30)
        print(report)
        print("="*30)
        
        # Also print head for visual inspection
        print("\nFirst 5 rows:")
        print(df.head().to_string())
        
    except Exception as e:
        print(f"An error occurred: {e}")
