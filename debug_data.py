from analyst_core import load_data
import pandas as pd

df = load_data("Instagram_Final_Data.xlsx")

with open("debug_output.txt", "w", encoding="utf-8") as f:
    f.write("COLUMNS:\n")
    f.write(str(list(df.columns)) + "\n\n")
    
    f.write("DTYPES:\n")
    f.write(str(df.dtypes) + "\n\n")
    
    f.write("HEAD (10):\n")
    f.write(df.head(10).to_string() + "\n")
