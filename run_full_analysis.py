from analyst_core import load_data, clean_dataframe
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_theme(style="whitegrid")

def save_plot(fig, filename):
    fig.savefig(filename, bbox_inches='tight')
    print(f"Saved plot: {filename}")

def run_analysis():
    file_path = "Instagram_Final_Data.xlsx"
    print(f"Loading {file_path}...")
    df = load_data(file_path)
    
    print("Cleaning data...")
    df = clean_dataframe(df)
    
    # Open report file
    with open("analysis_report.md", "w", encoding="utf-8") as f:
        f.write("# Instagram Data Analysis Report\n\n")
        
        # 1. Descriptive Statistics
        f.write("## 1. Descriptive Statistics\n")
        # Select numeric columns of interest
        numeric_cols = ['likes', 'comments', 'views', 'Username_Followers', 'Collaborator_Followers']
        stats = df[numeric_cols].describe().T
        f.write(stats.to_markdown() + "\n\n")
        
        # 2. Correlation Matrix
        print("\nGenerating Correlation Matrix...")
        plt.figure(figsize=(10, 8))
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix')
        save_plot(plt, 'correlation_matrix.png')
        plt.close()
        f.write("## 2. Correlation Matrix\n")
        f.write("![Correlation Matrix](correlation_matrix.png)\n\n")
        
        # 3. Scatter Plot: Views vs Likes
        print("\nGenerating Scatter Plot (Views vs Likes)...")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='views', y='likes', alpha=0.6)
        plt.title('Views vs Likes')
        plt.xlabel('Views')
        plt.ylabel('Likes')
        save_plot(plt, 'views_vs_likes.png')
        plt.close()
        f.write("## 3. Views vs Likes\n")
        f.write("![Views vs Likes](views_vs_likes.png)\n\n")
        
        # 4. Top 5 Performing Posts (by Views)
        f.write("## 4. Top 5 Posts by Views\n")
        top_5 = df.sort_values('views', ascending=False).head(5)
        f.write("| Rank | Views | Likes | User | URL |\n")
        f.write("|---|---|---|---|---|\n")
        for idx, row in top_5.iterrows():
            f.write(f"| {idx+1} | {row['views']:,.0f} | {row['likes']:,.0f} | {row['Username']} | [Link]({row['URL']}) |\n")
            
    print("Analysis complete. Report saved to 'analysis_report.md'.")

if __name__ == "__main__":
    run_analysis()
