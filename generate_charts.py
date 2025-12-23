from analyst_core import load_data, clean_dataframe
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid")

def generate_charts():
    file_path = "Instagram_Final_Data.xlsx"
    print(f"Loading {file_path}...")
    df = load_data(file_path)
    df = clean_dataframe(df)
    
    # 1. Pie Chart: Share of Total Views (Top 5 Users vs Others)
    print("\nGenerating Pie Chart...")
    # Group by Username and sum views
    user_views = df.groupby('Username')['views'].sum().sort_values(ascending=False)
    
    top_5 = user_views.head(5)
    others = pd.Series([user_views.iloc[5:].sum()], index=['Others'])
    pie_data = pd.concat([top_5, others])
    
    plt.figure(figsize=(10, 10))
    plt.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
    plt.title('Share of Total Views by Top 5 Creators')
    plt.savefig('pie_chart_views.png', bbox_inches='tight')
    print("Saved pie_chart_views.png")
    plt.close()
    
    # 2. Bar Graph: Top 10 Users by Followers
    print("\nGenerating Bar Graph...")
    # Get unique users and their max follower count (assuming it's constant per user)
    user_followers = df.groupby('Username')['Username_Followers'].max().sort_values(ascending=False).head(10)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=user_followers.values, y=user_followers.index, palette='viridis')
    plt.title('Top 10 Creators by Follower Count')
    plt.xlabel('Followers')
    plt.ylabel('Creator')
    plt.savefig('bar_chart_followers.png', bbox_inches='tight')
    print("Saved bar_chart_followers.png")
    plt.close()

if __name__ == "__main__":
    generate_charts()
