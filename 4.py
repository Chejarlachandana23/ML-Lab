import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

data = { 
    'student': ['A','B','C','D','E','F','G','H'], 
    'math':[88,92,80,89,100,67,78,85], 
    'science':[90,85,88,95,92,70,72,80], 
    'english': [70,78,85,82,89,60,65,72] 
} 
df = pd.DataFrame(data) 

# 1. Line Plot 
plt.figure(figsize=(8, 5)) 
plt.plot(df['student'], df['math'], marker='o', label='math') 
plt.plot(df['student'], df['science'], marker='s', label='science') 
plt.plot(df['student'], df['english'], marker='^', label='english') 
plt.title("Line plot of student scores") 
plt.xlabel("students") 
plt.ylabel("scores") 
plt.legend() 
plt.show() 

# 2. Bar Chart
plt.figure(figsize=(8,5)) 
plt.bar(df['student'], df['math'], color='skyblue') 
plt.title("Bar chart-math scores") 
plt.xlabel("students") 
plt.ylabel("scores") 
plt.show() 

# 3. Histogram
plt.figure(figsize=(8,5)) 
plt.hist(df['math'], bins=5, color='lightgreen', edgecolor='black') 
plt.title("histogram of the math scores") 
plt.xlabel("score Range") 
plt.ylabel("Frequency") 
plt.show() 

# 4. Scatter Plot
plt.figure(figsize=(8, 5)) 
plt.scatter(df['math'], df['science'], color="red") 
plt.title("scatter plot-math vs science") 
plt.xlabel("Math scores") 
plt.ylabel("science scores") 
plt.show() 

# 5. Box Plot
plt.figure(figsize=(8, 5)) 
sns.boxplot(data=df[['math','science','english']]) 
plt.title("box plot of subject scores") 
plt.ylabel("scores") 
plt.show() 

# 6. Heatmap
plt.figure(figsize=(6, 4)) 
sns.heatmap(df[['math', 'science', 'english']].corr(), annot=True, cmap='coolwarm') 
plt.title("heatmap-correlation Between subjects") 
plt.show()
