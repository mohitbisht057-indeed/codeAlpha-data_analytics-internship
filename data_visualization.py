import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Create scatter plot
sns.scatterplot(
    x='sepal length (cm)',
    y='petal length (cm)',
    data=df
)

# Add title
plt.title("Sepal Length vs Petal Length")

# Display graph
plt.show()