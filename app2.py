from io import StringIO
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
import scipy.cluster.hierarchy as sch
import seaborn as sns

def app():
    # Load the Iris dataset
    def load_data():
        return pd.read_csv('iris.csv')
    
    iris = load_data()
    
    # Streamlit app
    st.title("🍃 Iris Dataset Clustering and Visualization 🍃")
    st.markdown("""<style>body {background-color: #f9f9f9;}</style>""", unsafe_allow_html=True)

    # Display dataset
    st.subheader("\U0001F4CA Dataset Preview")
    st.dataframe(iris, height=400)

    # Dataset info
    st.subheader("\u2139\ufe0f Dataset Information")
    st.write("**Dataset Shape**: ", iris.shape)
    st.write("**Columns and Data Types**: ")
    st.write(iris.dtypes)

    # Summary statistics
    st.write("**Summary Statistics:**")
    st.write(iris.describe())

    # Boxplots for Outlier Detection
    st.subheader("\U0001F4E6 Outlier Detection: Boxplots")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)

    # Boxplot for Sepal Width
    iris.boxplot(column=['SepalWidthCm'], ax=axes[0], patch_artist=True, 
                boxprops=dict(facecolor='blue'))
    axes[0].set_title("Sepal Width")

    # Boxplot for Sepal Length
    iris.boxplot(column=['SepalLengthCm'], ax=axes[1], patch_artist=True, 
                boxprops=dict(facecolor='blue'))
    axes[1].set_title("Sepal Length")

    st.pyplot(fig)

    # Handle outliers in Sepal Width
    st.subheader("\U0001F6E0\ufe0f Outlier Correction for Sepal Width")
    iris['SepalWidthCm'] = np.where(iris['SepalWidthCm'] > iris['SepalWidthCm'].quantile(0.90),
                                    iris['SepalWidthCm'].quantile(0.50), iris['SepalWidthCm'])
    iris['SepalWidthCm'] = np.where(iris['SepalWidthCm'] < iris['SepalWidthCm'].quantile(0.05),
                                    iris['SepalWidthCm'].quantile(0.50), iris['SepalWidthCm'])

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)
    iris.boxplot(column=['SepalWidthCm'], ax=axes[0], patch_artist=True, boxprops=dict(facecolor='blue'))
    axes[0].set_title("Before Correction")
    iris.boxplot(column=['SepalWidthCm'], ax=axes[1], patch_artist=True, boxprops=dict(facecolor='blue'))
    axes[1].set_title("After Correction")
    st.pyplot(fig)

    # One-hot encoding for Species
    dummies = pd.get_dummies(iris['Species'], prefix='Species')
    iris = pd.concat([iris, dummies], axis=1)
    iris.drop('Species', axis=1, inplace=True)

    # Scaling
    scaler = MinMaxScaler()
    iris_scaled = scaler.fit_transform(iris)

    # Interactive selection for number of clusters
    num_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=3)

    # KMeans Clustering
    st.subheader("\U0001F50D K-Means Clustering")

    # Elbow Method and Silhouette Analysis
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), tight_layout=True)

    # Elbow Method
    model = KMeans(n_init=10)  
    elbow = KElbowVisualizer(model, k=(1, 10), ax=axes[0])
    elbow.fit(iris_scaled)
    axes[0].set_title("Elbow Method")

    # Silhouette Analysis
    silhouette = SilhouetteVisualizer(KMeans(n_clusters=num_clusters, n_init=10), ax=axes[1])  
    silhouette.fit(iris_scaled)
    axes[1].set_title("Silhouette Analysis")

    st.pyplot(fig)

    # Dendrogram
    st.subheader("\U0001F332 Hierarchical Clustering: Dendrogram")
    fig, ax = plt.subplots(figsize=(10, 7))
    sch.dendrogram(sch.linkage(iris_scaled, method='ward'), ax=ax, color_threshold=0.7 * max(sch.linkage(iris_scaled, method='ward')[:, 2]))
    st.pyplot(fig)

    # Cluster visualization
    st.subheader("\U0001F4CC Cluster Visualization")
    kmeans = KMeans(n_clusters=num_clusters, n_init=10) 
    labels = kmeans.fit_predict(iris_scaled)
    cluster_centers = kmeans.cluster_centers_

    # Scatter plot
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("Set2", num_clusters)
    for i in range(num_clusters):
        cluster_data = iris_scaled[labels == i]
        ax.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {i+1}', color=colors[i])

    ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=100, c='black', label='Centroids')
    ax.legend()
    st.pyplot(fig)

    # Pairplot and Heatmap for correlations
    st.subheader("\U0001F4D9 Correlation Heatmap and Pairplot")
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(iris.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

