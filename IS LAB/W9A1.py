# Import necessary libraries 
import nltk 
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.cluster import KMeans 
from sklearn.manifold import TSNE
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer 
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.cluster import KMeans 

# Download necessary NLTK resources
nltk.download('punkt_tab')  # Tokenizer models
nltk.download('stopwords')  # Stopwords list
nltk.download('wordnet')  # WordNet lexicon

# Sample text for processing
text = ["Natural Language Processing is fun!",
        "Text preprocessing is important for NLP tasks.",
        "Stemming helps reduce words to their root forms.",
        "Clustering algorithms like K-means are important in machine learning.",
        "Clustering text data is useful for text mining applications.",
        "Preprocessing includes tasks like stemming and lemmatization."]

# Tokenization (splitting the text into words)
tokens = [word_tokenize(doc) for doc in text]

# Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_tokens = [[word for word in doc if word.lower() not in stop_words] for doc in tokens]

# Initialize the Porter Stemmer and WordNet Lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Apply stemming to the filtered tokens
stemmed_tokens = [[stemmer.stem(word) for word in doc] for doc in filtered_tokens]

# Apply lemmatization to the filtered tokens
lemmatized_tokens = [[lemmatizer.lemmatize(word) for word in doc] for doc in filtered_tokens]

# Convert tokens back to string format for vectorization
lemmatized_texts = [" ".join(doc) for doc in lemmatized_tokens]

# Apply Bag of Words Vectorization
vectorizer_bow = CountVectorizer()
bow_vectors = vectorizer_bow.fit_transform(lemmatized_texts)

# Apply K-means Clustering
num_clusters = 2  # You can change this based on the number of topics or clusters you expect
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(bow_vectors)

# Get the cluster labels (which document belongs to which cluster)
labels = kmeans.labels_

# Get the centroids (mean vectors of each cluster)
centroids = kmeans.cluster_centers_

# Print the cluster labels for each document
print("Cluster labels for each document:")
for i, label in enumerate(labels):
    print(f"Document {i}: Cluster {label}")

# Print the terms that are closest to the centroids
terms = vectorizer_bow.get_feature_names_out()
for i, centroid in enumerate(centroids):
    print(f"\nCluster {i} centroid words:")
    sorted_centroid = centroid.argsort()  # Sort the centroid to get top terms
    for index in sorted_centroid[:10]:  # Print top 10 words closest to the centroid
        print(f"  {terms[index]}")

# Apply T-SNE for dimensionality reduction (2D) 
tsne = TSNE(n_components=2, random_state=42, perplexity=2)  # Lower perplexity 
reduced_vectors = tsne.fit_transform(bow_vectors.toarray())  # .toarray() converts sparse matrix to dense 
  
# Step 10: Plot the 2D representation of the clusters 
plt.figure(figsize=(8, 6)) 
scatter = plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=labels, 
cmap='viridis') 
  
# Add labels and title 
plt.title("K-means Clustering of Text Data") 
plt.xlabel("T-SNE Component 1") 
plt.ylabel("T-SNE Component 2") 
  
# Show a color bar to indicate cluster assignments 
plt.colorbar(scatter) 
  
# Show the plot 
#plt.show() 
plt.savefig("pngW9A1.png")
