import streamlit as st
import pandas as pd
import numpy as np
import openai
from sklearn.metrics.pairwise import cosine_similarity

# Set Up API Key for Git
openai.api_key = st.secrets["mykey"]

# Load the dataset
df = pd.read_csv('qa_dataset_with_embeddings.csv')
df['Question_Embedding'] = df['Question_Embedding'].apply(lambda x: np.array(eval(x)))

# Function to generate embeddings for new user question
def generate_embedding(text):
  response = openai.Embedding.create(
      model="text-embedding-ada-002",
      input=text
  )
  return np.array(response['data'][0]['embedding'])

# Streamlit Interface
st.title("Health Q&A")

user_question = st.text_input("Enter your question:")

if st.button("Get Answer"):
  # Generate embedding for user's question
  user_embedding = generate_embedding(user_question)

  # Calculate cosine similarity
  similarities = cosine_similarity([user_embedding], df['Question_Embedding'].tolist())

  # Find the index of the most similar question
  best_match_idx = np.argmax(similarities)
  best_score = similarities[0][best_match_idx]

  # Set a threshold for matching
  threshold = 0.75
  if best_score > threshold:
    st.write(f"Answer: {df.iloc[best_match_idx]['Answer']}")
    st.write(f"Similarity Score: {best_score}")
  else:
    st.write("I apologize, but I don't have information on that topic yet. Could you please ask another question?")

if st.button("Clear"):
  st.session_state.user_question = "" # Clear the input field
  st.experimental_rerun() # Rerun the script to update UI
