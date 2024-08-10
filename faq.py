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

# Initialize session state for the text input and rating
if 'user_question' not in st.session_state:
  st.session_state.user_question = ""

if 'rating' not in st.session_state:
  st.session_state.rating = 3

# Streamlit Interface
st.title("Health Q&A")

# Sidebar for FAQs or search
st.sidebar.title("FAQs")
faq_query = st.sidebar.text_input("Search FAQs:")

if faq_query:
    # Filter FAQs based on the search query
    faq_matches = df[df['Question'].str.contains(faq_query, case=False)]
    for idx, row in faq_matches.iterrows():
        st.sidebar.write(f"**Q:** {row['Question']}")
        st.sidebar.write(f"**A:** {row['Answer']}")
else:
    # Display a few common FAQs
    st.sidebar.write("Here are some common questions:")
    common_faqs = df.sample(5)  # Randomly select 5 questions for display
    for idx, row in common_faqs.iterrows():
        st.sidebar.write(f"**Q:** {row['Question']}")
        st.sidebar.write(f"**A:** {row['Answer']}")
  
# Main question and answer section
user_question = st.text_input("Enter your question:", st.session_state.user_question)

if st.button("Get Answer"):
  if user_question:
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
        st.write(f"**Answer:** {df.iloc[best_match_idx]['Answer']}")
        st.write(f"**Similarity Score:** {best_score:.2f}")
            
        # Rating slider for helpfulness
        st.session_state.rating = st.slider("Rate the helpfulness of the answer:", 1, 5, st.session_state.rating)
        if st.button("Submit Rating"):
            st.write(f"Thank you for your feedback! You rated the answer: {st.session_state.rating}/5")
            # Here you could save the rating to a database or log it
    else:
        st.write("I apologize, but I don't have information on that topic yet. Could you please ask another question?")

if st.button("Clear"):
  st.session_state.user_question = "" # Clear the input field
  st.session_state.rating = 3 # Reset rating
