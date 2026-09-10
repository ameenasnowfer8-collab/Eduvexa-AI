import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="Eduvexa AI",
    page_icon="📚",
    layout="centered"
)

st.title("Eduvexa AI")
st.subheader("Your AI-Powered Student Learning Assistant")

st.write(
    "Ask questions about Python, AI, DBMS, mathematics, "
    "projects, or other learning topics."
)

# Load model
@st.cache_resource
def load_model():
    return pipeline(
        "text-generation",
        model="HuggingFaceTB/SmolLM2-360M-Instruct"
    )

# Load the model
with st.spinner("Loading Eduvexa AI..."):
    model = load_model()

# Question box
user_input = st.text_area(
    "Enter your question",
    placeholder="Example: Explain Python loops in simple words."
)

# Button
if st.button("Ask Eduvexa AI"):

    if user_input.strip():

        prompt = f"""You are Eduvexa AI, a helpful student learning assistant.

Answer the student's question clearly and simply.

Student Question:
{user_input}

Answer:"""

        with st.spinner("Eduvexa AI is thinking..."):

            response = model(
                prompt,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7
            )

        answer = response[0]["generated_text"]

        if "Answer:" in answer:
            answer = answer.split("Answer:", 1)[1].strip()

        st.success("Eduvexa AI")
        st.write(answer)

    else:
        st.warning("Please enter a question first.")