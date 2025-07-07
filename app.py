import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv

load_dotenv() #loading the .env

#Langsmith tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot with Groq"

#Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assitant, please respond to the user's queries and try to keep it concise."),
        ("user", "Question : {question}")
    ]
)

#generate response function
#the temperature(0-1) refers to how much creative the model will be wrt the answers, temp=0 is not creative, temp=1 is creative
def generate_response(question, api_key, llm , temperature, max_tokens):
    llm = ChatGroq(model= llm, groq_api_key=api_key, temperature=temperature, max_tokens=max_tokens)
    #output parser
    output_parser = StrOutputParser()
    #creating chain
    chain = prompt| llm | output_parser
    answer = chain.invoke({'question' : question})
    return answer

#creating the streamlit app
#Title of the app
st.title("Q&A Chatbot with Groq")
#sidebar to enter groq api key
api_key = st.sidebar.text_input("Enter your Groq API Key :", type= "password")

#create a dropdown to select different llm models
#we will provide a list of models that user can choose from
llm = st.sidebar.selectbox("Select a model from Groq", ["Compound-Beta-Mini", "Compound-Beta", "Gemma2-9b-It", "Mistral-Saba-24b", "Qwen/Qwen3-32b", "Llama3-8b-8192", "Deepseek-R1-Distill-Llama-70b"])

#sliders to adjust response parameters
temperature = st.sidebar.slider("Temperature(model randomness and creativity)", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max tokens that you want to spend", min_value=50, max_value=300, value=100)

#Main Interface for user input
st.write("Go ahead and ask any question:")
user_input = st.text_input("You:", placeholder="Ask me anything about tech, science, philosophy...")

if not api_key:
    st.warning("Please enter your Groq API key to use the chatbot.") #to make sure that the user is using their key
elif user_input:
    response= generate_response(question= user_input, api_key = api_key, llm = llm, temperature= temperature, max_tokens= max_tokens)
    st.spinner("Generating answer⏰")
    st.write(response)
else:
    st.write("Waiting for a question...🤔")