import gradio as gr
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize tools
arxiv_tool = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=3, doc_content_chars_max=800))
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=500))
search_tool = DuckDuckGoSearchRun()

llm = ChatGroq(
    model="meta-llama/llama-4-maverick-17b-128e-instruct",
    api_key=groq_api_key,
    temperature=0.1
)

def chatbot(message, history):
    """Simple chatbot that uses tools directly"""
    try:
        # Use appropriate tool based on query content
        if any(keyword in message.lower() for keyword in ['arxiv', 'research paper', 'academic']):
            result = arxiv_tool.run(message)
            response = f"**From Arxiv research papers:**\n\n{result}"
        elif any(keyword in message.lower() for keyword in ['wikipedia', 'encyclopedia', 'definition']):
            result = wiki_tool.run(message)
            response = f"**From Wikipedia:**\n\n{result}"
        else:
            # Use web search for general queries
            result = search_tool.run(message)
            response = f"**Search results:**\n\n{result}"
        
        return response
        
    except Exception as e:
        # Fallback to direct LLM response
        fallback = llm.invoke(message).content
        return f"{fallback}\n\n*Note: Used direct knowledge base due to search tool error.*"

demo = gr.ChatInterface(
    fn=chatbot,
    title="💬 Desmond Chat with Search",
    description="🤖 I can search various sources for you!",
    examples=["Latest AI research papers", "What is machine learning?", "Quantum computing explained"]
)

if __name__ == "__main__":
    demo.launch()