import os
import faiss
import numpy as np
from typing import TypedDict, List, Optional, Dict, Any, Annotated
from typing_extensions import Annotated
from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from IPython.display import Image, display

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


os.environ["USER_AGENT"] = "Research-Agent/1.0"

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is missing. Check your .env file.")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Check your .env file.")

class AgentState(TypedDict, total=False):
    query: str
    search_results: List[Dict[str, Any]]
    web_content: List[Document]
    processed_chunks: List[Document]
    research_summary: str
    final_answer: str
    error: Optional[Annotated[str, "append"]]  


tavily_search = TavilySearchResults(api_key=TAVILY_API_KEY)
researcher_llm = ChatGroq(api_key=GROQ_API_KEY, model_name="mixtral-8x7b-32768")
drafter_llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama3-70b-8192")


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vs = None


def research_agent(state: AgentState) -> Dict[str, Any]:
    print("🔍 Research Agent: Gathering and processing information...")
    try:
        
        search_results = tavily_search.invoke({
            "query": state["query"],
            "max_results": 5
        })
        
        
        web_content = []
        for result in search_results:
            try:
                if "url" in result:
                    loader = WebBaseLoader(result["url"])
                    docs = loader.load()
                    
                    for doc in docs:
                        doc.metadata["source"] = result["url"]
                        doc.metadata["title"] = result.get("title", "Unknown Title")
                    web_content.extend(docs)
            except Exception as e:
                print(f"Error loading content from {result.get('url')}: {e}")
                continue
        
        if not web_content:
            return {"error": "No web content could be retrieved from the search results."}
            
        
        all_chunks = []
        for doc in web_content:
            chunks = text_splitter.split_documents([doc])
            all_chunks.extend(chunks)
        
        if not all_chunks:
            return {"error": "No chunks created from web content."}
            
        
        texts = [doc.page_content for doc in all_chunks]
        metadatas = [doc.metadata for doc in all_chunks]
        
        vector_store = FAISS.from_texts(
            texts=texts,
            embedding=embedding_model,
            metadatas=metadatas
        )
        
        
        global vs
        vs = vector_store
        
        
        chunks_text = "\n\n".join([
            f"Source: {chunk.metadata.get('source', 'Unknown')}\nTitle: {chunk.metadata.get('title', 'Untitled')}\nContent: {chunk.page_content[:500]}..." 
            for chunk in all_chunks[:10]
        ])
        
        synthesis_prompt = f"""
        You're a research analyst tasked with synthesizing information from multiple sources.
        
        QUERY: {state["query"]}
        
        SOURCES:
        {chunks_text}
        
        Provide a detailed research summary that addresses the query. Include key findings, trends, 
        and insights from the sources. Structure your summary with clear sections and highlight any 
        contradictions or knowledge gaps.
        """
        
        response = researcher_llm.invoke(synthesis_prompt)
        summary = response.content if hasattr(response, 'content') else str(response)
        
        return {
            "search_results": search_results,
            "web_content": web_content,
            "processed_chunks": all_chunks,
            "research_summary": summary
        }
    except Exception as e:
        return {"error": f"Research agent error: {str(e)}"}

def answer_drafting_agent(state: AgentState) -> Dict[str, Any]:
    print("✍️ Answer Drafting Agent: Drafting final response...")
    try:
        if "research_summary" not in state or not state["research_summary"]:
            return {"error": "No research summary available for drafting."}
            
        relevant_text = "No specific details available."
        global vs
        try:
            if "processed_chunks" in state and state["processed_chunks"] and vs is not None:
                relevant_docs = vs.similarity_search(state["query"], k=3)
                relevant_text = "\n\n".join([
                    f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}" 
                    for doc in relevant_docs
                ])
        except Exception as e:
            print(f"Warning: Could not retrieve similar documents: {e}")
        
        drafting_prompt = f"""
        You're a specialized AI trained to provide comprehensive answers based on research.
        
        QUERY: {state["query"]}
        
        RESEARCH SUMMARY:
        {state["research_summary"]}
        
        ADDITIONAL RELEVANT DETAILS:
        {relevant_text}
        
        Provide a well-structured, thorough answer to the query. Your response should:
        1. Directly address the main question
        2. Include specific facts, data points, and examples from the research
        3. Organize information logically with clear sections
        4. Include citations to sources where appropriate
        5. Maintain academic rigor and objectivity
        
        Format your answer in a clean, professional style suitable for researchers.
        """
        
        response = drafter_llm.invoke(drafting_prompt)
        answer = response.content if hasattr(response, 'content') else str(response)
        return {"final_answer": answer}
    except Exception as e:
        return {"error": f"Answer drafting agent error: {str(e)}"}

def fallback_agent(state: AgentState) -> Dict[str, Any]:
    print("🔄 Fallback Agent: Generating direct response...")
    try:
        fallback_prompt = f"""
        You're tasked with answering a research question directly, without the benefit of web search results.
        
        QUERY: {state["query"]}
        
        Please provide the best answer you can based on your knowledge. Be honest about limitations and 
        indicate where additional research would be beneficial. Structure your answer clearly and logically.
        """
        
        response = drafter_llm.invoke(fallback_prompt)
        answer = response.content if hasattr(response, 'content') else str(response)
        
        return {"final_answer": f"NOTE: The research system encountered issues while gathering information. Here's an answer based on general knowledge:\n\n{answer}"}
    except Exception as e:
        return {"final_answer": f"An error occurred while processing your query. The research pipeline failed, and the fallback system also encountered an error: {str(e)}"}

def error_handler(state: AgentState) -> Dict[str, Any]:
    print(f"❌ Error: {state.get('error', 'Unknown error')}")
    return {}

def should_handle_error(state: AgentState) -> str:
    if "error" in state and state["error"]:
        return "error_handler"
    return "continue"

workflow = StateGraph(AgentState)

workflow.add_node("research", research_agent)
workflow.add_node("answer_drafting", answer_drafting_agent)
workflow.add_node("error_handler", error_handler)
workflow.add_node("fallback", fallback_agent)

workflow.add_edge("research", "answer_drafting")
workflow.add_edge("answer_drafting", END)

workflow.add_conditional_edges(
    "research",
    should_handle_error,
    {
        "error_handler": "error_handler",
        "continue": "answer_drafting"
    }
)
workflow.add_conditional_edges(
    "answer_drafting",
    should_handle_error,
    {
        "error_handler": "error_handler",
        "continue": END
    }
)

workflow.add_edge("error_handler", "fallback")
workflow.add_edge("fallback", END)

workflow.set_entry_point("research")

graph = workflow.compile()


try:
    display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
except Exception:
    pass
def run_research_system(query: str) -> str:
    """Run the research system with the given query."""
    initial_state = {"query": query}
    result = graph.invoke(initial_state)
    return result.get("final_answer", "No answer generated.")

if __name__ == "__main__":
    query = "What is quantum computing?"
    print("\n🔬 Starting Deep Research System...")
    print(f"📝 Query: {query}")
    
    answer = run_research_system(query)
    
    print("\n✅ Final Answer:")
    print(answer)