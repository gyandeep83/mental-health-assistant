import os
from dotenv import load_dotenv, find_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, ConversationChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import SystemMessage, HumanMessage
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from transformers import pipeline

# Load environment variables
load_dotenv(find_dotenv())
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

DB_FAISS_PATH = "vectorstore/db_faiss"
GROQ_MODEL = "llama-3.3-70b-versatile"

# Prompt templates
RAG_PROMPT_TEMPLATE = """
You're a helpful assistant answering questions based on the provided context only.
If the context doesn't contain the answer, reply: "I'm not sure based on what I found."

Context:
{context}

Conversation History:
{chat_history}

Question:
{question}

Answer (no greetings, focus just on factual information):
"""

DIRECT_INFO_PROMPT = """
You are providing factual information about mental health topics.
Answer the question based solely on the provided context.
Do NOT assume the user is personally affected by this topic or experiencing it themselves.
Use an educational, objective tone appropriate for general knowledge sharing.
If the answer isn't found in the context, say: "I don't have enough information to answer that fully."

Context:
{context}

Conversation History:
{chat_history}

Question:
{question}

Answer:
"""

EMPATHETIC_SYSTEM_PROMPT = """
You are Medibot, a compassionate and emotionally intelligent mental health assistant.
Your role is to be a warm, supportive companion when someone is feeling vulnerable or hurt.

Respond with genuine empathy, acknowledging the user's emotional experience.
Avoid repeating phrases. Use natural, human-like language.
Focus on validation and emotional support.
"""

BLENDING_PROMPT_TEMPLATE = """
Create a cohesive response that naturally blends these elements:

Emotional recognition: {emotional}

Factual information: {factual}

User query: {user_input}

Provide a natural-sounding response that:
1. First acknowledges emotions and validates the user's experience (if emotional content is present)
2. Then provides helpful factual information (if there's an informational request)
3. Ends with supportive guidance or a gentle invitation to share more if they wish

Keep the response concise and focused on the user's needs. Don't ask too many questions in succession.
If there's practical information the user has asked for, make sure to provide it directly.

Response:
"""

# Load LLM
def load_llm():
    return ChatOpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=0.6,
        max_tokens=1024,
    )

# Load FAISS
def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Load a sentiment analysis model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def get_intent_scores(text):
    result = classifier(text, ["emotional", "informational"])
    scores = {label: score for label, score in zip(result["labels"], result["scores"])}
    return scores

def process_blended_response(user_input, chat_history):
    """
    Process user input and return a blended response of emotional and factual components.
    
    Args:
        user_input (str): The user's message
        chat_history (list): A list of dictionaries with "user" and "ai" keys
        
    Returns:
        tuple: (response_text, source_documents)
    """
    llm = load_llm()
    vectorstore = load_vectorstore()
    
    # Get emotional and informational scores
    intent_scores = get_intent_scores(user_input)
    emotional_score = intent_scores.get("emotional", 0)
    informational_score = intent_scores.get("informational", 0)
    
    # Detect if there's a question or if clear informational keywords are present
    has_question = "?" in user_input
    info_keywords = ["how", "what", "why", "when", "where", "who", "can", "should", "would", "could", "is", "are", 
                    "explain", "describe", "difference", "between", "compare", "tell me about", "define"]
    has_info_keywords = any(keyword in user_input.lower() for keyword in info_keywords)
    
    # Check for emotional markers in the text
    emotional_markers = ["feel", "felt", "feeling", "sad", "anxious", "depressed", "worried", "stressed", 
                         "overwhelmed", "scared", "afraid", "nervous", "upset", "desperate", "hopeless",
                         "tired", "exhausted", "frightened", "lonely", "isolated", "angry", "frustrated"]
    has_emotional_markers = any(marker in user_input.lower() for marker in emotional_markers)
    
    # Initialize response components
    emotional_component = ""
    factual_component = ""
    source_docs = []
    
    # Determine if this is primarily an informational query
    is_primarily_informational = (
        (informational_score > emotional_score * 1.2) or  # Score is significantly higher
        (has_question and not has_emotional_markers) or   # Has question but no emotional markers
        (has_info_keywords and not has_emotional_markers) # Has info keywords but no emotional markers
    )
    
    # Process according to primary intent
    if is_primarily_informational:
        # This is a primarily informational query - use direct RAG with minimal emotional content
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 15, "fetch_k": 50})
        
        # Use a more direct factual prompt for purely informational queries
        direct_info_prompt = """
        Provide a clear, factual answer to the following question based on the provided context.
        Do NOT assume the user is personally affected by this topic or experiencing it themselves.
        Respond in an educational, objective tone appropriate for general knowledge sharing.
        
        Context:
        {context}
        
        Conversation History:
        {chat_history}
        
        Question:
        {question}
        
        Answer:
        """
        
        rag_prompt = PromptTemplate(
            input_variables=["context", "question", "chat_history"],
            template=direct_info_prompt,
        )
        
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
        for turn in chat_history:
            memory.chat_memory.add_user_message(turn["user"])
            memory.chat_memory.add_ai_message(turn["ai"])
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": rag_prompt},
        )
        
        result = qa_chain.invoke({
            "question": user_input,
            "chat_history": memory.buffer
        })
        
        factual_component = result["answer"]
        source_docs = result.get("source_documents", [])
        
        # For informational queries, we don't need the blending prompt - just return the factual answer
        return factual_component, source_docs
    
    else:
        # This query has significant emotional content - proceed with blended approach
        
        # Process emotional component if needed
        if emotional_score > 0.3 or has_emotional_markers:
            empathy_memory = ConversationBufferMemory(memory_key="history", return_messages=True)
            for turn in chat_history:
                empathy_memory.chat_memory.add_user_message(turn["user"])
                empathy_memory.chat_memory.add_ai_message(turn["ai"])
                
            empathy_prompt = PromptTemplate(
                input_variables=["history", "input"],
                template=f"{EMPATHETIC_SYSTEM_PROMPT}\n\n{{history}}\nUser: {{input}}\nAI:"
            )
            
            empathy_chain = LLMChain(
                llm=llm,
                prompt=empathy_prompt,
                memory=empathy_memory,
                verbose=False,
            )
            
            result = empathy_chain.invoke({"input": user_input})
            emotional_component = result["text"]
        
        # Process informational component if needed
        if has_question or has_info_keywords or informational_score > 0.3:
            retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 15, "fetch_k": 50})
            rag_prompt = PromptTemplate(
                input_variables=["context", "question", "chat_history"],
                template=RAG_PROMPT_TEMPLATE,
            )
            
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
            for turn in chat_history:
                memory.chat_memory.add_user_message(turn["user"])
                memory.chat_memory.add_ai_message(turn["ai"])
            
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=memory,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": rag_prompt},
            )
            
            result = qa_chain.invoke({
                "question": user_input,
                "chat_history": memory.buffer
            })
            
            factual_component = result["answer"]
            source_docs = result.get("source_documents", [])
        
        # Blend the responses
        blending_prompt = PromptTemplate(
            input_variables=["emotional", "factual", "user_input"],
            template=BLENDING_PROMPT_TEMPLATE
        )
        
        blending_chain = LLMChain(llm=llm, prompt=blending_prompt)
        
        final_response = blending_chain.invoke({
            "emotional": emotional_component,
            "factual": factual_component,
            "user_input": user_input
        })["text"]
        
        return final_response, source_docs