import os
import streamlit as st
import requests
import json
from datetime import datetime

# Constants
API_BASE_URL = "http://localhost:8000/api"

def format_timestamp(timestamp_str):
    """Format ISO timestamp to human-readable format"""
    dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
    return dt.strftime("%b %d, %Y %I:%M %p")

def main():
    st.set_page_config(
        page_title="Medibot - Mental Health Chat Companion",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS for modern chatbot UI
    st.markdown("""
    <style>
    /* Global dark theme */
    .stApp {
        background-color: #0d1117 !important;
    }
    
    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1000px;
        background-color: #0d1117 !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #1e1e1e !important;
        border-right: 1px solid #333;
    }
    
    section[data-testid="stSidebar"] > div {
        background-color: #1e1e1e !important;
    }
    
    /* Sidebar text color */
    section[data-testid="stSidebar"] .stMarkdown {
        color: #ffffff !important;
    }
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p {
        color: #ffffff !important;
    }
    
    /* Main content text colors */
    .main .stMarkdown,
    .main h1, .main h2, .main h3, .main h4, .main h5, .main h6,
    .main p, .main div, .main span {
        color: #e6edf3 !important;
    }
    
    /* Welcome section styling */
    .welcome-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 60vh;
        text-align: center;
        padding: 2rem;
        background: transparent;
    }
    
    .welcome-title {
        font-size: 2.5rem;
        font-weight: 600;
        color: #e6edf3 !important;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #8b5cf6, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
    }
    
    .welcome-subtitle {
        font-size: 1.1rem;
        color: #8b949e !important;
        margin-bottom: 3rem;
        line-height: 1.6;
        max-width: 600px;
        text-align: center;
    }
    
    .suggestion-card {
        background: #21262d;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        cursor: pointer;
        transition: all 0.2s ease;
        text-align: left;
    }
    
    .suggestion-card:hover {
        background: #262c36;
        border-color: #8b5cf6;
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(139, 92, 246, 0.15);
    }
    
    .suggestion-title {
        font-size: 1rem;
        font-weight: 600;
        color: #e6edf3 !important;
        margin-bottom: 0.5rem;
    }
    
    .suggestion-desc {
        font-size: 0.875rem;
        color: #8b949e !important;
        line-height: 1.4;
    }
    
    /* Chat input area styling */
    .chat-input-container {
        position: fixed;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: calc(100% - 320px);
        max-width: 1000px;
        padding: 1.5rem 2rem;
        background: linear-gradient(to top, #0d1117 80%, transparent);
        z-index: 10;
    }
    
    .chat-input-wrapper {
        background: #21262d;
        border: 1px solid #30363d;
        border-radius: 24px;
        padding: 0.75rem 1.5rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        transition: all 0.2s ease;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    .chat-input-wrapper:focus-within {
        border-color: #8b5cf6;
        box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.1), 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Hide default streamlit form elements */
    .stForm {
        background: transparent !important;
        border: none !important;
    }
    
    /* Custom text area styling */
    .stTextArea > div > div > textarea {
        background: transparent !important;
        border: none !important;
        color: #e6edf3 !important;
        font-size: 1rem !important;
        line-height: 1.5 !important;
        resize: none !important;
        outline: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        min-height: 68px !important;
        max-height: 120px !important;
    }
    
    .stTextArea > div > div > textarea::placeholder {
        color: #8b949e !important;
    }
    
    .stTextInput > div > div > input {
        background: transparent !important;
        border: none !important;
        color: #e6edf3 !important;
        font-size: 1rem !important;
        outline: none !important;
        box-shadow: none !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #8b949e !important;
    }
    
    /* Send button styling */
    .send-button {
        background: #8b5cf6 !important;
        border: none !important;
        border-radius: 12px !important;
        color: white !important;
        padding: 0.5rem 1rem !important;
        font-weight: 500 !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        min-width: 60px !important;
        height: 36px !important;
    }
    
    .send-button:hover {
        background: #7c3aed !important;
        transform: translateY(-1px) !important;
    }
    
    .send-button:disabled {
        background: #30363d !important;
        cursor: not-allowed !important;
        transform: none !important;
    }
    
    /* Chat message styling */
    .user-message {
        background-color: #2563eb;
        color: white;
        border-radius: 15px;
        padding: 10px 15px;
        margin: 5px 0;
        text-align: right;
        border: 1px solid #1d4ed8;
        max-width: 80%;
        margin-left: auto;
    }
    
    .assistant-message {
        background-color: #21262d;
        color: #e6edf3;
        border-radius: 15px;
        padding: 10px 15px;
        margin: 5px 0;
        border: 1px solid #30363d;
        max-width: 80%;
        margin-right: auto;
    }
    
    /* Assistant message container to control width of child elements */
    .assistant-message-container {
        max-width: 80%;
        margin-right: auto;
    }
    
    /* Fix expander to stay within assistant message container width */
    .assistant-message-container .stExpander {
        width: 100% !important;
        max-width: 100% !important;
    }
    
    .assistant-message-container .stExpander > div {
        width: 100% !important;
        max-width: 100% !important;
    }
    
    .assistant-message-container .stExpander > div > div {
        width: 100% !important;
        max-width: 100% !important;
    }
    
    /* New chat button styling */
    section[data-testid="stSidebar"] button[kind="primary"] {
        background-color: #2563eb !important;
        border: none !important;
        border-radius: 8px !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.75rem 1rem !important;
        transition: all 0.2s ease !important;
        margin-bottom: 1rem !important;
    }
    
    section[data-testid="stSidebar"] button[kind="primary"]:hover {
        background-color: #1d4ed8 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(37, 99, 235, 0.3) !important;
    }
    
    /* Chat history buttons - compact styling */
    section[data-testid="stSidebar"] button[kind="secondary"] {
        background-color: transparent !important;
        border: 1px solid transparent !important;
        border-radius: 6px !important;
        color: #d1d5db !important;
        font-weight: 400 !important;
        padding: 0.4rem 0.5rem !important;
        transition: all 0.2s ease !important;
        text-align: left !important;
        font-size: 0.875rem !important;
        box-shadow: none !important;
        margin-bottom: 0.1rem !important;
        min-height: 2rem !important;
        width: 100% !important;
        line-height: 1.2 !important;
    }
    
    /* Hover effect for chat history */
    section[data-testid="stSidebar"] button[kind="secondary"]:hover {
        background-color: #374151 !important;
        color: #ffffff !important;
        border: 1px solid #4b5563 !important;
    }
    
    /* Remove focus outline */
    section[data-testid="stSidebar"] button[kind="secondary"]:focus {
        outline: none !important;
        box-shadow: none !important;
    }
    
    /* Virtual conversation indicator */
    .virtual-conversation-indicator {
        color: #10b981;
        font-weight: 500;
        font-size: 0.875rem;
        margin-bottom: 0.5rem;
        padding: 0.25rem 0.5rem;
        background-color: rgba(16, 185, 129, 0.1);
        border-radius: 4px;
        border-left: 3px solid #10b981;
    }
    
    /* Active conversation styling */
    .active-conversation {
        background-color: #374151 !important;
        color: #ffffff !important;
        font-weight: 500 !important;
        border-left: 3px solid #2563eb !important;
    }
    
    /* Logout button styling */
    .logout-button {
        background-color: #dc2626 !important;
        border: none !important;
        border-radius: 8px !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.2s ease !important;
        margin-top: 1rem !important;
    }
    
    .logout-button:hover {
        background-color: #b91c1c !important;
        transform: translateY(-1px) !important;
    }
    
    /* Reduce spacing in sidebar sections */
    section[data-testid="stSidebar"] .element-container {
        margin-bottom: 0.25rem !important;
    }
    
    /* Compact chat history section */
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > [style*="flex-direction: column"] > div {
        gap: 0.1rem !important;
    }
    
    /* Hide scrollbar for cleaner look */
    ::-webkit-scrollbar {
        width: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0d1117;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #30363d;
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #484f58;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state variables
    if "token" not in st.session_state:
        st.session_state.token = None
    
    if "current_conversation_id" not in st.session_state:
        st.session_state.current_conversation_id = None
    
    if "conversations" not in st.session_state:
        st.session_state.conversations = []
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    if "should_load_messages" not in st.session_state:
        st.session_state.should_load_messages = False
        
    if "last_conversation_reload" not in st.session_state:
        st.session_state.last_conversation_reload = None
    
    # New flag to track if we just logged in
    if "just_logged_in" not in st.session_state:
        st.session_state.just_logged_in = False
    
    # Flag to track if we're in a "virtual" new conversation (not yet saved to DB)
    if "is_virtual_conversation" not in st.session_state:
        st.session_state.is_virtual_conversation = False
    
    # Helper functions
    def create_virtual_new_chat():
        """Create a virtual new chat that doesn't get saved until first message"""
        st.session_state.current_conversation_id = "virtual_new"
        st.session_state.messages = []
        st.session_state.should_load_messages = False
        st.session_state.is_virtual_conversation = True
    
    def create_actual_conversation(user_input):
        """Create the actual conversation in the database when user sends first message"""
        headers = {"Authorization": f"Token {st.session_state.token}"}
        try:
            response = requests.post(
                f"{API_BASE_URL}/chatbot/conversations/", 
                headers=headers, 
                json={"title": "New Conversation"}
            )
            response.encoding = 'utf-8'  # <- Add this line before you access response.text or response.json()
            
            if response.status_code == 201:
                new_conversation = response.json()
                # Update session state to reflect the real conversation
                st.session_state.current_conversation_id = new_conversation["id"]
                st.session_state.is_virtual_conversation = False
                
                # Add to conversations list
                st.session_state.conversations.insert(0, new_conversation)
                
                return new_conversation["id"]
        except Exception as e:
            st.error(f"Error creating conversation: {str(e)}")
            return None
    
    def select_conversation(conv_id):
        if st.session_state.current_conversation_id != conv_id:
            st.session_state.current_conversation_id = conv_id
            st.session_state.should_load_messages = True
            st.session_state.is_virtual_conversation = False  # Real conversation
    
    def load_conversation_messages():
        if not st.session_state.should_load_messages:
            return
            
        headers = {"Authorization": f"Token {st.session_state.token}"}
        try:
            response = requests.get(
                f"{API_BASE_URL}/chatbot/conversations/{st.session_state.current_conversation_id}/", 
                headers=headers
            )
            response.encoding = 'utf-8'  # <- Add this line
            if response.status_code == 200:
                data = response.json()
                st.session_state.messages = data["messages"]
                st.session_state.should_load_messages = False  # Reset flag after loading
        except Exception as e:
            st.error(f"Error loading conversation: {str(e)}")
            st.session_state.should_load_messages = False  # Reset flag on error too
    
    def handle_logout():
        st.session_state.token = None
        st.session_state.current_conversation_id = None
        st.session_state.conversations = []
        st.session_state.messages = []
        st.session_state.should_load_messages = False
        st.session_state.just_logged_in = False
        st.session_state.is_virtual_conversation = False
    
    def send_message(user_input):
        if not user_input.strip():
            return
          
        # Remove problematic unicode characters
        user_input = user_input.replace("\u2028", " ").replace("\u2029", " ")
        
        # If this is a virtual conversation, create the actual conversation first
        if st.session_state.is_virtual_conversation:
            actual_conv_id = create_actual_conversation(user_input)
            if not actual_conv_id:
                st.error("Failed to create conversation. Please try again.")
                return
            
            # Refresh conversations list to get the updated list
            headers = {"Authorization": f"Token {st.session_state.token}"}
            try:
                response = requests.get(f"{API_BASE_URL}/chatbot/conversations/", headers=headers)
                response.encoding = 'utf-8'  # <- Add this line before you access response.text or response.json()
                if response.status_code == 200:
                    st.session_state.conversations = response.json()
            except Exception as e:
                st.error(f"Error refreshing conversations: {str(e)}")
            
        # Add user message locally first for immediate feedback
        temp_user_message = {
            "id": -1,  # Temporary ID
            "role": "user",
            "content": user_input,
            "created_at": datetime.now().isoformat()
        }
        st.session_state.messages.append(temp_user_message)
        
        # Send the message to the API
        headers = {"Authorization": f"Token {st.session_state.token}"}
        try:
            response = requests.post(
                f"{API_BASE_URL}/chatbot/conversations/{st.session_state.current_conversation_id}/messages/",
                headers=headers,
                json={"content": user_input}
            )
            response.encoding = 'utf-8'  # <- Add this line
            
            print("Response status code:", response.status_code) 
            if response.status_code == 200:
                data = response.json()
                
                # Update our local state with the server's response
                st.session_state.messages.pop()  # Remove the temporary message
                st.session_state.messages.append(data["user_message"])
                st.session_state.messages.append(data["assistant_message"])
                
                # If the conversation title was updated, refresh it
                current_conversation = next(
                    (c for c in st.session_state.conversations if c["id"] == st.session_state.current_conversation_id), 
                    {"title": "New Conversation"}
                )
                
                if data["conversation_title"] != current_conversation["title"]:
                    # Update the title in our local conversations list
                    for conv in st.session_state.conversations:
                        if conv["id"] == st.session_state.current_conversation_id:
                            conv["title"] = data["conversation_title"]
                            break
            else:
                st.error(f"Error: {response.text.encode('utf-8', errors='replace').decode()}")
                st.session_state.messages.pop()
                # Remove the temporary message since the request failed
                st.session_state.messages.pop()
        except Exception as e:
            st.error(f"Error sending message: {str(e)}")
            # Remove the temporary message since the request failed
            st.session_state.messages.pop()
    
    # Load conversation messages if flag is set (only for real conversations)
    if (st.session_state.should_load_messages and 
        st.session_state.current_conversation_id and 
        not st.session_state.is_virtual_conversation):
        load_conversation_messages()
    
    # Login section (if not logged in)
    if not st.session_state.token:
        st.title("🧠 Medibot - Login")
        
        with st.form("login_form"):
            token = st.text_input("Enter your Django auth token:", type="password")
            submitted = st.form_submit_button("Login")
            
            if submitted and token:
                # Validate token by making a request to the API
                headers = {"Authorization": f"Token {token}"}
                try:
                    response = requests.get(f"{API_BASE_URL}/chatbot/conversations/", headers=headers)
                    response.encoding = 'utf-8' 
                    if response.status_code == 200:
                        st.session_state.token = token
                        st.session_state.conversations = response.json()
                        st.session_state.just_logged_in = True  # Mark that user just logged in
                        st.rerun()
                    else:
                        st.error("Invalid token. Please try again.")
                except Exception as e:
                    st.error(f"Error connecting to the server: {str(e)}")
    
    # Main chatbot interface (when logged in)
    else:
        # Handle first login - automatically create a virtual new conversation
        if st.session_state.just_logged_in:
            create_virtual_new_chat()
            st.session_state.just_logged_in = False
            st.rerun()
        
        # Sidebar with conversation history
        with st.sidebar:
            st.title("🧠 Medibot")
            
            # New Chat button
            if st.button("➕ New Chat", key="new_chat_btn", use_container_width=True, type="primary"):
                create_virtual_new_chat()
                st.rerun()
            
            st.markdown("---")
            
            # Conversation history section
            st.subheader("Chat History")
            
            # Show virtual conversation indicator
            if st.session_state.is_virtual_conversation:
                st.markdown('<div class="virtual-conversation-indicator">● New Conversation (unsaved)</div>', unsafe_allow_html=True)
            
            # List conversations with improved styling
            if st.session_state.conversations:
                for i, conv in enumerate(st.session_state.conversations):
                    conv_id = conv["id"]
                    title = conv["title"]
                    is_selected = (
                        st.session_state.current_conversation_id == conv_id and 
                        not st.session_state.is_virtual_conversation
                    )
                    
                    # Create a unique key for each conversation button
                    button_key = f"conv_btn_{conv_id}_{i}"
                    
                    # Use a simpler approach without label_visibility
                    if is_selected:
                        # Selected conversation - show as highlighted but disabled
                        st.markdown(f"""
                        <div style="
                            background-color: #374151;
                            color: #ffffff;
                            font-weight: 500;
                            border: 1px solid #2563eb;
                            box-shadow: 0 0 0 1px #2563eb;
                            border-radius: 6px;
                            padding: 0.4rem 0.5rem;
                            margin-bottom: 0.1rem;
                            min-height: 2rem;
                            line-height: 1.2;
                            font-size: 0.875rem;
                            display: flex;
                            align-items: center;
                        ">
                            {title}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Regular button for non-selected conversations
                        button_clicked = st.button(
                            title,
                            key=button_key,
                            use_container_width=True,
                            help=f"Switch to: {title}",
                            type="secondary"
                        )
                        
                        if button_clicked:
                            select_conversation(conv_id)
                            st.rerun()
            else:
                st.info("No previous conversations")

            # Logout button at the bottom of the sidebar
            st.markdown("---")
            if st.button("🚪 Logout", key="logout_button", use_container_width=True):
                handle_logout()
                st.rerun()
        
        # Main chat area
        if st.session_state.current_conversation_id:
            # Chat message area (existing messages)
            if st.session_state.messages:
                chat_container = st.container()
                
                with chat_container:
                    for message in st.session_state.messages:
                        if message["role"] == "user":
                            with st.container():
                                st.markdown(f"""
                                <div class="user-message">
                                    <strong>You:</strong> {message["content"]}
                                </div>
                                """, unsafe_allow_html=True)
                        else:  
                            # assistant
                            # Split content to separate sources if present
                            content = message["content"]
                            sources_section = ""
                            
                            if "---\n\n**Sources:**" in content:
                                main_content, sources = content.split("---\n\n**Sources:**", 1)
                                sources_section = f"**Sources:**{sources}"
                            else:
                                main_content = content
                            
                            # Create a container that matches the assistant message width
                            with st.container():
                                # Wrap everything in a div with controlled width
                                st.markdown('<div class="assistant-message-container">', unsafe_allow_html=True)
                                
                                # Display the assistant's main response
                                st.markdown(f"""
                                <div class="assistant-message">
                                    <strong>Medibot:</strong> {main_content}
                            
                                """, unsafe_allow_html=True)

                                # Show resources in a dropdown if available - within the same container
                                if sources_section.strip():
                                    with st.expander("📚 View Suggested Resources"):
                                        st.markdown(sources_section, unsafe_allow_html=True)
                                
                                st.markdown('</div>', unsafe_allow_html=True)
            
            # Welcome section for empty conversations - FIXED VERSION
            else:
                # Center container
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col2:
                    st.markdown('<div style="text-align: center; padding: 2rem 0;">', unsafe_allow_html=True)
                    
                    # Title with gradient effect
                    st.markdown("""
                    <h1 class="welcome-title">How can I help you today?</h1>
                    """, unsafe_allow_html=True)
                    
                    # Subtitle
                    st.markdown("""
                    <p class="welcome-subtitle">I'm Medibot, your mental health companion. I'm here to provide support, guidance, and resources for your mental wellness journey.</p>
                    """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Suggestion cards using columns for layout
                st.markdown("<br>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    <div class="suggestion-card">
                        <div class="suggestion-title">🧘 Stress Management</div>
                        <div class="suggestion-desc">Learn techniques for managing daily stress and anxiety</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("""  
                    <div class="suggestion-card">
                        <div class="suggestion-title">😊 Mood Support</div>
                        <div class="suggestion-desc">Get help understanding and improving your emotional well-being</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div class="suggestion-card">
                        <div class="suggestion-title">💭 Mindfulness</div>
                        <div class="suggestion-desc">Discover mindfulness practices and meditation guidance</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div class="suggestion-card">
                        <div class="suggestion-title">🛌 Sleep & Wellness</div>
                        <div class="suggestion-desc">Find tips for better sleep and overall mental health</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Fixed bottom input area
            st.markdown('<div style="height: 120px;"></div>', unsafe_allow_html=True)  # Spacer for fixed input
            
            # Chat input using a form - FIXED VERSION
            with st.form(key="chat_input_form", clear_on_submit=True):
                # Use text_input instead of text_area for better mobile experience
                user_input = st.text_input(
                    "",
                    placeholder="Message Medibot...",
                    key="user_input_field",
                    label_visibility="collapsed"
                )
                
                # Create two columns for layout but put submit button in the form
                col1, col2 = st.columns([4, 1])
                
                with col2:
                    # This is the required submit button
                    submit_button = st.form_submit_button(
                        "Send", 
                        use_container_width=True,
                        type="primary"
                    )
                
                # Handle form submission
                if submit_button and user_input:
                    send_message(user_input)
                    st.rerun()
        
        else:
            # No conversation selected - Create a virtual one
            create_virtual_new_chat()
            st.rerun()

if __name__ == "__main__":
    main()
