import streamlit as st
from llama_cpp import Llama
from rag import RAGPipeline
from tools import get_tools
from agent import RAGAgent
import os
import time
import pickle

# Page config
st.set_page_config(
    page_title="IntelliCode RAG Assistant",
    page_icon="🧠",
    layout="wide"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'rag' not in st.session_state:
    st.session_state.rag = None
if 'code_file' not in st.session_state:
    st.session_state.code_file = None
if 'trigger_query' not in st.session_state:
    st.session_state.trigger_query = None


def process_uploaded_file(uploaded_file, file_path):
    """Process uploaded files based on type"""
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    if file_type == 'pdf':
        import pypdf
        pdf_reader = pypdf.PdfReader(file_path)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n\n"
        
        # Save as txt for RAG processing
        txt_path = file_path.replace('.pdf', '.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text)
        return txt_path
    
    return file_path

@st.cache_resource
def load_system():
    """Load RAG, LLM, and Agent (cached)"""
    with st.spinner("🔄 Loading AI system... (this takes ~30 seconds)"):
        # Load RAG
        rag = RAGPipeline()
        if os.path.exists("vector_store/index.faiss"):
            rag.load_index("vector_store")
        else:
            st.warning("⚠️ No vector store found. Upload documents to build knowledge base.")
        
        # Load LLM
        llm = Llama(
            model_path="models/qwen2.5-3b-instruct-q4_k_m.gguf",
            n_ctx=2048,
            verbose=False
        )
        
        # Initialize tools and agent
        tools = get_tools(rag)
        agent = RAGAgent(llm, tools)
        
        return agent, rag

# Sidebar
with st.sidebar:
    st.title("🧠 IntelliCode RAG")
    st.markdown("AI-Powered Code & Document Assistant")
    st.markdown("---")
    
    # Tab selection
    tab = st.radio("Mode", ["📄 Documents", "💻 Code Analysis"], label_visibility="collapsed")
    
    if tab == "📄 Documents":
        st.subheader("📄 Document Upload")
        doc_file = st.file_uploader(
            "Upload documents",
            type=['txt', 'pdf', 'csv'],
            help="Upload documents to add to knowledge base",
            key='doc_upload'
        )
        
        if doc_file:
            file_type = doc_file.name.split('.')[-1]
            os.makedirs("data", exist_ok=True)
            file_path = os.path.join("data", doc_file.name)
            
            with open(file_path, "wb") as f:
                f.write(doc_file.getbuffer())
            
            st.success(f"✅ Uploaded: {doc_file.name}")
            
            if file_type.lower() == 'pdf':
                with st.spinner("Processing PDF..."):
                    process_uploaded_file(doc_file, file_path)
                    st.success("✅ PDF processed!")
            
            if st.button("🔄 Rebuild Knowledge Base"):
                with st.spinner("Rebuilding vector index..."):
                    rag = RAGPipeline()
                    rag.load_documents("data")
                    rag.build_index()
                    rag.save_index()
                    st.success("✅ Knowledge base updated!")
                    st.cache_resource.clear()
                    st.rerun()
    
    else:  # Code Analysis
        st.subheader("💻 Code Analysis")
        code_file = st.file_uploader(
            "Upload Python file",
            type=['py'],
            help="Upload .py file for analysis",
            key='code_upload'
        )
        
        if code_file:
            os.makedirs("temp", exist_ok=True)
            code_path = os.path.join("temp", code_file.name)
            
            with open(code_path, "wb") as f:
                f.write(code_file.getbuffer())
            
            st.session_state.code_file = {
                'name': code_file.name,
                'path': code_path,
                'type': 'code'
            }
            
            st.success(f"✅ Loaded: {code_file.name}")
            
            # Show file preview
            with st.expander("📝 File Preview"):
                with open(code_path, 'r') as f:
                    code_content = f.read()
                st.code(code_content[:500] + "..." if len(code_content) > 500 else code_content, 
                       language='python', 
                       line_numbers=True)
            
            # Quick action buttons
            st.markdown("**Quick Actions:**")
            col1, col2 = st.columns(2)

            with col1:
                if st.button("🔍 Analyze Code", use_container_width=True, key="analyze_btn"):
                    # Add message and trigger processing
                    query = f"Analyze {code_file.name} for code quality issues"
                    st.session_state.messages.append({
                        "role": "user", 
                        "content": query
                    })
                    st.session_state.trigger_query = query
                    st.rerun()

            with col2:
                if st.button("🛡️ Security Scan", use_container_width=True, key="security_btn"):
                    # Add message and trigger processing
                    query = f"Check {code_file.name} for security vulnerabilities"
                    st.session_state.messages.append({
                        "role": "user", 
                        "content": query
                    })
                    st.session_state.trigger_query = query
                    st.rerun()

    
    st.markdown("---")
    
    # Stats
    st.subheader("📊 System Info")
    if os.path.exists("vector_store/index.faiss"):
        try:
            with open("vector_store/chunks.pkl", 'rb') as f:
                chunks = pickle.load(f)
            st.metric("Document Chunks", len(chunks))
        except:
            st.metric("Document Chunks", "N/A")
    
    if st.session_state.code_file:
        st.metric("Code File", st.session_state.code_file['name'])
    
    # Clear chat
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.subheader("💡 Example Queries")
    
    if tab == "📄 Documents":
        examples = [
            "What are the main features?",
            "Summarize the documentation",
            "Explain the API endpoints"
        ]
    else:
        examples = [
            "Analyze the code for issues",
            "Check for security vulnerabilities",
            "Find potential bugs"
        ]
    
    for query in examples:
        if st.button(query, key=f"ex_{query}"):
            st.session_state.messages.append({"role": "user", "content": query})
            st.rerun()

# Main content
st.title("💬 IntelliCode RAG Assistant")
st.markdown("Advanced AI assistant for code analysis and document intelligence")

# Load system
if st.session_state.agent is None:
    try:
        agent, rag = load_system()
        st.session_state.agent = agent
        st.session_state.rag = rag
        st.success("✅ AI System Ready!")
    except Exception as e:
        st.error(f"❌ Error loading system: {e}")
        st.info("Make sure the model file exists in models/ folder")
        st.stop()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message["role"] == "assistant":
            if "tool" in message:
                st.caption(f"🔧 Tool: **{message['tool']}**")
            if "response_time" in message:
                st.caption(f"⏱️ Time: **{message['response_time']:.2f}s**")
            
            # Show detailed results for code analysis
            if "raw_output" in message:
                raw = message["raw_output"]
                
                if message.get("tool") == "code_analyzer":
                    with st.expander("📊 Detailed Analysis"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Issues Found", len(raw.get("issues", [])))
                        with col2:
                            st.metric("Severity", raw.get("severity", "N/A"))
                        with col3:
                            st.metric("Lines", raw.get("lines", 0))
                        
                        # Show issues
                        if raw.get("issues"):
                            st.markdown("**Issues:**")
                            for issue in raw["issues"]:
                                severity_color = {
                                    "HIGH": "🔴",
                                    "MEDIUM": "🟡",
                                    "LOW": "🟢"
                                }.get(issue["severity"], "⚪")
                                
                                st.markdown(f"{severity_color} **Line {issue['line']}**: {issue['type']}")
                                st.caption(f"_{issue['message']}_")
                
                elif message.get("tool") == "security_scanner":
                    with st.expander("🛡️ Security Report"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Risk Level", raw.get("risk_level", "N/A"))
                        with col2:
                            st.metric("Vulnerabilities", raw.get("total_issues", 0))
                        
                        # Show vulnerabilities
                        if raw.get("vulnerabilities"):
                            for vuln in raw["vulnerabilities"]:
                                severity_emoji = {
                                    "CRITICAL": "🚨",
                                    "HIGH": "🔴",
                                    "MEDIUM": "🟡",
                                    "LOW": "🟢"
                                }.get(vuln["severity"], "⚪")
                                
                                st.markdown(f"{severity_emoji} **{vuln['type']}** ({vuln['severity']})")
                                st.caption(vuln["description"])
                                if "fix" in vuln:
                                    st.info(f"💡 Fix: {vuln['fix']}")

# Process triggered query from quick actions
if 'trigger_query' in st.session_state and st.session_state.trigger_query:
    prompt = st.session_state.trigger_query
    st.session_state.trigger_query = None  # Clear trigger
    
    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("🤔 Analyzing..."):
            start_time = time.time()
            
            # Get file context
            file_context = st.session_state.code_file
            
            # Execute agent
            result = st.session_state.agent.execute(prompt, file_context=file_context)
            
            response_time = time.time() - start_time
        
        # Stream answer
        full_answer = result['answer']
        words = full_answer.split()
        displayed_text = ""
        
        for word in words:
            displayed_text += word + " "
            message_placeholder.markdown(displayed_text + "▌")
            time.sleep(0.04)
        
        message_placeholder.markdown(full_answer)
        
        # Display metadata
        col1, col2 = st.columns(2)
        with col1:
            st.caption(f"🔧 Tool: **{result['tool_used']}**")
        with col2:
            st.caption(f"⏱️ Time: {response_time:.2f}s")
        
        # Show detailed results
        raw = result['raw_output']
        
        if result['tool_used'] == "code_analyzer":
            with st.expander("📊 Detailed Analysis"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Issues", len(raw.get("issues", [])))
                with col2:
                    st.metric("Severity", raw.get("severity", "N/A"))
                with col3:
                    st.metric("Lines", raw.get("lines", 0))
                
                if raw.get("issues"):
                    st.markdown("**Issues:**")
                    for issue in raw["issues"]:
                        severity_color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(issue["severity"], "⚪")
                        st.markdown(f"{severity_color} **Line {issue['line']}**: {issue['type']}")
                        st.caption(f"_{issue['message']}_")
        
        elif result['tool_used'] == "security_scanner":
            with st.expander("🛡️ Security Report"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Risk Level", raw.get("risk_level", "N/A"))
                with col2:
                    st.metric("Vulnerabilities", raw.get("total_issues", 0))
                
                if raw.get("vulnerabilities"):
                    for vuln in raw["vulnerabilities"]:
                        severity_emoji = {"CRITICAL": "🚨", "HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(vuln["severity"], "⚪")
                        st.markdown(f"{severity_emoji} **{vuln['type']}** ({vuln['severity']})")
                        st.caption(vuln["description"])
                        if "fix" in vuln:
                            st.info(f"💡 Fix: {vuln['fix']}")
    
    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": result['answer'],
        "tool": result['tool_used'],
        "response_time": response_time,
        "raw_output": result['raw_output']
    })


# Chat input
if prompt := st.chat_input("Ask about your documents or code..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("🤔 Analyzing..."):
            start_time = time.time()
            
            # Get file context if code file uploaded
            file_context = st.session_state.code_file
            
            # Execute agent
            result = st.session_state.agent.execute(prompt, file_context=file_context)
            
            response_time = time.time() - start_time
        
        # Stream the answer word-by-word
        full_answer = result['answer']
        words = full_answer.split()
        displayed_text = ""
        
        for word in words:
            displayed_text += word + " "
            message_placeholder.markdown(displayed_text + "▌")
            time.sleep(0.04)
        
        # Show final answer
        message_placeholder.markdown(full_answer)
        
        # Display metadata
        col1, col2 = st.columns(2)
        with col1:
            st.caption(f"🔧 Tool: **{result['tool_used']}**")
        with col2:
            st.caption(f"⏱️ Time: {response_time:.2f}s")
        
        # Show detailed output based on tool
        raw = result['raw_output']
        
        if result['tool_used'] == "code_analyzer":
            with st.expander("📊 Detailed Analysis"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Issues", len(raw.get("issues", [])))
                with col2:
                    st.metric("Severity", raw.get("severity", "N/A"))
                with col3:
                    st.metric("Lines", raw.get("lines", 0))
                
                if raw.get("issues"):
                    st.markdown("**Issues:**")
                    for issue in raw["issues"]:
                        severity_color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(issue["severity"], "⚪")
                        st.markdown(f"{severity_color} **Line {issue['line']}**: {issue['type']}")
                        st.caption(f"_{issue['message']}_")
        
        elif result['tool_used'] == "security_scanner":
            with st.expander("🛡️ Security Report"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Risk Level", raw.get("risk_level", "N/A"))
                with col2:
                    st.metric("Vulnerabilities", raw.get("total_issues", 0))
                
                if raw.get("vulnerabilities"):
                    for vuln in raw["vulnerabilities"]:
                        severity_emoji = {"CRITICAL": "🚨", "HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(vuln["severity"], "⚪")
                        st.markdown(f"{severity_emoji} **{vuln['type']}** ({vuln['severity']})")
                        st.caption(vuln["description"])
                        if "fix" in vuln:
                            st.info(f"💡 Fix: {vuln['fix']}")
    
    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": result['answer'],
        "tool": result['tool_used'],
        "response_time": response_time,
        "raw_output": result['raw_output']
    })

# Footer
st.markdown("---")
st.caption("🚀 Powered by: Qwen2.5-3B | FAISS | AST Parser | Local CPU | IntelliCode RAG v2.0")
