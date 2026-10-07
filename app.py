import os
import sys
import uuid
import time
import streamlit as st
from dotenv import load_dotenv
from typing import cast, Any, TypedDict, Annotated, Sequence

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_community.document_loaders import PyPDFLoader

# Configuração de encoding e variáveis de ambiente
getattr(sys.stdout, 'reconfigure')(encoding='utf-8')
load_dotenv()

# Configuração da página da aplicação
st.set_page_config(page_title="Leitor Inteligente de PDFs", page_icon="📄", layout="wide")

# Remove a barra superior do Streamlit (Fork, GitHub, etc) para uma aparência 100% profissional e limpa
st.markdown("""
    <style>
        [data-testid="stToolbar"] {visibility: hidden !important;}
        footer {visibility: hidden !important;}
    </style>
""", unsafe_allow_html=True)
st.title("Assistente Virtual de Leitura de PDFs 📄🤖")
st.markdown("Faça o upload de documentos em formato PDF. A inteligência artificial extrairá o conteúdo e responderá às suas perguntas baseando-se exclusivamente no texto processado.")

# Estrutura de Estado para o LangGraph
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    context: str

# Instanciação e configuração do modelo de linguagem (LLM)
@st.cache_resource
def inicializar_chatbot():
    # Inicializa o LLM compatível com a API da OpenAI (utilizando endpoint customizado)
    llm = ChatOpenAI(
        model="gemma-4-31b-it", 
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        temperature=0.3
    )

    system_prompt = (
        "Você é um assistente de inteligência artificial especializado em análise de documentos e extração de dados. "
        "Seu objetivo é auxiliar o usuário a compreender o conteúdo do documento PDF fornecido. "
        "Responda SEMPRE com base estrita no contexto extraído do documento abaixo. "
        "Caso o usuário faça uma pergunta cuja resposta não esteja no escopo do documento, informe polidamente "
        "que o documento não contém essa informação.\n\n"
        "CONTEXTO EXTRAÍDO DO DOCUMENTO:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages")
    ])

    # Nó principal de processamento
    def gerar_resposta(state: State):
        chain = prompt | llm
        
        # Estrutura de retentativas (retry) para garantir alta disponibilidade da API
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = chain.invoke({
                    "context": state.get("context", "Nenhum documento processado."), 
                    "messages": state["messages"]
                })
                
                # Pós-processamento para remover possíveis artefatos de raciocínio (Chain of Thought)
                clean_text = str(response.content).split("</thought>")[-1].strip()
                return {"messages": [AIMessage(content=clean_text)]}
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(3)
                else:
                    return {"messages": [AIMessage(content="O serviço está temporariamente indisponível devido ao alto volume de tráfego. Por favor, aguarde e tente novamente em alguns instantes. ⏳")]}

    # Definição do fluxo (Workflow)
    workflow = StateGraph(cast(Any, State))
    workflow.add_node("chatbot", gerar_resposta)
    workflow.add_edge(START, "chatbot")
    workflow.add_edge("chatbot", END)
    
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

app = inicializar_chatbot()

# Gerenciamento de sessão (Thread) para manter histórico independente por usuário
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": st.session_state.thread_id}}

# ==========================================
# PAINEL LATERAL (UPLOAD DE DOCUMENTOS)
# ==========================================
with st.sidebar:
    st.header("📄 Gerenciador de Arquivos")
    st.write("Faça o upload de artigos, relatórios, contratos ou manuais.")
    
    arquivo_pdf = st.file_uploader("Upload de arquivo (PDF)", type="pdf")
    
    if arquivo_pdf is not None:
        if st.button("Processar Documento"):
            with st.spinner("Extraindo e indexando texto... Isso pode levar alguns segundos."):
                
                # Armazenamento temporário para processamento pelo PyPDFLoader
                temp_path = "temp_upload.pdf"
                with open(temp_path, "wb") as f:
                    f.write(arquivo_pdf.getvalue())
                
                try:
                    loader = PyPDFLoader(temp_path)
                    paginas = loader.load()
                    st.session_state.contexto_pdf = "\n".join([p.page_content for p in paginas])
                    st.success("Documento processado com sucesso! A IA está pronta para responder.")
                except Exception as e:
                    st.error(f"Ocorreu um erro durante a leitura do arquivo: {e}")
                finally:
                    # Exclusão do arquivo físico do servidor para garantir segurança e privacidade
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

# ==========================================
# INTERFACE PRINCIPAL DE CHAT
# ==========================================
contexto_atual = st.session_state.get("contexto_pdf", "Nenhum documento processado.")
estado_atual = app.get_state(config)

if not estado_atual.values.get("messages"):
    with st.chat_message("assistant"):
        st.write("Bem-vindo(a)! Sou seu Assistente de Análise. Faça o upload de um PDF no painel lateral e faça perguntas sobre o conteúdo.")
else:
    for msg in estado_atual.values["messages"]:
        if msg.type == "human":
            with st.chat_message("user"):
                st.write(msg.content)
        elif msg.type == "ai":
            with st.chat_message("assistant"):
                st.write(msg.content)

pergunta = st.chat_input("Digite sua pergunta a respeito do documento...")

if pergunta:
    with st.chat_message("user"):
        st.write(pergunta)

    with st.chat_message("assistant"):
        with st.spinner("Analisando o contexto do documento..."):
            resposta_final = app.invoke(
                {"messages": [HumanMessage(content=pergunta)], "context": contexto_atual},
                config=config
            )
            
            texto_resposta = resposta_final["messages"][-1].content
            st.write(texto_resposta)
