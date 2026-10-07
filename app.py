import os
import sys
import uuid
import time
import streamlit as st
from dotenv import load_dotenv
from typing import cast, Any, TypedDict, Annotated, Sequence

# Langchain e Langgraph
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_community.document_loaders import PyPDFLoader

# 1. Configuração Inicial e Segurança
# Aqui carregamos o arquivo .env (que será ignorado pelo Git) apenas para testes locais.
# Quando hospedado no Streamlit Community Cloud, o .env não existirá lá.
# O Streamlit Cloud puxará a chave diretamente do painel de Secrets!
getattr(sys.stdout, 'reconfigure')(encoding='utf-8')
load_dotenv()

st.set_page_config(page_title="Leitor Inteligente de PDFs", page_icon="📄", layout="wide")
st.title("Assistente Virtual de Leitura de PDFs 📄🤖")
st.markdown("Faça o upload de qualquer documento em PDF. Nossa Inteligência Artificial lerá o documento e responderá suas perguntas com base exclusiva no texto extraído!")

# 2. Definição de Estado do Grafo
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    context: str

# 3. Inicialização da IA (Usando Gemini Gratuito)
@st.cache_resource
def inicializar_chatbot():
    # Usaremos o modelo gemma-4-31b-it grátis para que seu portfólio não gere custos.
    llm = ChatOpenAI(
        model="gemma-4-31b-it", 
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        temperature=0.3 # Temperatura baixa para ser mais focado na leitura do documento e não "inventar" coisas
    )

    system_prompt = (
        "Você é um assistente de IA profissional focado em análise de documentos e extração de informações. "
        "Você deve ajudar o usuário a entender o documento PDF que ele enviou. "
        "Responda SEMPRE com base no contexto fornecido abaixo. "
        "Se o usuário fizer uma pergunta cuja resposta não esteja no documento, seja honesto e diga "
        "que o documento não possui essa informação.\n\n"
        "CONTEXTO EXTRAÍDO DO DOCUMENTO:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages")
    ])

    # 4. Construção do Nó Principal
    def gerar_resposta(state: State):
        chain = prompt | llm
        
        # Estrutura de repetição para garantir que o Gemini grátis responda, 
        # mesmo se o servidor der erro 503 (congestionamento).
        max_tentativas = 3
        for tentativa in range(max_tentativas):
            try:
                response = chain.invoke({
                    "context": state.get("context", "Nenhum documento enviado."), 
                    "messages": state["messages"]
                })
                # Limpeza da tag <thought> que costuma vir no modelo gratuito do Google
                texto_limpo = str(response.content).split("</thought>")[-1].strip()
                return {"messages": [AIMessage(content=texto_limpo)]}
            except Exception as e:
                if tentativa < max_tentativas - 1:
                    time.sleep(3)
                else:
                    return {"messages": [AIMessage(content="Desculpe, os servidores gratuitos estão muito congestionados no momento. Tente novamente! ⏳")]}

    # 5. Compilação do Grafo
    workflow = StateGraph(cast(Any, State))
    workflow.add_node("chatbot", gerar_resposta)
    workflow.add_edge(START, "chatbot")
    workflow.add_edge("chatbot", END)
    
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app

app = inicializar_chatbot()

# Controle de sessão e identidade do usuário
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": st.session_state.thread_id}}

# ==========================================
# INTERFACE LATERAL (UPLOAD)
# ==========================================
with st.sidebar:
    st.header("📄 Envie seu Arquivo")
    st.write("Faça o upload de artigos, contratos ou manuais.")
    
    arquivo_pdf = st.file_uploader("Upload do PDF", type="pdf")
    
    if arquivo_pdf is not None:
        if st.button("Ler Documento"):
            with st.spinner("Extraindo texto... Isso pode levar alguns segundos dependendo do tamanho do PDF."):
                # Salvar temporariamente para o Langchain conseguir ler
                with open("temp_upload.pdf", "wb") as f:
                    f.write(arquivo_pdf.getvalue())
                
                try:
                    # Leitura e extração do texto
                    loader = PyPDFLoader("temp_upload.pdf")
                    paginas = loader.load()
                    st.session_state.contexto_pdf = "\n".join([p.page_content for p in paginas])
                    st.success("PDF processado com sucesso! Pode fazer sua pergunta no chat.")
                except Exception as e:
                    st.error(f"Erro ao ler o PDF: {e}")
                finally:
                    # Apaga o PDF do servidor por questões de segurança (Ninguém terá acesso aos arquivos)
                    if os.path.exists("temp_upload.pdf"):
                        os.remove("temp_upload.pdf")

# ==========================================
# CHAT PRINCIPAL
# ==========================================
contexto_atual = st.session_state.get("contexto_pdf", "Nenhum documento enviado.")
estado_atual = app.get_state(config)

if not estado_atual.values.get("messages"):
    with st.chat_message("assistant"):
        st.write("Olá! Sou seu Assistente de Leitura. Envie um PDF na barra lateral para começarmos a análise.")
else:
    for msg in estado_atual.values["messages"]:
        if msg.type == "human":
            with st.chat_message("user"):
                st.write(msg.content)
        elif msg.type == "ai":
            with st.chat_message("assistant"):
                st.write(msg.content)

pergunta = st.chat_input("Pergunte algo sobre o documento...")

if pergunta:
    with st.chat_message("user"):
        st.write(pergunta)

    with st.chat_message("assistant"):
        with st.spinner("Analisando o documento..."):
            resposta_final = app.invoke(
                {"messages": [HumanMessage(content=pergunta)], "context": contexto_atual},
                config=config
            )
            
            texto_resposta = resposta_final["messages"][-1].content
            st.write(texto_resposta)
