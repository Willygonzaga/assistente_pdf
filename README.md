# Assistente Virtual de Leitura de PDFs 📄🤖

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://assistente-pdf.streamlit.app)

Um aplicativo web interativo alimentado por Inteligência Artificial capaz de ler documentos em formato PDF e responder a perguntas com base estrita no contexto extraído. Desenvolvido com **Streamlit**, **LangChain** e **Google Gemini**, este projeto demonstra a implementação prática de RAG (Retrieval-Augmented Generation) e fluxos baseados em grafos.

**👉 Teste a aplicação ao vivo:** [assistente-pdf.streamlit.app](https://assistente-pdf.streamlit.app)

## 🚀 Funcionalidades

- **Upload Dinâmico de Arquivos:** Permite o carregamento de qualquer documento em PDF (manuais, relatórios, artigos).
- **Leitura Inteligente (RAG):** O conteúdo do arquivo é extraído e utilizado como contexto absoluto pela IA, evitando respostas baseadas em dados externos (alucinações).
- **Gerenciamento de Estado (Memória):** O histórico do chat é mantido e gerenciado durante a sessão através de grafos de estado.
- **Segurança de Dados:** O arquivo PDF do usuário é deletado fisicamente do servidor imediatamente após o processamento.

## 🛠️ Tecnologias Utilizadas

- **Python 3**
- **Streamlit** (Interface gráfica e hospedagem web)
- **LangChain & LangGraph** (Orquestração do modelo de IA e memória)
- **Google Gemini API** (Modelo de Inteligência Artificial: `gemma-4-31b-it`)
- **PyPDFLoader** (Processamento e extração de texto)

## ⚙️ Como Executar Localmente

Siga os passos abaixo para testar a aplicação em sua própria máquina:

**1. Clone o repositório:**
```bash
git clone https://github.com/SEU_USUARIO/NOME_DO_REPOSITORIO.git
cd NOME_DO_REPOSITORIO
```

**2. Instale as dependências:**
Certifique-se de que o Python está instalado e execute:
```bash
pip install -r requirements.txt
```

**3. Configure as Variáveis de Ambiente:**
Crie um arquivo chamado `.env` na raiz do projeto e adicione sua chave de API do Google Gemini:
```env
GEMINI_API_KEY="SUA_CHAVE_AQUI"
```

**4. Execute a aplicação:**
```bash
streamlit run app.py
```
O aplicativo abrirá automaticamente no seu navegador.

## 🌐 Deploy na Nuvem
O aplicativo está preparado para ser hospedado gratuitamente no **Streamlit Community Cloud**. Em ambiente de produção, não suba o arquivo `.env`. Configure a chave da API diretamente na aba **Advanced Settings > Secrets** do Streamlit.

---
*Projeto desenvolvido por Willy Gonzaga com foco em arquitetura profissional e segurança de dados.*
