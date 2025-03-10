# Xadrez das IAs

Uma aplicação que permite partidas de xadrez entre diferentes modelos de inteligência artificial ou entre humano e IA, usando vários provedores de LLM como OpenAI, Anthropic, Google e Hugging Face.

## Visão Geral

O projeto "Xadrez das IAs" é uma plataforma que permite:

- Partidas de xadrez entre diferentes modelos de IA (GPT-4, Claude, Gemini, Llama, etc.)
- Partidas entre humanos e IAs
- Ajuste do nível de habilidade das IAs
- Visualização de estatísticas e log de partidas
- Configuração avançada dos modelos

## Requisitos

- Python 3.8 ou superior
- Tkinter (geralmente vem com Python)
- Bibliotecas adicionais (ver `requirements.txt`)
- Acesso a chaves de API para os serviços de IA

## Instalação

1. Clone o repositório:
   ```
   git clone https://github.com/SEU_USUARIO/xadrez-ia.git
   cd xadrez-ia
   ```

2. Crie e ative um ambiente virtual:
   ```
   # No Windows
   python -m venv venv
   venv\Scripts\activate
   
   # No Linux/Mac
   python -m venv venv
   source venv/bin/activate
   ```

3. Instale as dependências:
   ```
   pip install -r requirements.txt
   ```

4. Configure suas chaves de API:
   - Crie um arquivo `.env` na raiz do projeto
   - Adicione suas chaves de API:
     ```
     OPENAI_API_KEY=sua_chave_aqui
     CLAUDE_API_KEY=sua_chave_aqui
     GOOGLE_API_KEY=sua_chave_aqui
     HF_TOKEN=sua_chave_aqui
     ```

## Como Usar

1. Inicie a aplicação:
   ```
   python api/main2.py
   ```

2. Interface e Controles:

   - **Modo de Jogo**:
     - IA vs IA: Duas IAs jogam entre si
     - Jogador vs IA: Você joga contra uma IA

   - **Configurações da IA**:
     - Selecione o provedor de IA (OpenAI, Claude, Gemini, HuggingFace)
     - Escolha modelos específicos para brancas e pretas
     - Ajuste o nível de habilidade (1-5)
     - Configure o tempo entre jogadas e limite de jogadas

   - **Opções Avançadas**:
     - Use jogadas aleatórias quando a IA falhar (fallback)

   - **Controles**:
     - Iniciar Partida: Começa um novo jogo
     - Pausar/Continuar: Pausa ou continua o jogo em andamento
     - Parar: Interrompe o jogo atual

3. Jogando como Humano:
   - Selecione "Jogador vs IA" no modo de jogo
   - Escolha sua cor (brancas ou pretas)
   - Clique nas peças para movê-las quando for sua vez

## Recursos

### Modelos Suportados

- **OpenAI**: gpt-4o, gpt-4-turbo, gpt-3.5-turbo
- **Anthropic**: claude-3-opus, claude-3-sonnet, claude-3-haiku
- **Google Gemini**: gemini-pro, gemini-1.5-pro
- **Hugging Face**: modelos como Llama, Mistral, DeepSeek e outros

### Personalização dos Modelos

É possível ajustar:
- Nível de habilidade das IAs (1-5)
- Tempo de resposta
- Uso de jogadas aleatórias como fallback

### Funcionalidades

- Visualização de tabuleiro interativa
- Histórico de jogadas
- Estatísticas da partida
- Salvamento de logs

## Estrutura do Projeto

- `api/main2.py` - Arquivo principal da aplicação
- `api/models_config.py` - Configuração dos modelos de IA
- `.env` - Arquivo de configuração das chaves de API (você deve criar)
- `requirements.txt` - Dependências do projeto

## Solução de Problemas

**Erro na inicialização do tabuleiro**:
- Certifique-se de que cairosvg está instalado corretamente
- No Windows, pode ser necessário instalar o GTK+

**Problemas com APIs**:
- Verifique se as chaves de API estão configuradas corretamente no arquivo `.env`
- Certifique-se de ter créditos/acesso suficiente nas plataformas

**Jogadas inválidas**:
- Aumente o nível de habilidade das IAs
- Verifique se o sistema de fallback está ativado

## Contribuindo

Contribuições são bem-vindas! Por favor, sinta-se à vontade para enviar pull requests ou abrir issues.

## Licença

Este projeto está licenciado sob os termos da Licença MIT.

## Como contribuir com o projeto

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/nova-funcionalidade`)
3. Faça commit das suas alterações (`git commit -m 'Adiciona nova funcionalidade'`)
4. Faça push para a branch (`git push origin feature/nova-funcionalidade`)
5. Abra um Pull Request
