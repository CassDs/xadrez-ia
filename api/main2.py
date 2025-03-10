import tkinter as tk
from tkinter import messagebox, scrolledtext, filedialog, ttk
import chess
import chess.pgn
import chess.svg
from PIL import Image, ImageTk
import openai
import anthropic
import google.generativeai as genai
from huggingface_hub import InferenceClient
import os
import random
import traceback
from dotenv import load_dotenv
import cairosvg
import re
from datetime import datetime
import threading
import tempfile
import shutil
# Importar a configuração de modelos
from models_config import AIModelsConfig

# Carrega variáveis do arquivo .env
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
claude_client = anthropic.Anthropic(api_key=os.getenv('CLAUDE_API_KEY'))
hf_token = os.getenv('HF_TOKEN')  # Token da Hugging Face

# Configurar API do Google Gemini
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Carregar configuração de IAs
ai_config = AIModelsConfig()

class ChessAI:
    def __init__(self, model_name, board=None, skill_level=3, specific_model=None):
        self.model_name = model_name
        self.board = board if board else chess.Board()
        self.move_cache = {}  # Cache para evitar chamadas repetidas em posições idênticas
        self.skill_level = skill_level  # Nível de habilidade: 1 (iniciante) a 5 (avançado)
        self.specific_model = specific_model  # Modelo específico (gpt-4o, claude-3-opus, etc.)
        self.log_errors = True  # Registrar erros detalhados
        self.api_name = ai_config.get_api_name(model_name)  # Nome da API associada ao modelo
        self.use_random_fallback = True  # Valor padrão para uso de fallback

    def get_move(self, history, fen):
        # Verificar no cache primeiro
        cache_key = f"{self.model_name}:{self.specific_model}:{fen}:{self.skill_level}"
        if cache_key in self.move_cache:
            return self.move_cache[cache_key]
            
        # Se não estiver em cache, obter da API
        move = None
        raw_response = None
        if self.api_name == "openai":
            move, raw_response = self.chatgpt_move(history, fen)
        elif self.api_name == "anthropic":
            move, raw_response = self.claude_move(history, fen)
        elif self.api_name == "Gemini":
            move, raw_response = self.gemini_move(history, fen)
        elif self.api_name == "huggingface":
            move, raw_response = self.huggingface_move(history, fen)
            
        # Log detalhado da resposta da API
        print(f"\n===== RESPOSTA BRUTA DO {self.api_name} ({self.specific_model}) =====")
        print(f"Posição FEN: {fen}")
        print(f"Texto recebido: '{raw_response}'")
        print(f"Jogada extraída: {move}")
            
        # Se a IA não conseguiu gerar uma jogada válida, escolher uma jogada aleatória
        if move is None:
            if self.use_random_fallback:
                fallback_move = self.random_move(fen)
                if fallback_move:
                    print(f"⚠️ Usando jogada aleatória de fallback: {fallback_move}")
                move = fallback_move
            else:
                print("❌ Fallback desativado - nenhuma jogada válida retornada pela IA")
            
        # Armazenar no cache e retornar
        if move:
            self.move_cache[cache_key] = move
        return move
        
    def random_move(self, fen):
        """Escolhe uma jogada aleatória válida quando a IA falha"""
        try:
            self.board.set_fen(fen)
            legal_moves = list(self.board.legal_moves)
            if legal_moves:
                # Escolher jogada aleatória
                move = random.choice(legal_moves)
                return self.board.san(move)
            return None
        except Exception as e:
            if self.log_errors:
                print(f"Erro ao gerar jogada aleatória: {e}")
            return None

    def extract_san_move(self, text, fen):
        """Versão melhorada da extração de jogadas SAN que considera mais padrões e faz verificação robusta"""
        try:
            # Atualizar o tabuleiro com a posição atual
            self.board.set_fen(fen)
            
            # Se não houver texto, retornar None
            if not text or len(text) < 1:
                if self.log_errors:
                    print("Texto vazio ou muito curto recebido da IA")
                return None
            
            # Remover caracteres especiais e normalizar o texto
            text = text.replace('−', '-').strip()
            
            # Primeiro, verificar se o texto completo é uma jogada SAN válida
            try:
                move = self.board.parse_san(text)
                if move in self.board.legal_moves:
                    return self.board.san(move)  # Retorna a versão canônica
            except:
                pass
            
            # Lista de jogadas legais para verificação rápida
            legal_moves_san = [self.board.san(move) for move in self.board.legal_moves]
            
            # Se uma jogada legal aparece diretamente no texto, usá-la
            # Ordenar do mais longo para o mais curto para priorizar notações mais específicas
            legal_moves_san.sort(key=len, reverse=True)
            for legal_move in legal_moves_san:
                if legal_move in text:
                    return legal_move
            
            # Padrões comuns para jogadas SAN
            patterns = [
                r'\b([KQNBRP]?[a-h]?[1-8]?x?[a-h][1-8](=[QRNB])?[+#]?)\b',  # Padrão normal SAN
                r'\b(O-O(?:-O)?)\b',  # Roque
                r'\b(0-0(?:-0)?)\b',  # Roque com zeros (comum em respostas de IA)
                r'([a-h][1-8][-x][a-h][1-8])',  # Formato simplificado e3-e4 ou e3xe4
                r'move ([KQNBRP]?[a-h]?[1-8]?x?[a-h][1-8])',  # "move e4"
                r'([KQNBRP]?[a-h][1-8]-[a-h][1-8])'  # Formato alternativo e2-e4
            ]
            
            # Extrair todos os candidatos de todos os padrões
            candidates = []
            for pattern in patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    candidates.append(match.group(1))
            
            # Testar cada candidato em ordem - do mais longo para o mais curto
            # Isso favorece notações mais específicas primeiro
            candidates.sort(key=len, reverse=True)
            for candidate in candidates:
                # Corrigir notação de roque com zeros
                if candidate in ['0-0', '0-0-0']:
                    corrected = candidate.replace('0', 'O')
                    try:
                        move = self.board.parse_san(corrected)
                        if move in self.board.legal_moves:
                            return corrected
                    except:
                        continue
                
                # Lidar com formato simplificado e3-e4 -> e4
                if '-' in candidate and 'O-O' not in candidate:
                    try:
                        parts = candidate.split('-')
                        if len(parts) == 2:
                            dest = parts[1]  # Usar apenas o destino
                            # Verificar se corresponde a alguma jogada legal
                            for move in self.board.legal_moves:
                                if dest == chess.square_name(move.to_square):
                                    from_square = chess.square_name(move.from_square)
                                    if from_square == parts[0]:
                                        return self.board.san(move)
                    except:
                        pass
                
                # Tentar a jogada normal
                try:
                    move = self.board.parse_san(candidate)
                    if move in self.board.legal_moves:
                        return self.board.san(move)  # Retornar formato canônico
                except:
                    continue
            
            # Como último recurso, verificar cada palavra no texto
            for word in text.split():
                word = word.strip('.,;:"\'()[]{}')
                if word in legal_moves_san:
                    return word
                try:
                    move = self.board.parse_san(word)
                    if move in self.board.legal_moves:
                        return self.board.san(move)
                except:
                    continue
            
            if self.log_errors:
                print(f"Não foi possível extrair uma jogada válida de: '{text}'")
                print(f"Jogadas legais: {', '.join(legal_moves_san)}")
                
            return None
        except Exception as e:
            if self.log_errors:
                print(f"Erro na extração da jogada SAN: {e}")
                traceback.print_exc()
            return None

    def chatgpt_move(self, history, fen):
        self.board.set_fen(fen)
        color = "brancas" if self.board.turn else "pretas"
        legal_moves = [self.board.san(move) for move in self.board.legal_moves]
        
        if not legal_moves:
            return None, "Sem jogadas disponíveis"  # Sem jogadas disponíveis
        
        # Construir contexto estratégico baseado no nível de habilidade
        strategy_context = self._build_strategy_context()
        
        # Informações sobre material e posição
        material_info = self._get_material_info()
        position_info = self._analyze_position()
        
        # Melhorar o prompt com instruções mais claras e explícitas
        prompt = (
            f"Você é um mestre de xadrez jogando as {color} com nível de habilidade {self.skill_level}/5. "
            f"Posição atual (FEN): {fen}\n"
            f"Histórico das jogadas: {history}\n\n"
            f"Análise da posição:\n{position_info}\n"
            f"Material: {material_info}\n\n"
            f"{strategy_context}\n"
            f"JOGADAS LEGAIS DISPONÍVEIS: {', '.join(legal_moves)}\n\n"
            f"Analise a posição e responda com uma única jogada válida dentre as listadas acima. "
            f"Sua resposta deve conter APENAS a jogada escolhida em notação SAN (por exemplo: e4, Nf3, exd5). "
            f"NÃO inclua explicações, análises, texto adicional ou pontuação. "
            f"Responda apenas com a notação SAN exata da sua jogada."
        )
        
        try:
            system_prompt = (
                "Você é um jogador de xadrez experiente. "
                "Responda APENAS com uma jogada válida em notação SAN, exatamente como aparece na lista fornecida. "
                "Não adicione explicações, pontuação ou qualquer outro texto."
            )
            
            # Usar o modelo específico selecionado
            response = openai.chat.completions.create(
                model=self.specific_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,  # Reduzir tokens para forçar respostas curtas
                temperature=self._get_temperature(),
                timeout=30
            )
            move_text = response.choices[0].message.content.strip()
            
            # Verificar se temos uma jogada SAN válida
            move = self.extract_san_move(move_text, fen)
            if move:
                return move, move_text
                
            # Se a extração falhou, tentar uma abordagem ainda mais direta - verificar se a resposta está na lista
            if move_text in legal_moves:
                return move_text, move_text
                
            # Se ainda não encontrou, verificar substrings
            for legal_move in legal_moves:
                if legal_move in move_text:
                    return legal_move, move_text
                    
            if self.log_errors:
                print(f"ChatGPT ({self.specific_model}) não conseguiu gerar uma jogada válida. Resposta: '{move_text}'")
                
            return None, move_text
        except Exception as e:
            if self.log_errors:
                print(f"Erro na API do OpenAI: {e}")
                traceback.print_exc()
            return None, str(e)

    def claude_move(self, history, fen):
        self.board.set_fen(fen)
        color = "brancas" if self.board.turn else "pretas"
        legal_moves = [self.board.san(move) for move in self.board.legal_moves]
        
        if not legal_moves:
            return None, "Sem jogadas disponíveis"  # Sem jogadas disponíveis
        
        # Construir contexto estratégico baseado no nível de habilidade
        strategy_context = self._build_strategy_context()
        
        # Informações sobre material e posição
        material_info = self._get_material_info()
        position_info = self._analyze_position()
        
        # Melhorar o prompt com instruções mais claras e explícitas
        prompt = (
            f"Você é um mestre de xadrez jogando as {color} com nível de habilidade {self.skill_level}/5. "
            f"Posição atual (FEN): {fen}\n"
            f"Histórico das jogadas: {history}\n\n"
            f"Análise da posição:\n{position_info}\n"
            f"Material: {material_info}\n\n"
            f"{strategy_context}\n"
            f"JOGADAS LEGAIS DISPONÍVEIS: {', '.join(legal_moves)}\n\n"
            f"Analise a posição e responda com uma única jogada válida dentre as listadas acima. "
            f"Sua resposta deve conter APENAS a jogada escolhida em notação SAN (por exemplo: e4, Nf3, exd5). "
            f"NÃO inclua explicações, análises, texto adicional ou pontuação. "
            f"Responda apenas com a notação SAN exata da sua jogada."
        )
        
        try:
            # Usar o modelo específico selecionado com instruções claras
            completion = claude_client.messages.create(
                model=self.specific_model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,  # Reduzir tokens para forçar respostas curtas
                temperature=self._get_temperature(),
                timeout=30
            )
            move_text = completion.content[0].text.strip()
            
            # Verificar se temos uma jogada SAN válida
            move = self.extract_san_move(move_text, fen)
            if move:
                return move, move_text
                
            # Se a extração falhou, tentar uma abordagem ainda mais direta - verificar se a resposta está na lista
            if move_text in legal_moves:
                return move_text, move_text
                
            # Se ainda não encontrou, verificar substrings
            for legal_move in legal_moves:
                if legal_move in move_text:
                    return legal_move, move_text
                    
            if self.log_errors:
                print(f"Claude ({self.specific_model}) não conseguiu gerar uma jogada válida. Resposta: '{move_text}'")
                
            return None, move_text
        except Exception as e:
            if self.log_errors:
                print(f"Erro na API do Anthropic: {e}")
                traceback.print_exc()
            return None, str(e)
    
    def gemini_move(self, history, fen):
        """Obter jogada usando a API do Google Gemini com métodos otimizados"""
        self.board.set_fen(fen)
        color = "brancas" if self.board.turn else "pretas"
        legal_moves = [self.board.san(move) for move in self.board.legal_moves]
        
        if not legal_moves:
            return None, "Sem jogadas disponíveis"
        
        # Simplificar o prompt para melhorar a precisão da resposta
        prompt = (
            f"Você é um mestre de xadrez jogando as {color}.\n\n"
            f"Posição atual (FEN): {fen}\n"
            f"JOGADAS LEGAIS DISPONÍVEIS: {', '.join(legal_moves)}\n\n"
            f"Responda com uma ÚNICA jogada da lista acima em notação SAN. "
            f"Apenas a notação SAN sem explicações. Exemplos: e4, Nf3, exd5, O-O"
        )
        
        try:
            # Método 1: Direto com GenerativeModel sem prefixo "models/"
            model_name = self.specific_model
            if "models/" in model_name:
                model_name = model_name.replace("models/", "")
                
            print(f"Tentando API Gemini com modelo: {model_name} (Método 1)")
            
            model = genai.GenerativeModel(model_name)
            
            # Configurações mais restritas para forçar resposta curta e direta
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,  # Menor temperatura para maior precisão
                    max_output_tokens=5,  # Muito restritivo para forçar respostas curtas
                    top_p=0.95
                )
            )
            
            move_text = response.text.strip()
            print(f"Resposta do Gemini (Método 1): '{move_text}'")
            
            # Tentativa de extração direta - verificar se é exatamente uma jogada legal
            if move_text in legal_moves:
                print(f"✓ Jogada exata encontrada em legal_moves: {move_text}")
                return move_text, move_text
                
            move = self.extract_san_move(move_text, fen)
            if move:
                print(f"✓ Jogada extraída com sucesso: {move}")
                return move, move_text
                
            # Se falhar, tentar métodos alternativos
            print("⚠️ Primeiro método falhou, tentando alternativas...")
            
            # Método 2: Tentar com prompt mais simples ainda
            simpler_prompt = f"Escolha uma jogada de xadrez da seguinte lista: {', '.join(legal_moves)}. Responda apenas com a jogada."
            
            response2 = model.generate_content(
                simpler_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=5
                )
            )
            
            move_text2 = response2.text.strip()
            print(f"Resposta do Gemini (Método 2): '{move_text2}'")
            
            if move_text2 in legal_moves:
                print(f"✓ Método 2: Jogada exata encontrada: {move_text2}")
                return move_text2, f"{move_text} -> {move_text2}"
                
            move2 = self.extract_san_move(move_text2, fen)
            if move2:
                print(f"✓ Método 2: Jogada extraída: {move2}")
                return move2, f"{move_text} -> {move_text2}"
                
            # Se ainda falhar, registrar o problema detalhadamente
            print("❌ Todos os métodos de extração falharam")
            print(f"Legal moves: {', '.join(legal_moves)}")
            return None, f"Falha em obter jogada válida. Respostas: '{move_text}', '{move_text2}'"
            
        except Exception as e:
            error_message = f"Erro na API Gemini: {str(e)}"
            print(f"❌ {error_message}")
            traceback.print_exc()
            return None, error_message

    def huggingface_move(self, history, fen):
        """Obter jogada usando a API da Hugging Face"""
        self.board.set_fen(fen)
        color = "brancas" if self.board.turn else "pretas"
        legal_moves = [self.board.san(move) for move in self.board.legal_moves]
        
        if not legal_moves:
            return None, "Sem jogadas disponíveis"
        
        # Verificar se é um modelo de reflexão
        is_reflection_model = ai_config.is_reflection_model(self.specific_model)
        
        # Verificar se é um modelo DeepSeek
        is_deepseek_model = "deepseek" in self.specific_model.lower()
        
        # Configurar timeout maior para modelos de reflexão
        timeout = 180 if is_reflection_model else 60  # 3 minutos vs 1 minuto
        
        # Ajustar o prompt de acordo com o tipo de modelo
        if is_reflection_model:
            prompt = (
                f"Você é um mestre de xadrez jogando as {color}.\n\n"
                f"Posição atual (FEN): {fen}\n"
                f"JOGADAS LEGAIS DISPONÍVEIS: {', '.join(legal_moves)}\n\n"
                f"Por favor, use sua capacidade de reflexão para analisar essa posição de xadrez.\n"
                f"Pense passo a passo sobre as melhores jogadas possíveis, considerando táticas e estratégia.\n"
                f"Depois da sua análise, escolha uma única jogada da lista acima.\n\n"
                f"Após sua reflexão, responda APENAS com essa jogada sem nenhuma explicação adicional."
            )
            print("Usando prompt especial para modelo de reflexão DeepSeek")
            self.status_display_message = "Aguarde... Modelo DeepSeek está refletindo sobre a jogada (pode levar mais tempo)"
        else:
            prompt = (
                f"Você é um mestre de xadrez jogando as {color}.\n\n"
                f"Posição atual (FEN): {fen}\n"
                f"JOGADAS LEGAIS DISPONÍVEIS: {', '.join(legal_moves)}\n\n"
                f"Responda com uma ÚNICA jogada da lista acima. "
                f"Apenas a notação SAN sem explicações. Exemplos: e4, Nf3, exd5, O-O"
            )
            self.status_display_message = None
        
        try:
            # Inicializar cliente Hugging Face
            client = InferenceClient(api_key=hf_token)
            
            log_prefix = "[REFLEXÃO]" if is_reflection_model else ""
            print(f"{log_prefix} Chamando API Hugging Face com modelo: {self.specific_model}")
            
            # Atualizar status na interface (se aplicável)
            if hasattr(self, '_notify_status_update') and callable(self._notify_status_update) and self.status_display_message:
                self._notify_status_update(self.status_display_message)
            
            # Preparar mensagens para o formato de chat
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            # Ajustar temperatura e configurações para cada tipo de modelo
            # DeepSeek funciona melhor com temperatura menor para xadrez
            temperature = 0.3 if is_deepseek_model else (0.1 if is_reflection_model else self._get_temperature())
            
            # DeepSeek precisa de 2048 tokens para reflexão
            max_tokens = 2048 if is_deepseek_model else (256 if is_reflection_model else 10)
            
            # Ajustar top_p para DeepSeek
            top_p = 0.7 if is_deepseek_model else 0.95
            
            print(f"{log_prefix} Configurando max_tokens={max_tokens}, temperature={temperature}, top_p={top_p} para o modelo")
            
            # Fazer chamada à API
            print(f"{log_prefix} Iniciando requisição (timeout: {timeout}s)")
            start_time = datetime.now()
            
            move_text = ""
            
            # Para modelos DeepSeek, usar stream=True e coletar a resposta completa
            if is_deepseek_model:
                print(f"{log_prefix} Usando modo streaming para modelo DeepSeek")
                
                try:
                    # Usar streaming conforme o exemplo fornecido
                    stream = client.chat.completions.create(
                        model=self.specific_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        stream=True
                    )
                    
                    # Coletar a resposta completa do streaming
                    full_response = ""
                    for chunk in stream:
                        if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                            delta = chunk.choices[0].delta
                            if hasattr(delta, 'content') and delta.content:
                                full_response += delta.content
                                # Verificar se temos o método de atualização de status
                                if len(full_response) % 100 == 0 and hasattr(self, '_notify_status_update') and callable(self._notify_status_update):  
                                    self._notify_status_update(f"DeepSeek refletindo... ({len(full_response)} caracteres)")
                    
                    move_text = full_response.strip()
                    print(f"{log_prefix} Resposta de streaming completa: '{move_text[:100]}...' ({len(move_text)} caracteres)")
                    
                except Exception as stream_error:
                    print(f"{log_prefix} Erro no streaming: {stream_error}")
                    # Fallback para modo não-streaming se streaming falhar
                    print(f"{log_prefix} Tentando modo não-streaming como fallback")
                    response = client.chat.completions.create(
                        model=self.specific_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        stream=False
                    )
                    move_text = response.choices[0].message.content.strip()
            else:
                # Modo normal não-streaming para outros modelos
                response = client.chat.completions.create(
                    model=self.specific_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stream=False
                )
                move_text = response.choices[0].message.content.strip()
            
            elapsed_time = (datetime.now() - start_time).total_seconds()
            print(f"{log_prefix} Resposta recebida após {elapsed_time:.2f} segundos")
            
            print(f"{log_prefix} Resposta da API HuggingFace: '{move_text[:100]}...'")
            
            # Para modelos DeepSeek, verificar a tag </think>
            if is_reflection_model:
                think_tag = "</think>"
                if think_tag in move_text:
                    # Extrair apenas o texto após a tag </think>
                    post_reflection = move_text.split(think_tag, 1)[1].strip()
                    print(f"{log_prefix} Texto após tag de reflexão: '{post_reflection}'")
                    move_text = post_reflection
            
            # Para modelos de reflexão, a resposta pode conter o processo de pensamento seguido da jogada
            # Buscamos a resposta final que geralmente está no final do texto
            if is_reflection_model and len(move_text) > 10:  # Se é uma resposta longa
                # Procurar por padrões comuns de respostas finais
                patterns = [
                    r".*jogada final:?\s*([A-Za-z0-9\-]+).*$",  # "jogada final: e4"
                    r".*escolho:?\s*([A-Za-z0-9\-]+).*$",       # "escolho: e4"
                    r".*escolhida:?\s*([A-Za-z0-9\-]+).*$",     # "jogada escolhida: e4" 
                    r".*resposta:?\s*([A-Za-z0-9\-]+).*$",      # "resposta: e4"
                    r".*\n\s*([A-Za-z0-9\-]+)\s*$"              # última linha contendo apenas a jogada
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, move_text, re.IGNORECASE)
                    if match:
                        potential_move = match.group(1).strip()
                        print(f"{log_prefix} Extraído potencial movimento do texto de reflexão: '{potential_move}'")
                        if potential_move in legal_moves:
                            print(f"{log_prefix} Movimento final extraído da reflexão: {potential_move}")
                            return potential_move, move_text
            
            # Verificar se temos uma jogada SAN válida
            move = self.extract_san_move(move_text, fen)
            if move:
                print(f"✓ Jogada válida extraída: {move}")
                return move, move_text
                
            # Se a extração falhou, verificar se a resposta está na lista
            if move_text in legal_moves:
                print(f"✓ Jogada exata encontrada em legal_moves: {move_text}")
                return move_text, move_text
                
            # Se ainda não encontrou, verificar substrings
            for legal_move in legal_moves:
                if legal_move in move_text:
                    print(f"✓ Substring encontrada como jogada válida: {legal_move}")
                    return legal_move, move_text
            
            # Tentativa alternativa com prompt ainda mais simplificado
            if self.log_errors:
                print(f"❌ HuggingFace ({self.specific_model}) falhou na primeira tentativa. Tentando com prompt simples.")
            
            simple_prompt = f"Escolha uma jogada de xadrez da lista: {', '.join(legal_moves)}"
            messages = [{"role": "user", "content": simple_prompt}]
            
            response = client.chat.completions.create(
                model=self.specific_model,
                messages=messages,
                temperature=0.1,
                max_tokens=5,
                top_p=0.95,
                stream=False
            )
            
            move_text2 = response.choices[0].message.content.strip()
            print(f"Resposta da segunda tentativa: '{move_text2}'")
            
            if move_text2 in legal_moves:
                return move_text2, f"{move_text} -> {move_text2}"
                
            move = self.extract_san_move(move_text2, fen)
            if move:
                return move, f"{move_text} -> {move_text2}"
            
            # Se ainda falhar
            print("❌ Todas as tentativas falharam")
            return None, f"Falha em obter jogada válida. Respostas: '{move_text}', '{move_text2}'"
            
        except Exception as e:
            error_message = f"Erro na API HuggingFace: {str(e)}"
            print(f"❌ {error_message}")
            traceback.print_exc()
            return None, error_message

    def _get_material_info(self):
        """Calcula vantagem material e controle do centro"""
        # Valores das peças
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0  # Rei não contribui para cálculo de material
        }
        
        # Calcular total de material para cada lado
        white_material = 0
        black_material = 0
        
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                value = piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    white_material += value
                else:
                    black_material += value
        
        # Verificar controle do centro
        center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        white_center_control = 0
        black_center_control = 0
        
        for square in center_squares:
            # Contar ataques para cada casa central
            white_attacks = len(list(self.board.attackers(chess.WHITE, square)))
            black_attacks = len(list(self.board.attackers(chess.BLACK, square)))
            
            white_center_control += white_attacks
            black_center_control += black_attacks
        
        # Mensagem formatada com informações de material
        material_diff = white_material - black_material
        if material_diff > 0:
            material_status = f"Brancas têm vantagem material de {material_diff} pontos"
        elif material_diff < 0:
            material_status = f"Pretas têm vantagem material de {abs(material_diff)} pontos"
        else:
            material_status = "Material equilibrado"
            
        center_status = "Brancas controlam mais o centro" if white_center_control > black_center_control else \
                       "Pretas controlam mais o centro" if black_center_control > white_center_control else \
                       "Controle do centro equilibrado"
                       
        return f"{material_status}. {center_status}."
    
    def _analyze_position(self):
        """Analisa a posição para fornecer contexto estratégico"""
        # Verificar estado do jogo
        if self.board.is_check():
            check_status = "Xeque! Rei " + ("branco" if self.board.turn == chess.WHITE else "preto") + " está sob ataque."
        else:
            check_status = "Não há xeque no momento."
            
        # Fase da partida
        piece_count = len(self.board.piece_map())
        if piece_count > 28:
            phase = "Abertura"
        elif piece_count > 15:
            phase = "Meio de jogo"
        else:
            phase = "Final de jogo"
            
        # Desenvolvimento das peças (simplificado)
        white_developed = len([square for square in chess.SQUARES if 
                             self.board.piece_at(square) and 
                             self.board.piece_at(square).color == chess.WHITE and
                             self.board.piece_at(square).piece_type != chess.PAWN and
                             self.board.piece_at(square).piece_type != chess.KING and
                             square not in [chess.A1, chess.B1, chess.C1, chess.D1, chess.E1, chess.F1, chess.G1, chess.H1]])
                             
        black_developed = len([square for square in chess.SQUARES if 
                             self.board.piece_at(square) and 
                             self.board.piece_at(square).color == chess.BLACK and
                             self.board.piece_at(square).piece_type != chess.PAWN and
                             self.board.piece_at(square).piece_type != chess.KING and
                             square not in [chess.A8, chess.B8, chess.C8, chess.D8, chess.E8, chess.F8, chess.G8, chess.H8]])
                             
        development = f"Brancas têm {white_developed} peças desenvolvidas. Pretas têm {black_developed} peças desenvolvidas."
        
        # Mobilidade (simplificada - número de movimentos legais)
        original_turn = self.board.turn
        
        self.board.turn = chess.WHITE
        white_mobility = len(list(self.board.legal_moves))
        
        self.board.turn = chess.BLACK
        black_mobility = len(list(self.board.legal_moves))
        
        self.board.turn = original_turn
        
        mobility = f"Mobilidade: Brancas têm {white_mobility} movimentos disponíveis. Pretas têm {black_mobility} movimentos disponíveis."
        
        return f"Fase: {phase}. {check_status} {development} {mobility}"
        
    def _build_strategy_context(self):
        """Constrói contexto estratégico baseado no nível de habilidade"""
        if self.skill_level <= 1:  # Iniciante
            return (
                "Estratégia: Como iniciante, foque em proteger suas peças e capture peças desprotegidas do oponente. "
                "Tente desenvolver suas peças para o centro e não se preocupe muito com estratégias complexas. "
                "Priorize capturas simples e evite trocas desvantajosas."
            )
        elif self.skill_level <= 2:  # Básico
            return (
                "Estratégia: No nível básico, concentre-se em desenvolvimento, controle do centro e segurança do rei. "
                "Considere capturas quando vantajosas, mas também pense um movimento à frente. "
                "Reconheça ameaças básicas e defenda-se apropriadamente."
            )
        elif self.skill_level <= 3:  # Intermediário
            return (
                "Estratégia: Como jogador intermediário, equilibre desenvolvimento com táticas. "
                "Considere planos de médio prazo e avalie trocas considerando posição e material. "
                "Identifique ameaças táticas como garfos, espetos e capturas."
            )
        elif self.skill_level <= 4:  # Avançado
            return (
                "Estratégia: Em nível avançado, considere temas posicionais como estrutura de peões, "
                "casas fracas e fortes, e controle de colunas abertas. "
                "Avalie sacrifícios táticos e busque iniciativa. Planeje vários movimentos à frente."
            )
        else:  # Mestre
            return (
                "Estratégia: Como mestre, analise profundamente a posição, considerando desequilíbrios, "
                "sacrifícios posicionais, e manobras estratégicas complexas. "
                "Pense várias jogadas à frente e avalie posições de forma abrangente, "
                "considerando material, tempo, espaço e estrutura."
            )
    
    def _get_temperature(self):
        """Determina a temperatura baseada no nível de habilidade"""
        # Quanto menor a temperatura, mais consistente e "preciso" é o modelo
        # Quanto maior a temperatura, mais criativo/aleatório
        if self.skill_level >= 5:  # Mestre
            return 0.2  # Mais preciso
        elif self.skill_level >= 4:  # Avançado
            return 0.3
        elif self.skill_level >= 3:  # Intermediário
            return 0.5
        elif self.skill_level >= 2:  # Básico
            return 0.7
        else:  # Iniciante
            return 0.9  # Mais variável/criativo/aleatório

class ChessApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Xadrez das IAs')
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Carregar configuração de IAs
        self.ai_config = AIModelsConfig()
        
        # Obter lista de IAs e modelos disponíveis
        self.available_ais = self.ai_config.get_available_ais()
        
        # Container principal
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Frame lateral para controles
        control_frame = tk.Frame(main_frame)
        control_frame.pack(side=tk.RIGHT, padx=20, pady=10, fill=tk.Y)

        # Frame do tabuleiro
        board_frame = tk.Frame(main_frame)
        board_frame.pack(side=tk.LEFT, padx=20, pady=10)

        # Seleção de modo de jogo
        self.mode_frame = ttk.LabelFrame(control_frame, text="Modo de Jogo")
        self.mode_frame.pack(pady=10, fill=tk.X)
        
        self.game_mode_var = tk.StringVar(value="IA vs IA")
        ttk.Radiobutton(self.mode_frame, text="IA vs IA", variable=self.game_mode_var, value="IA vs IA", command=self.update_mode).pack(anchor=tk.W, padx=5, pady=2)
        ttk.Radiobutton(self.mode_frame, text="Jogador vs IA", variable=self.game_mode_var, value="Jogador vs IA", command=self.update_mode).pack(anchor=tk.W, padx=5, pady=2)
        
        # Seleção de cor do jogador (inicialmente desativado)
        self.player_color_frame = ttk.LabelFrame(control_frame, text="Sua Cor")
        self.player_color_frame.pack(pady=5, fill=tk.X)
        
        self.player_color_var = tk.StringVar(value="Brancas")
        ttk.Radiobutton(self.player_color_frame, text="Brancas", variable=self.player_color_var, value="Brancas", state=tk.DISABLED).pack(anchor=tk.W, padx=5, pady=2)
        ttk.Radiobutton(self.player_color_frame, text="Pretas", variable=self.player_color_var, value="Pretas", state=tk.DISABLED).pack(anchor=tk.W, padx=5, pady=2)

        # Configurações
        self.config_frame = ttk.LabelFrame(control_frame, text="Configurações da IA")
        self.config_frame.pack(pady=10, fill=tk.X)
        
        # Seleção de modelos e níveis para brancas
        tk.Label(self.config_frame, text="Brancas:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.white_model_var = tk.StringVar(value=self.available_ais[0])  # Primeira IA da lista como padrão
        self.white_model_menu = ttk.OptionMenu(
            self.config_frame, 
            self.white_model_var, 
            self.white_model_var.get(), 
            *self.available_ais,
            command=self.update_white_model_options
        )
        self.white_model_menu.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Menu de modelos específicos para brancas
        tk.Label(self.config_frame, text="Modelo:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        first_ai_default = self.ai_config.get_default_model(self.white_model_var.get())
        self.white_specific_model_var = tk.StringVar(value=first_ai_default)
        self.white_specific_model_menu = ttk.OptionMenu(
            self.config_frame,
            self.white_specific_model_var,
            first_ai_default,
            *self.ai_config.get_models_for_ai(self.white_model_var.get())
        )
        self.white_specific_model_menu.grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
        
        tk.Label(self.config_frame, text="Nível:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.white_skill_var = tk.IntVar(value=3)
        ttk.Spinbox(self.config_frame, from_=1, to=5, width=5, textvariable=self.white_skill_var).grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Seleção de modelos e níveis para pretas
        tk.Label(self.config_frame, text="Pretas:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.black_model_var = tk.StringVar(value=self.available_ais[1] if len(self.available_ais) > 1 else self.available_ais[0])
        self.black_model_menu = ttk.OptionMenu(
            self.config_frame, 
            self.black_model_var, 
            self.black_model_var.get(), 
            *self.available_ais,
            command=self.update_black_model_options
        )
        self.black_model_menu.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Menu de modelos específicos para pretas
        tk.Label(self.config_frame, text="Modelo:").grid(row=2, column=2, padx=5, pady=5, sticky=tk.W)
        second_ai_default = self.ai_config.get_default_model(self.black_model_var.get())
        self.black_specific_model_var = tk.StringVar(value=second_ai_default)
        self.black_specific_model_menu = ttk.OptionMenu(
            self.config_frame,
            self.black_specific_model_var,
            second_ai_default,
            *self.ai_config.get_models_for_ai(self.black_model_var.get())
        )
        self.black_specific_model_menu.grid(row=2, column=3, padx=5, pady=5, sticky=tk.W)
        
        tk.Label(self.config_frame, text="Nível:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        self.black_skill_var = tk.IntVar(value=3)
        ttk.Spinbox(self.config_frame, from_=1, to=5, width=5, textvariable=self.black_skill_var).grid(row=3, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Tempo entre jogadas
        tk.Label(self.config_frame, text="Tempo (ms):").grid(row=4, column=0, padx=5, pady=5, sticky=tk.W)
        self.time_var = tk.IntVar(value=1500)
        ttk.Entry(self.config_frame, textvariable=self.time_var, width=8).grid(row=4, column=1, padx=5, pady=5, sticky=tk.W)

        # Limite de jogadas
        tk.Label(self.config_frame, text="Limite de jogadas:").grid(row=4, column=2, padx=5, pady=5, sticky=tk.W)
        self.move_limit_var = tk.IntVar(value=30)
        ttk.Entry(self.config_frame, textvariable=self.move_limit_var, width=5).grid(row=4, column=3, padx=5, pady=5, sticky=tk.W)

        # Opções avançadas
        self.config_advanced_frame = ttk.LabelFrame(control_frame, text="Opções Avançadas")
        self.config_advanced_frame.pack(pady=10, fill=tk.X)
        
        # Checkbox para ativar o fallback de jogadas aleatórias
        self.use_random_fallback_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            self.config_advanced_frame, 
            text="Usar jogadas aleatórias quando a IA falhar", 
            variable=self.use_random_fallback_var
        ).pack(anchor=tk.W, padx=5, pady=2)

        # Tabuleiro
        self.board = chess.Board()
        self.canvas = tk.Canvas(board_frame, width=500, height=500)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.canvas_click)
        
        # Status bar
        self.status_var = tk.StringVar(value="Pronto para iniciar")
        self.status_bar = tk.Label(root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Área de log
        self.log_frame = tk.Frame(root)
        self.log_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        tk.Label(self.log_frame, text="Log de jogadas:").pack(anchor=tk.W)
        self.log_area = scrolledtext.ScrolledText(self.log_frame, width=60, height=5)
        self.log_area.pack(fill=tk.BOTH, expand=True)
        
        # Estatísticas
        self.stats_frame = tk.Frame(root)
        self.stats_frame.pack(pady=5, fill=tk.X)
        
        self.move_count_var = tk.StringVar(value="Jogadas: 0")
        tk.Label(self.stats_frame, textvariable=self.move_count_var).pack(side=tk.LEFT, padx=10)
        
        self.white_moves_var = tk.StringVar(value="Brancas: 0 jogadas")
        tk.Label(self.stats_frame, textvariable=self.white_moves_var).pack(side=tk.LEFT, padx=10)
        
        self.black_moves_var = tk.StringVar(value="Pretas: 0 jogadas")
        tk.Label(self.stats_frame, textvariable=self.black_moves_var).pack(side=tk.LEFT, padx=10)
        
        self.api_calls_var = tk.StringVar(value="Chamadas API: 0")
        tk.Label(self.stats_frame, textvariable=self.api_calls_var).pack(side=tk.LEFT, padx=10)
        
        # Botões
        self.button_frame = tk.Frame(control_frame)
        self.button_frame.pack(pady=15, fill=tk.X)
        
        # Botão de iniciar destacado
        self.start_button = ttk.Button(
            self.button_frame,
            text="Iniciar Partida", 
            command=self.start_game,
            style="Accent.TButton",
            width=20
        )
        self.start_button.pack(pady=5, fill=tk.X)
        
        # Demais botões
        self.pause_button = ttk.Button(self.button_frame, text="Pausar", command=self.toggle_pause, width=20)
        self.pause_button.pack(pady=5, fill=tk.X)

        self.stop_button = ttk.Button(self.button_frame, text="Parar", command=self.stop_game, width=20)
        self.stop_button.pack(pady=5, fill=tk.X)
        
        # Estado do jogo
        self.paused = False
        self.game_running = False
        self.move_count = 0
        self.white_moves = 0
        self.black_moves = 0
        self.api_calls = 0
        self.game_after_id = None
        self.thread = None
        self.thinking = False
        self.ai_retry_count = 0
        self.selected_square = None
        self.legal_moves = []
        self.player_turn = False
        
        # Inicializar modelos
        self.model_white = ChessAI(
            self.white_model_var.get(), 
            self.board, 
            self.white_skill_var.get(),
            self.white_specific_model_var.get()
        )
        self.model_black = ChessAI(
            self.black_model_var.get(), 
            self.board, 
            self.black_skill_var.get(),
            self.black_specific_model_var.get()
        )
        
        # Configurar estilo ttk
        self.setup_styles()
        
        # Inicializar tabuleiro
        self.temp_dir = tempfile.mkdtemp(prefix="chess_app_")
        self.temp_files = []
        self.update_board()
        
    def update_white_model_options(self, *args):
        """Atualiza as opções de modelos específicos com base no modelo geral selecionado para brancas"""
        model_list = self.ai_config.get_models_for_ai(self.white_model_var.get())
        default_model = self.ai_config.get_default_model(self.white_model_var.get())
        
        # Limpar menu atual
        self.white_specific_model_menu['menu'].delete(0, 'end')
        
        # Configurar novo valor padrão
        self.white_specific_model_var.set(default_model)
        
        # Adicionar novas opções
        for model in model_list:
            self.white_specific_model_menu['menu'].add_command(
                label=model, 
                command=lambda value=model: self.white_specific_model_var.set(value)
            )
    
    def update_black_model_options(self, *args):
        """Atualiza as opções de modelos específicos com base no modelo geral selecionado para pretas"""
        model_list = self.ai_config.get_models_for_ai(self.black_model_var.get())
        default_model = self.ai_config.get_default_model(self.black_model_var.get())
        
        # Limpar menu atual
        self.black_specific_model_menu['menu'].delete(0, 'end')
        
        # Configurar novo valor padrão
        self.black_specific_model_var.set(default_model)
        
        # Adicionar novas opções
        for model in model_list:
            self.black_specific_model_menu['menu'].add_command(
                label=model, 
                command=lambda value=model: self.black_specific_model_var.set(value)
            )

    def setup_styles(self):
        # Configurar estilos personalizados
        style = ttk.Style()
        style.configure("TButton", padding=6)
        style.configure("Accent.TButton", font=("Arial", 10, "bold"))
        
    def update_mode(self):
        if self.game_mode_var.get() == "Jogador vs IA":
            # Ativar seleção de cor
            for child in self.player_color_frame.winfo_children():
                if isinstance(child, ttk.Radiobutton):
                    child.configure(state=tk.NORMAL)
        else:
            # Desativar seleção de cor
            for child in self.player_color_frame.winfo_children():
                if isinstance(child, ttk.Radiobutton):
                    child.configure(state=tk.DISABLED)
    
    def update_board(self, highlight_squares=None):
        if highlight_squares is None:
            highlight_squares = []
            
        # Criar um estilo com bordas coloridas para casas destacadas
        colors = {}
        # Adicionar destaque para casas selecionadas
        if self.selected_square is not None:
            colors[self.selected_square] = "#12aa12"  # Verde para a casa selecionada
            
        # Adicionar destaque para movimentos legais
        for move in self.legal_moves:
            colors[move.to_square] = "#aaaaff"  # Azul claro para movimentos possíveis
            
        # Criar SVG do tabuleiro com cores destacadas
        svg_board = chess.svg.board(self.board, size=500, squares=colors)
        temp_file = os.path.join(self.temp_dir, f'board_{datetime.now().strftime("%Y%m%d%H%M%S%f")}.png')
        cairosvg.svg2png(bytestring=svg_board, write_to=temp_file)
        self.temp_files.append(temp_file)
        
        img = Image.open(temp_file)
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.create_image(250, 250, image=self.tk_img)
        
        # Limpar arquivos temporários antigos quando houver muitos
        if len(self.temp_files) > 5:
            old_file = self.temp_files.pop(0)
            try:
                os.remove(old_file)
            except:
                pass
                
    def canvas_click(self, event):
        # Ignorar cliques se não for um jogo jogador vs IA ou não for a vez do jogador
        if not self.game_running or self.game_mode_var.get() != "Jogador vs IA" or not self.player_turn:
            return
            
        # Converter coordenadas do clique para casa do tabuleiro
        file_size = self.canvas.winfo_width() / 8
        rank_size = self.canvas.winfo_height() / 8
        
        file_idx = int(event.x / file_size)
        rank_idx = int(event.y / rank_size)
        
        # Inverter o rank_idx se as pretas estiverem na parte inferior
        if self.player_color_var.get() == "Brancas":
            rank_idx = 7 - rank_idx
        else:
            file_idx = 7 - file_idx
            
        square = chess.square(file_idx, rank_idx)
        
        # Se já temos uma casa selecionada
        if self.selected_square is not None:
            # Verificar se a nova casa é um destino válido
            for move in self.legal_moves:
                if move.to_square == square:
                    # Executar o movimento
                    self.make_player_move(move)
                    return
                    
            # Se clicar na mesma casa, desselecionar
            if square == self.selected_square:
                self.selected_square = None
                self.legal_moves = []
                self.update_board()
                return
                
            # Se clicar em outra peça própria, atualizar a seleção
            piece = self.board.piece_at(square)
            if piece and piece.color == self.board.turn:
                self.selected_square = square
                self.legal_moves = [move for move in self.board.legal_moves if move.from_square == square]
                self.update_board()
                return
                
            # Se clicar em um espaço vazio ou peça adversária (não sendo um destino válido), desselecionar
            self.selected_square = None
            self.legal_moves = []
            self.update_board()
            
        else:
            # Primeira seleção - verificar se há uma peça própria na casa
            piece = self.board.piece_at(square)
            if piece and piece.color == self.board.turn:
                self.selected_square = square
                self.legal_moves = [move for move in self.board.legal_moves if move.from_square == square]
                self.update_board()
                
    def make_player_move(self, move):
        """Executa o movimento do jogador"""
        # Converter para SAN para registro
        move_san = self.board.san(move)
        
        # Executar o movimento
        self.board.push(move)
        
        # Resetar seleção
        self.selected_square = None
        self.legal_moves = []
        
        # Atualizar contadores
        self.move_count += 1
        if self.board.turn:  # Se true, então as pretas acabaram de jogar
            self.black_moves += 1
        else:  # Caso contrário, as brancas acabaram de jogar
            self.white_moves += 1
            
        # Log e atualização da interface
        player_color = "Brancas" if not self.board.turn else "Pretas"
        self.log_move(f"Jogador ({player_color})", move_san)
        self.update_board()
        self.update_stats()
        
        # Verificar fim de jogo
        if self.check_game_end():
            return
            
        # Passar a vez para a IA
        self.player_turn = False
        self.game_after_id = self.root.after(1000, self.auto_play)

    def log_move(self, player, move):
        move_num = (self.move_count // 2) + 1
        prefix = f"{move_num}." if not self.board.turn else ""  # Número de movimento antes da jogada das brancas
        self.log_area.insert(tk.END, f"{prefix} {player}: {move}\n")
        self.log_area.see(tk.END)

    def start_game(self):
        # Resetar o tabuleiro e estatísticas
        self.board.reset()
        self.log_area.delete(1.0, tk.END)
        self.move_count = 0
        self.white_moves = 0
        self.black_moves = 0
        self.api_calls = 0
        self.ai_retry_count = 0
        self.selected_square = None
        self.legal_moves = []
        self.update_stats()
        
        # Atualizar interface
        self.update_board()
        self.start_button.config(state=tk.DISABLED)
        self.pause_button.config(text="Pausar", state=tk.NORMAL)
        self.stop_button.config(state=tk.NORMAL)
        
        # Inicializar modelos com base nas seleções
        self.model_white = ChessAI(
            self.white_model_var.get(), 
            self.board, 
            self.white_skill_var.get(),
            self.white_specific_model_var.get()
        )
        self.model_black = ChessAI(
            self.black_model_var.get(), 
            self.board, 
            self.black_skill_var.get(),
            self.black_specific_model_var.get()
        )
        
        # Atualizar estado
        self.game_running = True
        self.paused = False
        
        # Verificar o modo de jogo
        if self.game_mode_var.get() == "Jogador vs IA":
            player_is_white = self.player_color_var.get() == "Brancas"
            self.player_turn = player_is_white  # Se o jogador for brancas, começa jogando
            
            if player_is_white:
                self.status_var.set("Sua vez - escolha uma peça para mover")
            else:
                self.status_var.set("IA pensando...")
                self.game_after_id = self.root.after(1000, self.auto_play)
        else:
            # Modo IA vs IA
            self.player_turn = False
            self.status_var.set("Jogo iniciado - IA vs IA")
            self.game_after_id = self.root.after(1000, self.auto_play)
        
    def toggle_pause(self):
        if not self.game_running:
            return
            
        self.paused = not self.paused
        self.pause_button.config(text="Continuar" if self.paused else "Pausar")
        
        if self.paused:
            if self.game_after_id:
                self.root.after_cancel(self.game_after_id)
                self.game_after_id = None
            self.status_var.set("Jogo pausado")
        else:
            if not self.player_turn:
                self.status_var.set("Jogo continuando")
                self.game_after_id = self.root.after(1000, self.auto_play)
            else:
                self.status_var.set("Sua vez - escolha uma peça para mover")
            
    def stop_game(self):
        if self.game_after_id:
            self.root.after_cancel(self.game_after_id)
            self.game_after_id = None
            
        self.game_running = False
        self.paused = False
        self.player_turn = False
        
        self.start_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.DISABLED)
        
        self.status_var.set("Jogo interrompido")
    
    def update_stats(self):
        self.move_count_var.set(f"Jogadas: {self.move_count}")
        self.white_moves_var.set(f"Brancas: {self.white_moves} jogadas")
        self.black_moves_var.set(f"Pretas: {self.black_moves} jogadas")
        self.api_calls_var.set(f"Chamadas API: {self.api_calls}")
    
    def on_closing(self):
        self.stop_game()
        # Limpar diretório temporário
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass
        self.root.destroy()
    
    def get_ai_move(self):
        # Essa função é executada em uma thread separada
        self.thinking = True
        
        history = ' '.join(move.uci() for move in self.board.move_stack)
        fen = self.board.fen()
        current_model = self.model_white if self.board.turn else self.model_black
        
        # Configurar o uso de fallback baseado na opção da interface
        current_model.use_random_fallback = self.use_random_fallback_var.get()
        
        # Verificar se é um modelo de reflexão
        is_reflection = ai_config.is_reflection_model(current_model.specific_model)
        
        # Atualizar status para indicar que o modelo está "pensando"
        if is_reflection:
            self.status_var.set(f"Aguardando reflexão do modelo DeepSeek (pode levar mais tempo)")
            
        # Adicionar uma referência para atualizar o status (usado na função huggingface_move)
        current_model._notify_status_update = self.update_status_from_thread
            
        # Obter jogada
        move_san = current_model.get_move(history, fen)
        
        self.api_calls += 1
        self.thinking = False
        
        # Enviar resultado de volta para a thread principal
        if self.game_running and not self.paused:
            self.root.after(0, lambda: self.process_ai_move(move_san))
    
    def update_status_from_thread(self, message):
        # Função para atualizar o status a partir da thread de IA
        self.root.after(0, lambda: self.status_var.set(message))

    def process_ai_move(self, move_san):
        if not self.game_running or self.paused:
            return
            
        if move_san is None:
            # Agora que temos o sistema de fallback, esse caso deve ser raro
            self.status_var.set("Erro: IA não conseguiu gerar uma jogada válida")
            messagebox.showerror("Erro", "A IA não conseguiu gerar uma jogada válida mesmo após tentativas de fallback.")
            self.stop_game()
            return

        try:
            # Verificar explicitamente se a jogada é válida
            move = self.board.parse_san(move_san)
            if move not in self.board.legal_moves:
                raise ValueError(f"Jogada ilegal: {move_san}")
                
            # Se chegou até aqui, a jogada é válida
            self.board.push(move)
            
            # Atualizar contadores
            self.move_count += 1
            if self.board.turn:  # Se true, então as pretas acabaram de jogar
                self.black_moves += 1
            else:  # Caso contrário, as brancas acabaram de jogar
                self.white_moves += 1
                
            # Log e atualização da interface
            current_model = self.model_white if not self.board.turn else self.model_black
            model_short_name = current_model.specific_model.split('-')[0]
            self.log_move(f"{current_model.model_name} ({model_short_name})", move_san)
            self.update_board()
            self.update_stats()
            
            # Verificar fim de jogo
            if self.check_game_end():
                return
                
            # Verificar limite de jogadas
            if self.move_count >= self.move_limit_var.get() * 2:
                self.status_var.set(f"Jogo finalizado: limite de {self.move_limit_var.get()} jogadas atingido")
                messagebox.showinfo("Fim de jogo", f"Limite de {self.move_limit_var.get()} jogadas atingido")
                self.stop_game()
                return
                
            # Se for modo "Jogador vs IA", passar a vez para o jogador
            if self.game_mode_var.get() == "Jogador vs IA":
                player_color = self.player_color_var.get()
                if (player_color == "Brancas" and self.board.turn == chess.WHITE) or \
                   (player_color == "Pretas" and self.board.turn == chess.BLACK):
                    self.player_turn = True
                    self.status_var.set("Sua vez - escolha uma peça para mover")
                    return
                    
            # Continuar o jogo após a pausa definida pelo usuário
            self.game_after_id = self.root.after(self.time_var.get(), self.auto_play)
            
        except Exception as e:
            self.status_var.set(f"Erro na jogada: {str(e)}")
            print(f"Erro ao processar jogada: {e}")
            print(f"Jogada recebida: '{move_san}'")
            traceback.print_exc()
            
            # Tentar mais uma vez com o fallback
            if self.use_random_fallback_var.get():
                try:
                    legal_moves = list(self.board.legal_moves)
                    if legal_moves:
                        random_move = random.choice(legal_moves)
                        random_move_san = self.board.san(random_move)
                        self.status_var.set(f"Usando jogada aleatória: {random_move_san}")
                        self.game_after_id = self.root.after(1000, lambda: self.process_ai_move(random_move_san))
                        return
                except Exception as fallback_error:
                    print(f"Erro no fallback: {fallback_error}")
            
            # Se chegou aqui, realmente não conseguiu fazer uma jogada
            messagebox.showerror("Erro", f"Jogada inválida: {move_san}\nErro: {str(e)}")
            self.stop_game()

    def check_game_end(self):
        if self.board.is_game_over():
            result = self.board.result()
            reason = ""
            
            if self.board.is_checkmate():
                reason = "Xeque-mate"
            elif self.board.is_stalemate():
                reason = "Afogamento"
            elif self.board.is_insufficient_material():
                reason = "Material insuficiente"
            elif self.board.is_seventyfive_moves():
                reason = "Regra das 75 jogadas"
            elif self.board.is_fivefold_repetition():
                reason = "Repetição quíntupla"
            
            self.status_var.set(f"Jogo finalizado: {reason}, Resultado: {result}")
            messagebox.showinfo("Fim de jogo", f"{reason}\nResultado: {result}")
            self.stop_game()
            return True
        return False

    def auto_play(self):
        if not self.game_running or self.paused or self.thinking:
            return
            
        # Verificar se é a vez do jogador humano
        if self.game_mode_var.get() == "Jogador vs IA" and not self.player_turn:
            return
            
        # Verificar fim de jogo
        if self.check_game_end():
            return
            
        current_model = self.model_white if self.board.turn else self.model_black
        self.status_var.set(f"Aguardando jogada de {current_model.model_name}...")
        
        # Iniciar thread para não bloquear a interface
        self.thread = threading.Thread(target=self.get_ai_move)
        self.thread.daemon = True
        self.thread.start()

if __name__ == '__main__':
    root = tk.Tk()
    root.geometry("900x700")  # Largura aumentada
    app = ChessApp(root)
    root.mainloop()