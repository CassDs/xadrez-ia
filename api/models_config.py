"""
Configuração de modelos de IA para o jogo de xadrez.
Este arquivo define as IAs disponíveis e seus respectivos modelos.
"""

class AIModelsConfig:
    """Classe que define as configurações dos modelos de IA disponíveis"""
    
    def __init__(self):
        # Dicionário de IAs disponíveis e seus modelos
        self.ai_systems = {
            "ChatGPT": {
                "name": "ChatGPT",
                "models": [
                    "gpt-3.5-turbo-16k", 
                    "gpt-4-turbo-preview", 
                    "gpt-4o",
                    "gpt-4.5-preview"
                ],
                "default_model": "gpt-4o",
                "api_name": "openai"
            },
            "Claude": {
                "name": "Claude",
                "models": [
                    "claude-3-haiku-20240307", 
                    "claude-3-5-haiku-20241022", 
                    "claude-3-opus-20240229", 
                    "claude-3-5-sonnet-20241022", 
                    "claude-3-7-sonnet-20250219"
                ],
                "default_model": "claude-3-7-sonnet-20250219",
                "api_name": "anthropic"
            },
            "Gemini": {
                 "name": "Gemini",
                 "models": [
                     "gemini-1.5-pro",
                     "gemini-1.5-flash",
                     "gemini-1.5-pro-latest",
                     "gemini-2.0-pro-exp-02-05"
                 ],
                 "default_model": "gemini-2.0-pro-exp-02-05",
                 "api_name": "Gemini"
            },
            "HuggingFace": {
                "name": "HuggingFace",
                "models": [
                    "01-ai/Yi-1.5-34B-Chat",
                    "CohereForAI/c4ai-command-r-plus-08-2024",
                    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                    "google/gemma-1.1-7b-it",
                    "google/gemma-2-27b-it",
                    "google/gemma-2-2b-it",
                    "HuggingFaceH4/starchat2-15b-v0.1",
                    "HuggingFaceH4/zephyr-7b-alpha",
                    "HuggingFaceH4/zephyr-7b-beta",
                    "meta-llama/Llama-2-7b-chat-hf",
                    "meta-llama/Llama-3.1-70B-Instruct",
                    "meta-llama/Llama-3.2-1B-Instruct",
                    "meta-llama/Meta-Llama-3-70B-Instruct",
                    "meta-llama/Meta-Llama-3-8B-Instruct",
                    "microsoft/DialoGPT-medium",
                    "microsoft/Phi-3-mini-4k-instruct",
                    "microsoft/Phi-3.5-mini-instruct",
                    "mistralai/Mistral-7B-Instruct-v0.2",
                    "mistralai/Mistral-7B-Instruct-v0.3",
                    "mistralai/Mistral-Nemo-Instruct-2407",
                    "mistralai/Mixtral-8x7B-Instruct-v0.1",
                    "NousResearch/Hermes-3-Llama-3.1-8B",
                    "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
                    "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
                    "Qwen/Qwen2.5-Coder-32B-Instruct",
                    "Qwen/QwQ-32B",
                    "tiiuae/falcon-7b-instruct",
                    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                ],
                "default_model": "microsoft/Phi-3-mini-4k-instruct",
                "api_name": "huggingface",
                "reflection_models": [
                    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
                ]
            }
        }
    
    def get_available_ais(self):
        """Retorna a lista de nomes de IAs disponíveis"""
        return list(self.ai_systems.keys())
    
    def get_models_for_ai(self, ai_name):
        """Retorna a lista de modelos disponíveis para uma IA específica"""
        if ai_name in self.ai_systems:
            return self.ai_systems[ai_name]["models"]
        return []
    
    def get_default_model(self, ai_name):
        """Retorna o modelo padrão para uma IA específica"""
        if ai_name in self.ai_systems:
            return self.ai_systems[ai_name]["default_model"]
        return None
    
    def get_api_name(self, ai_name):
        """Retorna o nome da API associada a uma IA específica"""
        if ai_name in self.ai_systems:
            return self.ai_systems[ai_name]["api_name"]
        return None
        
    def is_reflection_model(self, model_name):
        """Verifica se o modelo possui capacidade de reflexão"""
        for ai_name, config in self.ai_systems.items():
            if "reflection_models" in config and model_name in config["reflection_models"]:
                return True
        return False
