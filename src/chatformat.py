from llama_cpp.llama_chat_format import *
from llama_supercharged.src.llm import Llm

FORMATS = {
    'ChatFormatter': ChatFormatter,
    'ChatFormatterResponse': ChatFormatterResponse,
    'GLM41VChatHandler': GLM41VChatHandler,
    'GLM46VChatHandler': GLM46VChatHandler,
    'Gemma3ChatHandler': Gemma3ChatHandler,
    'GraniteDoclingChatHandler': GraniteDoclingChatHandler,
    'Jinja2ChatFormatter': Jinja2ChatFormatter,
    'LFM2VLChatHandler': LFM2VLChatHandler,
    'Llama3VisionAlpha': Llama3VisionAlpha,
    'Llama3VisionAlphaChatHandler': Llama3VisionAlphaChatHandler,
    'LlamaChatCompletionHandler': LlamaChatCompletionHandler,
    'LlamaChatCompletionHandlerNotFoundException': LlamaChatCompletionHandlerNotFoundException,
    'LlamaChatCompletionHandlerRegistry': LlamaChatCompletionHandlerRegistry,
    'Llava15ChatHandler': Llava15ChatHandler,
    'Llava16ChatHandler': Llava16ChatHandler,
    'MiniCPMv26ChatHandler': MiniCPMv26ChatHandler,
    'MiniCPMv45ChatHandler': MiniCPMv45ChatHandler,
    'MoondreamChatHandler': MoondreamChatHandler,
    'NanoLlavaChatHandler': NanoLlavaChatHandler,
    'ObsidianChatHandler': ObsidianChatHandler,
    'Qwen25VLChatHandler': Qwen25VLChatHandler,
    'Qwen3VLChatHandler': Qwen3VLChatHandler
}

class ChatFormat(Llm):
    def __init__(self, format: str, clip_model_path: str, **kwargs):
        chat_handler = FORMATS[format](clip_model_path=clip_model_path) if format in FORMATS else None
        super().__init__(chat_handler=chat_handler, **kwargs)

    def __call__(self):
        return super().__call__()
