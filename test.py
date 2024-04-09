from llama_index.llms.ollama import Ollama
llm = Ollama(model="codellama:latest", request_timeout=30.0, base_url="http://hercules.local:11434")
resp = llm.complete("Who is Paul Graham?")
print(resp)
