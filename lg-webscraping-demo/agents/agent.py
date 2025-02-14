import os
from langchain.chains import LLMChain
from langchain_community.llms.cloudflare_workersai import CloudflareWorkersAI
from langchain_core.prompts import PromptTemplate

class Agent:
    def __init__(self):
        print(os.getenv("CLOUDFLARE_ACCOUNT_ID"))
        self.llm = CloudflareWorkersAI(
            account_id=os.getenv("CLOUDFLARE_ACCOUNT_ID"),
            api_token=os.getenv("CLOUDFLARE_API_TOKEN"),
            llm_model = "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b"
        )
        self.template = """Human: {question}

        AI Assistant:"""
        self.prompt = PromptTemplate.from_template(self.template)

    def run(self, question):
        llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)
        # question = "What is the capital of France?"
        result = llm_chain.run(question)
        return result