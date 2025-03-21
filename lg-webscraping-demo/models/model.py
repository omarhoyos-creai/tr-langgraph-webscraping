import os
from langchain_community.chat_models.cloudflare_workersai import ChatCloudflareWorkersAI

class CloudflareModel:
    def __init__(self):
        self.model = ChatCloudflareWorkersAI(
            account_id=os.getenv("CLOUDFLARE_ACCOUNT_ID"),
            api_token=os.getenv("CLOUDFLARE_API_TOKEN"),
            model = "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b"
        )