import os
from dotenv import load_dotenv
from tavily import TavilyClient

_ = load_dotenv()

tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])