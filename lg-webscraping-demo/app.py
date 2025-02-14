import os
import gradio as gr
import agents.agent as ag
from dotenv import load_dotenv

load_dotenv()
print(os.getenv("CLOUDFLARE_ACCOUNT_ID"))
print(os.getenv("CLOUDFLARE_API_TOKEN"))

def get_answer(question):
    agent = ag.Agent()
    return agent.run(question)

demo = gr.Interface(
    fn=get_answer,
    inputs=["text"],
    outputs=["text"],
)

demo.launch()