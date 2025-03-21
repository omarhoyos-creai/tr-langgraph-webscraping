import os
import gradio as gr
import agents.agent as ag
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
import logging

_ = load_dotenv()

def get_answer(question):
    with SqliteSaver.from_conn_string(":memory:") as checkpointer:
        agent = ag.Agent(checkpointer=checkpointer)
        thread = {"configurable": {"thread_id": "1"}}
        results = []
        for s in agent.graph.stream({
                'task': question,
                "max_revisions": 2,
                "revision_number": 1,
                "plan": "",
                "draft": "",
                "critique": "",
                "content": [],
                "revision_number": None,
                "max_revisions": None,
            },
            thread,
            stream_mode="debug",
            debug=True
        ):
            results.append(s)
    return results

demo = gr.Interface(
    fn=get_answer,
    inputs=["text"],
    outputs=["text"],
)

demo.launch()