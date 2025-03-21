import os
from langgraph.graph import StateGraph, START, END
from agents.agent_state import AgentState
from models.model import CloudflareModel
from langchain_core.messages import SystemMessage, HumanMessage
from constants.prompts import PLAN_PROMPT, RESEARCH_PLAN_PROMPT, WRITER_PROMPT, REFLECTION_PROMPT, RESEARCH_CRITIQUE_PROMPT
from tools.tavily import tavily
from agents.queries import Queries
from pprint import pprint

# Get model from Cloudflare WorkersAI
model = CloudflareModel().model

class Agent:
    def __init__(self, checkpointer=None):
        builder = StateGraph(AgentState)
        builder.add_node("planner", self.plan_node)
        builder.add_node("generate", self.generation_node)
        builder.add_node("reflect", self.reflection_node)
        builder.add_node("research_plan", self.research_plan_node)
        builder.add_node("research_critique", self.research_critique_node)
        builder.set_entry_point("planner")
        builder.add_conditional_edges(
            "generate", 
            self.should_continue, 
            {END: END, "reflect": "reflect"}
        )
        builder.add_edge("planner", "research_plan")
        builder.add_edge("research_plan", "generate")
        builder.add_edge("reflect", "research_critique")
        builder.add_edge("research_critique", "generate")
        self.graph = builder.compile(checkpointer=checkpointer)

    def plan_node(self, state: AgentState):
        messages = [
            SystemMessage(content=PLAN_PROMPT), 
            HumanMessage(content=state['task'])
        ]
        response = model.invoke(messages)
        return {"plan": response}
    
    def research_plan_node(self, state: AgentState):
        queries = model.with_structured_output(Queries).invoke([
            SystemMessage(content=RESEARCH_PLAN_PROMPT),
            HumanMessage(content=state['task'])
        ])
        content = state["content"] or []
        for q in queries.queries:
            response = tavily.search(query=q, max_results=2)
            for r in response['results']:
                content.append(r['content'])
        return {"content": content}
    
    def generation_node(self, state: AgentState):
        content = "\n\n".join(state['content'] or [])
        user_message = HumanMessage(
            content=f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}")
        messages = [
            SystemMessage(
                content=WRITER_PROMPT.format(content=content)
            ),
            user_message
            ]
        response = model.invoke(messages)
        return {
            "draft": response.content, 
            "revision_number": state.get("revision_number", 1) + 1
        }
    
    def reflection_node(self, state: AgentState):
        messages = [
            SystemMessage(content=REFLECTION_PROMPT), 
            HumanMessage(content=state['draft'])
        ]
        response = model.invoke(messages)
        return {"critique": response.content}
    
    def research_critique_node(self, state: AgentState):
        queries = model.with_structured_output(Queries).invoke([
            SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
            HumanMessage(content=state['critique'])
        ])
        content = state['content'] or []
        for q in queries.queries:
            response = tavily.search(query=q, max_results=2)
            for r in response['results']:
                content.append(r['content'])
        return {"content": content}
    
    def should_continue(self, state):
        if state["revision_number"] > state["max_revisions"]:
            return END
        return "reflect"
    