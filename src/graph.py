import re
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain.prompts import SystemMessagePromptTemplate, ChatPromptTemplate

from .utils import *

def get_query_prompt_template():
    nl2sql_prompt = """You are an expert in {dialect}, and now you need to read and understand the following 【Database schema】 description, as well as any 【Reference Information】 that may be useful, and use {dialect} knowledge to generate SQL statements to answer 【User Questions】. Print intermediate steps and final SQL statements
    【User Questions】
    {question}

    【Database schema】
    {db_schema}

    【Reference Information】
    {evidence}

    【User Questions】
    {question}

    ```sql"""

    system_message_prompt = SystemMessagePromptTemplate.from_template(nl2sql_prompt)

    query_prompt_template = ChatPromptTemplate.from_messages([system_message_prompt])

    return query_prompt_template

def extract_sql_query(response_text: str) -> str:
    match = re.search(r"```sql\s*([\s\S]+?)```", response_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return response_text.strip()

class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str
    evidence: str
    db_id: str

class QueryOutput(TypedDict):
    """Generated SQL query."""

    query: Annotated[str, ..., "Syntactically valid SQL query."]

def extract_sql_query(response_text: str) -> str:
    match = re.search(r"```sql\s*([\s\S]+?)```", response_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    candidates = re.findall(r"(SELECT[\s\S]+?)(;|$)", response_text, re.IGNORECASE)
    if candidates:
        longest_query = max(candidates, key=lambda x: len(x[0]))[0]
        return longest_query.strip()
    
    return response_text.strip()

def build_graph(llm):
    def write_query(state: State):
        db_id = state['db_id']
        db, engine = get_db(db_id)
        mschema_str = get_mschema(db_id, engine)

        prompt_template = get_query_prompt_template()

        prompt = prompt_template.invoke(
            {
                "dialect": db.dialect,
                "top_k": 10,
                "db_schema": mschema_str,
                "question": state["question"],
                "evidence": state["evidence"],
            }
        )
        response = llm.invoke(prompt)
        query = response if isinstance(response, str) else response.content
        query = extract_sql_query(query)
        return {"query": query}
    
    builder = StateGraph(State).add_sequence([
        write_query
    ])

    builder.add_edge(START, 'write_query')
    return builder.compile()