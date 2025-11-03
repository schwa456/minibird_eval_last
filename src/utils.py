import os
import json
import torch
import sqlite3
import pandas as pd
from tqdm import tqdm
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase

import sys
sys.path.append('/home/sql/people/hyeonjin/M-Schema/')
from schema_engine import SchemaEngine

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from langchain_huggingface import HuggingFacePipeline

from .config import *

_llm = None
_tokenizer = None

def get_db(db_id):
    db_path = os.path.join(BASE_DB_PATH, db_id, f"{db_id}.sqlite")
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    engine = create_engine(f"sqlite:///{db_path}")
    return db, engine

def get_mschema(db_id, engine):
    sys.path.append('/home/sql/people/hyeonjin/M-Schema/')
    schema_engine = SchemaEngine(engine)
    mschema_str = schema_engine.mschema.to_mschema()
    return mschema_str

def load_dataset():
    with open(SQLITE_JSON_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_llm():
    global _llm, _tokenizer
    if _llm is None:
        _llm, _tokenizer = load_llm()
    return _llm

def load_llm():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using Device: {device}")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype="auto"
    )

    print(f"[INFO] MODEL ID : {MODEL_ID}")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return HuggingFacePipeline(pipeline=pipe), tokenizer

def evaluate_sql_em_ex(dataset, graph):
    records = []
    with tqdm(dataset) as pbar:
        for item in tqdm(dataset):
            db_id = item["db_id"]
            question = item['question']
            evidence = item['evidence']
            gold_sql = item["SQL"].strip()
            
            try:
                state = graph.invoke({
                    'question': question,
                    'evidence': evidence,
                    'db_id': db_id
                })
                pred_sql = state['query'].strip()
                em = int(gold_sql.lower() == pred_sql.lower())

                db_path = f"/home/sql/people/hyeonjin/mini_dev/llm/mini_dev_data/data_minidev/MINIDEV/dev_databases/{db_id}/{db_id}.sqlite"
                conn = sqlite3.connect(db_path)
                gold_result = conn.execute(gold_sql).fetchall()
                pred_result = conn.execute(pred_sql).fetchall()

                ex = int(sorted(gold_result) == sorted(pred_result))
                conn.close()
            except Exception as e:
                pred_sql = str(e)
                em = 0
                ex = 0

            records.append({
                'question_id': item['question_id'],
                'db_id': db_id,
                'question': question,
                'gold_sql': gold_sql,
                'pred_sql': pred_sql,
                'EM': em,
                'EX': ex
            })

            pbar.set_postfix({
                "MEAN_EM": f"{sum(record['EM'] for record in records) / len(records)}",
                "MEAN_EX": f"{sum(record['EX'] for record in records) / len(records)}"
            })

    df = pd.DataFrame(records)
    df.to_csv("/home/sql/people/hyeonjin/minibird_eval_last/minibird_eval_result.csv", index=False)
    print("Total EM:", df['EM'].mean())
    print("Total EX:", df['EX'].mean())