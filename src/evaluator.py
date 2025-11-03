import json
import time
import sqlite3
import multiprocessing
import pandas as pd
from tqdm import tqdm

def get_data_df():
    diff_json_path = '/home/sql/people/hyeonjin/mini_dev/llm/mini_dev_data/data_minidev/MINIDEV/mini_dev_sqlite.json'
    with open(diff_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data_df = pd.DataFrame(data)
    return data_df

def get_df():
    df = pd.read_csv(f'/home/sql/people/hyeonjin/minibird_eval_last/minibird_eval_result.csv')
    
    data_df = get_data_df()

    df['difficulty'] = data_df['difficulty']

    return df

def get_diff_df(df):
    diff_dict = {}
    for _, row in df.iterrows():
        if row['db_id'] not in diff_dict.keys():
            diff_dict[row['db_id']] = {'simple': 0, 'moderate': 0, 'challenging': 0}
            diff = row['difficulty']
            diff_dict[row['db_id']][diff] += 1
        else:
            diff = row['difficulty']
            diff_dict[row['db_id']][diff] += 1
    
    diff_df = pd.DataFrame.from_dict(diff_dict)
    diff_df = diff_df.T

    diff_df['total'] = diff_df['simple'] + diff_df['moderate'] + diff_df['challenging']

    return diff_df

def run_query(db_path, query, return_dict):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        start = time.time()
        cursor.execute(query)
        result = cursor.fetchall()
        end = time.time()

        return_dict['time'] = end - start
    
    except Exception as e:
        return_dict['error'] = str(e)
    
    finally:
        if 'conn' in locals():
            conn.close()


def get_ex_time(db_id, query, timeout=5):
    db_path = f"/home/sql/people/hyeonjin/mini_dev/llm/mini_dev_data/data_minidev/MINIDEV/dev_databases/{db_id}/{db_id}.sqlite"

    manager = multiprocessing.Manager()

    return_dict = manager.dict()

    p = multiprocessing.Process(target=run_query, args=(db_path, query, return_dict))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        return 999999
    
    if 'error' in return_dict:
        return 999999
    
    return return_dict.get('time', 999999)

def get_time_ratio(db_id, gold_query, pred_query):
    gold_time = get_ex_time(db_id, gold_query)
    pred_time = get_ex_time(db_id, pred_query)

    if pred_time == 999999:
        return 0
    else:
        ratio = gold_time / pred_time
        return ratio
    
def get_reward(db_id, gold_query, pred_query):
    ratio = get_time_ratio(db_id, gold_query, pred_query)
    if ratio == 0:
        reward = 0
    elif ratio >= 2:
        reward = 1.25
    elif ratio >= 1 and ratio < 2:
        reward = 1
    elif ratio >= 0.5 and ratio < 1:
        reward = 0.75
    elif ratio >= 0.25 and ratio < 0.5:
        reward = 0.5
    else:
        reward = 0.25

    
    return reward

def get_rves(df):
    rewards = {}

    for _, row in tqdm(df.iterrows()):
        db_id = row['db_id']
        gold_sql = row['gold_sql']
        pred_sql = row['pred_sql']

        q_id = row['question_id']

        reward = get_reward(db_id, gold_sql, pred_sql)

        rewards[q_id] = reward

    return rewards

def merge_rves(rves_dict, df):
    rves_df = pd.DataFrame.from_dict(rves_dict, orient='index', columns=['r-ves'])
    rves_df = rves_df.reset_index()
    rves_df = rves_df.rename(columns={'index': 'question_id'})

    merged = df.merge(rves_df[['question_id', 'r-ves']], on='question_id', how='left')

    df['r-ves'] = merged['r-ves']

    return df

def calculate_row_match(pred_row, gold_row):
    total_cols = len(gold_row)
    matches = 0
    pred_only = 0
    gold_only = 0
    for pred_val in pred_row:
        if pred_val in gold_row:
            matches += 1
        else:
            pred_only += 1
    for gold_val in gold_row:
        if gold_val not in pred_row:
            gold_only += 1
    
    match_percentage = matches / total_cols
    pred_only_percentage = pred_only / total_cols
    gold_only_percentage = gold_only / total_cols

    return match_percentage, pred_only_percentage, gold_only_percentage

def calculate_soft_f1_score(pred, gold):
    # if both pred and gold are empty, return 1.0 for f_1 score
    if pred.empty and gold.empty:
        return 1.0
    
    # Drop duplicate
    pred_set = set(pred) if not pred.empty else set()
    gold_set = set(gold) if not gold.empty else set()

    # convert back to list
    pred_list = list(pred_set)
    gold_list = list(gold_set)

    # Calculate matching score for each possible pair
    match_scores = []
    pred_only_scores = []
    gold_only_scores = []

    for i, gold_row in enumerate(gold):
        # rows only in the gold results
        if i >= len(pred_list):
            match_scores.append(0)
            gold_only_scores.append(1)
            continue

        pred_row = pred_list[i]
        match_score, pred_only_score, gold_only_score = calculate_row_match(pred_row, gold_row)

        match_scores.append(match_score)
        pred_only_scores.append(pred_only_score)
        gold_only_scores.append(gold_only_score)

    # rows only in the pred results
    for i in range(len(pred_list) - len(gold_list)):
        match_scores.append(0)
        pred_only_scores.append(1)
        gold_only_scores.append(0)
    
    tp = sum(match_scores)
    fp = sum(pred_only_scores)
    fn = sum(gold_only_scores)

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0

    f1_score = (2 * precision * recall / (precision + recall) if precision + recall > 0 else 0)

    return f1_score

def evaluate_soft_f1_all(df, pred_col='pred_sql', gold_col='gold_sql'):
    per_query_f1s = []

    pred_series = df[pred_col]
    gold_series = df[gold_col]

    for pred_sql, gold_sql in zip(pred_series, gold_series):
        pred_set = pd.Series([pred_sql]) if pd.notna(pred_sql) else pd.Series([])
        gold_set = pd.Series([gold_sql]) if pd.notna(gold_sql) else pd.Series([])
        f1 = calculate_soft_f1_score(pred_set, gold_set)
        per_query_f1s.append(f1)

    df['f1_score'] = per_query_f1s
    
    pred_all = pred_series.dropna().reset_index(drop=True)
    gold_all = gold_series.dropna().reset_index(drop=True)
    overall_f1 = calculate_soft_f1_score(pred_all, gold_all)

    return per_query_f1s, overall_f1, df

def evaluate_all(df):
    rves_dict = get_rves(df)
    df = merge_rves(rves_dict, df)

    _, __, df = evaluate_soft_f1_all(df)

    return df

def print_result(df, metric='all'):
    diffs = list(df['difficulty'].unique())


    if metric == 'all':
        metrics = ['EM', 'EX', 'r-ves', 'f1_score']
        cnt_list = []
        score_dict = {}
        for metric in metrics:
            if metric not in score_dict.keys():
                score_dict[metric] = []
            for diff in diffs:
                diff_df = df[df['difficulty'] == diff]
                
                cnt_list.append(len(diff_df))
                score_dict[metric].append(diff_df[metric].mean() * 100)     
            cnt_list.append(len(df))
            score_dict[metric].append(df[metric].mean() * 100)

            levels = ["simple", "moderate", "challenging", "total"]

            print("{:20} {:20} {:20} {:20} {:20}".format("", *levels))
            print("{:20} {:<20} {:<20} {:<20} {:<20}".format("count", *cnt_list))

            print(
                f"======================================        {metric}        ====================================="
            )
            print("{:20} {:<20.2f} {:<20.2f} {:<20.2f} {:<20.2f}".format(metric, *score_dict[metric]))

            _, total_f1s, __ = evaluate_soft_f1_all(df)
            print(f"Macro F1 Score: {total_f1s:.4f}")


    else:
        cnt_list = []
        score_list = []
        for diff in diffs:
            diff_df = df[df['difficulty'] == diff]

            cnt_list.append(len(diff_df))
            score_list.append(diff_df[metric].mean() * 100)
        cnt_list.append(len(df))
        
        score_list.append(df[metric].mean() * 100)

        levels = ["simple", "moderate", "challenging", "total"]

        print("{:20} {:20} {:20} {:20} {:20}".format("", *levels))
        print("{:20} {:<20} {:<20} {:<20} {:<20}".format("count", *cnt_list))

        print(
            f"======================================        {metric}        ====================================="
        )
        print("{:20} {:<20.2f} {:<20.2f} {:<20.2f} {:<20.2f}".format(metric, *score_list))

        if metric == 'f1_score':
            _, total_f1s, __ = evaluate_soft_f1_all(df)
            print(f"Macro F1 Score: {total_f1s:.4f}")