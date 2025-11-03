from src.utils import *
from src.graph import *
from src.evaluator import *

def main():
    llm = get_llm()
    graph = build_graph(llm)
    dataset = load_dataset()

    evaluate_sql_em_ex(dataset, graph)
    
    df = get_df()
    df = evaluate_all(df)
    print_result(df, metric='all')

if __name__ == '__main__':
    main()