import duckdb
import data_ingestion as di
import data_cleaning as dc
import data_engineering as de
import model_building as mb
from data_ingestion import data_retrieval_desc, select_target


def run_pipeline(root_folder, target_input, target_index, n_trials=20, compare=True):
    target_results =   data_retrieval_desc(target_name=target_input)
    selected_meta = select_target(target_index, target_results)
    engine = di.DuckDBEngine(
        connection= duckdb.connect(),
        targeted=selected_meta,
        root_folder=root_folder,
    )
    csv_path, table_name = engine.create_and_load_csv()
    engine.create_database(csv_filepath=csv_path, table_name=table_name)
    print(engine.query_check(table_name=table_name))

    clean_path = dc.DataCleaning(path=csv_path).relationalvalue()
    data_engine = de.DataEng(path=clean_path)
    model_runner = mb.Model(path=clean_path)

    if compare:
        comparison = model_runner.compare_models(n_trials=n_trials)
    else:
        model_runner.RFmodelc()
        comparison = None

    print(f"Generated database CSV: {csv_path}")
    print(f"Generated cleaned CSV: {clean_path}")
    for name, path in model_runner.output_locations().items():
        print(f"{name}: {path}")
    return {
        "engine": engine,
        "data_engine": data_engine,
        "model_runner": model_runner,
        "comparison": comparison,
        "clean_path": clean_path,
    }
    
if __name__ == "__main__":
    mb.Model.main()