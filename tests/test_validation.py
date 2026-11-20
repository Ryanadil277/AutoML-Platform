from src.data_ingestion import load_data
from src.data_validation import validate_data


df = load_data("data/sample.csv")

report = validate_data(df)

print("\n===== DATA VALIDATION REPORT =====")

for key, value in report.items():
    print(f"\n{key}: {value}")