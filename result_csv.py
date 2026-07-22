import csv

file_path = "ragas_results.csv"

with open(file_path, "r", newline="", encoding="utf-8") as file:
    reader = csv.reader(file)

    for row in reader:
        print(row)