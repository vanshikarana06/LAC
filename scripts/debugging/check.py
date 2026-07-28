"""
Reads ragas_results.csv (already computed) and reprints the summary.
No LLM calls, no rerun — instant, since the data is already saved.
"""

import pandas as pd

CSV_PATH = "ragas_results.csv"


def main():
    df = pd.read_csv(CSV_PATH)

    print("Columns in saved file:", df.columns.tolist())

    metric_cols = [c for c in
                    ["faithfulness", "answer_relevancy",
                     "llm_context_precision_with_reference", "context_precision",
                     "llm_context_recall", "context_recall"]
                    if c in df.columns]
    id_cols = [c for c in ["user_input", "query_source", "category"] if c in df.columns]

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(df[id_cols + metric_cols].to_string(index=False))

    if "category" in df.columns:
        print("\n" + "-" * 60)
        print("ANSWERABLE QUESTIONS — all 4 metrics are meaningful here")
        print("-" * 60)
        answerable_df = df[df["category"] == "answerable"]
        for metric in metric_cols:
            print(f"{metric}: {answerable_df[metric].mean():.3f}")

        print("\n" + "-" * 60)
        print("SHOULD-REFUSE QUESTIONS — only faithfulness/answer_relevancy are meaningful")
        print("-" * 60)
        refuse_df = df[df["category"] == "should_refuse"]
        for metric in ["faithfulness", "answer_relevancy"]:
            if metric in refuse_df.columns:
                print(f"{metric}: {refuse_df[metric].mean():.3f}")

        print("\n" + "-" * 60)
        print("AVERAGES BY ROUTING PATH (answerable questions only)")
        print("-" * 60)
        print(answerable_df.groupby("query_source")[metric_cols].mean())
    else:
        print("\n(No 'category' column found — this may be an older-format CSV. "
              "Showing overall averages instead.)")
        for metric in metric_cols:
            print(f"{metric}: {df[metric].mean():.3f}")


if __name__ == "__main__":
    main()