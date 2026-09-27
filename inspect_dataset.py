import duckdb
import os

BASE = r"C:\Users\Thinesh\Downloads\6ab10eb3b23ba_student_resource\student_resource"

files = {
    "train_source1": os.path.join(BASE, "dataset", "train", "train_source1.tsv"),
    "train_source2": os.path.join(BASE, "dataset", "train", "train_source2.tsv"),
    "train_source3": os.path.join(BASE, "dataset", "train", "train_source3.tsv"),
    "ground_truth": os.path.join(BASE, "dataset", "train", "train_ground_truth.tsv"),
    "test_source1": os.path.join(BASE, "dataset", "test", "test_source1.tsv"),
    "test_source2": os.path.join(BASE, "dataset", "test", "test_source2.tsv"),
    "test_source3": os.path.join(BASE, "dataset", "test", "test_source3.tsv"),
}

con = duckdb.connect()

for name, path in files.items():

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    # Show columns
    schema = con.execute(f"""
        DESCRIBE
        SELECT *
        FROM read_csv(
            '{path}',
            delim='\\t',
            header=true,
            sample_size=10000
        )
    """).fetchall()

    print("\nColumns:")

    for row in schema:
        print(f"  {row[0]:30} {row[1]}")

    # Get a few records
    print("\nSample:")

    sample = con.execute(f"""
        SELECT *
        FROM read_csv(
            '{path}',
            delim='\\t',
            header=true,
            sample_size=10000
        )
        LIMIT 3
    """).fetchdf()

    print(sample.to_string(index=False))

con.close()

print("\nInspection completed.")