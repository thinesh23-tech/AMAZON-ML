import duckdb
import os
import time

# ============================================================
# CONFIGURATION
# ============================================================

BASE = r"C:\Users\Thinesh\Downloads\6ab10eb3b23ba_student_resource\student_resource"

TRAIN_DIR = os.path.join(BASE, "dataset", "train")
TEST_DIR = os.path.join(BASE, "dataset", "test")

OUTPUT_DIR = os.path.join(BASE, "processed")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# FILES
# ============================================================

TRAIN_FILES = {
    "source1": os.path.join(TRAIN_DIR, "train_source1.tsv"),
    "source2": os.path.join(TRAIN_DIR, "train_source2.tsv"),
    "source3": os.path.join(TRAIN_DIR, "train_source3.tsv"),
}

TEST_FILES = {
    "source1": os.path.join(TEST_DIR, "test_source1.tsv"),
    "source2": os.path.join(TEST_DIR, "test_source2.tsv"),
    "source3": os.path.join(TEST_DIR, "test_source3.tsv"),
}

GROUND_TRUTH = os.path.join(
    TRAIN_DIR,
    "train_ground_truth.tsv"
)

# ============================================================
# DUCKDB
# ============================================================

con = duckdb.connect(
    os.path.join(OUTPUT_DIR, "entity_resolution.duckdb")
)

# Adjust according to your RAM
con.execute("PRAGMA memory_limit='8GB'")

# Use multiple CPU cores
con.execute("PRAGMA threads=8")

# Temporary files
TEMP_DIR = os.path.join(OUTPUT_DIR, "duckdb_temp")
os.makedirs(TEMP_DIR, exist_ok=True)

con.execute(
    f"PRAGMA temp_directory='{TEMP_DIR}'"
)


# ============================================================
# READ TSV FUNCTION
# ============================================================

def read_tsv(path):

    return f"""
        read_csv(
            '{path}',
            delim='\\t',
            header=true,
            null_padding=true,
            ignore_errors=false
        )
    """


# ============================================================
# CREATE SOURCE TABLES
# ============================================================

print("\n========================================")
print("CREATING SOURCE TABLES")
print("========================================")

start = time.time()

for source, path in TRAIN_FILES.items():

    print(f"\nProcessing {source}...")

    query = f"""
        CREATE OR REPLACE TABLE train_{source} AS

        SELECT
            entity_id,
            business_name,
            business_address,
            country
        FROM {read_tsv(path)}
    """

    con.execute(query)

    count = con.execute(
        f"SELECT COUNT(*) FROM train_{source}"
    ).fetchone()[0]

    print(f"Rows: {count:,}")


print(
    f"\nSource loading completed in "
    f"{time.time() - start:.2f} seconds"
)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

print("\n========================================")
print("NORMALIZING TEXT")
print("========================================")

for source in TRAIN_FILES:

    print(f"Normalizing {source}...")

    con.execute(f"""
        CREATE OR REPLACE TABLE normalized_{source} AS

        SELECT

            entity_id,

            business_name,

            business_address,

            country,

            -- Normalized business name
            lower(
                regexp_replace(
                    regexp_replace(
                        coalesce(business_name, ''),
                        '[^\\p{{L}}\\p{{N}} ]',
                        ' ',
                        'g'
                    ),
                    '\\s+',
                    ' ',
                    'g'
                )
            ) AS name_normalized,

            -- Normalized address
            lower(
                regexp_replace(
                    regexp_replace(
                        coalesce(business_address, ''),
                        '[^\\p{{L}}\\p{{N}} ]',
                        ' ',
                        'g'
                    ),
                    '\\s+',
                    ' ',
                    'g'
                )
            ) AS address_normalized,

            -- Normalized country
            lower(
                trim(coalesce(country, ''))
            ) AS country_normalized,

            -- Missing indicators
            CASE
                WHEN business_name IS NULL
                     OR trim(business_name) = ''
                THEN 1
                ELSE 0
            END AS name_missing,

            CASE
                WHEN business_address IS NULL
                     OR trim(business_address) = ''
                THEN 1
                ELSE 0
            END AS address_missing

        FROM train_{source}
    """)


# ============================================================
# CREATE COMBINED TABLE
# ============================================================

print("\n========================================")
print("CREATING COMBINED DATASET")
print("========================================")

con.execute("""
    CREATE OR REPLACE TABLE all_entities AS

    SELECT
        *,
        'source1' AS source
    FROM normalized_source1

    UNION ALL

    SELECT
        *,
        'source2' AS source
    FROM normalized_source2

    UNION ALL

    SELECT
        *,
        'source3' AS source
    FROM normalized_source3
""")


# ============================================================
# SAVE AS PARQUET
# ============================================================

print("\n========================================")
print("WRITING PARQUET")
print("========================================")

output_file = os.path.join(
    OUTPUT_DIR,
    "normalized_entities.parquet"
)

con.execute(f"""
    COPY all_entities
    TO '{output_file}'
    (
        FORMAT PARQUET,
        COMPRESSION ZSTD
    )
""")


# ============================================================
# GROUND TRUTH
# ============================================================

print("\n========================================")
print("LOADING GROUND TRUTH")
print("========================================")

con.execute(f"""
    CREATE OR REPLACE TABLE ground_truth AS

    SELECT
        source1_entity_id,
        matched_entity_ids
    FROM {read_tsv(GROUND_TRUTH)}
""")


ground_truth_output = os.path.join(
    OUTPUT_DIR,
    "ground_truth.parquet"
)

con.execute(f"""
    COPY ground_truth
    TO '{ground_truth_output}'
    (
        FORMAT PARQUET,
        COMPRESSION ZSTD
    )
""")


# ============================================================
# BASIC STATISTICS
# ============================================================

print("\n========================================")
print("DATASET STATISTICS")
print("========================================")

stats = con.execute("""
    SELECT
        source,
        COUNT(*) AS rows,
        SUM(name_missing) AS missing_names,
        SUM(address_missing) AS missing_addresses
    FROM all_entities
    GROUP BY source
    ORDER BY source
""").fetchdf()

print(stats.to_string(index=False))


# ============================================================
# COUNTRY DISTRIBUTION
# ============================================================

print("\n========================================")
print("COUNTRY DISTRIBUTION")
print("========================================")

countries = con.execute("""
    SELECT
        country_normalized,
        COUNT(*) AS count
    FROM all_entities
    GROUP BY country_normalized
    ORDER BY count DESC
    LIMIT 20
""").fetchdf()

print(countries.to_string(index=False))


# ============================================================
# FINISH
# ============================================================

con.close()

print("\n========================================")
print("PREPROCESSING COMPLETED")
print("========================================")

print(f"\nOutput directory:")
print(OUTPUT_DIR)

print("\nGenerated:")
print("  normalized_entities.parquet")
print("  ground_truth.parquet")
print("  entity_resolution.duckdb")