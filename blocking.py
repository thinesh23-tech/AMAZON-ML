import duckdb
import os
import time


# ============================================================
# CONFIGURATION
# ============================================================

# IMPORTANT:
# "student_resource" is the actual project directory
BASE = r"C:\Users\Thinesh\Downloads\6ab10eb3b23ba_student_resource\student_resource"

PROCESSED_DIR = os.path.join(
    BASE,
    "processed"
)

DB_FILE = os.path.join(
    PROCESSED_DIR,
    "entity_resolution.duckdb"
)

NORMALIZED_PARQUET = os.path.join(
    PROCESSED_DIR,
    "normalized_entities.parquet"
)

GROUND_TRUTH_PARQUET = os.path.join(
    PROCESSED_DIR,
    "ground_truth.parquet"
)

OUTPUT_DIR = os.path.join(
    PROCESSED_DIR,
    "blocking"
)

TEMP_DIR = os.path.join(
    OUTPUT_DIR,
    "duckdb_temp"
)

CANDIDATE_FILE = os.path.join(
    OUTPUT_DIR,
    "candidate_pairs.parquet"
)


# ============================================================
# PERFORMANCE SETTINGS
# ============================================================

MEMORY_LIMIT = "8GB"
THREADS = 4
MAX_TEMP_SIZE = "20GB"

# Ignore extremely large blocks
MAX_BLOCK_SIZE = 1000


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    TEMP_DIR,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("ENTITY RESOLUTION - OPTIMIZED BLOCKING")
print("=" * 70)

print("\nProject directory:")
print(BASE)

print("\nProcessed directory:")
print(PROCESSED_DIR)


# ============================================================
# VERIFY FILES
# ============================================================

print("\nChecking required files...")


if not os.path.exists(DB_FILE):

    raise RuntimeError(
        "\nDuckDB database not found:\n"
        f"{DB_FILE}\n\n"
        "Run preprocess_entity_resolution.py first."
    )


if not os.path.exists(NORMALIZED_PARQUET):

    raise RuntimeError(
        "\nNormalized Parquet file not found:\n"
        f"{NORMALIZED_PARQUET}\n\n"
        "Run preprocess_entity_resolution.py first."
    )


if not os.path.exists(GROUND_TRUTH_PARQUET):

    raise RuntimeError(
        "\nGround truth Parquet file not found:\n"
        f"{GROUND_TRUTH_PARQUET}\n\n"
        "Run preprocess_entity_resolution.py first."
    )


print("Required files found.")

print(
    f"\nNormalized data:\n"
    f"{NORMALIZED_PARQUET}"
)

print(
    f"\nGround truth:\n"
    f"{GROUND_TRUTH_PARQUET}"
)

print(
    f"\nDuckDB:\n"
    f"{DB_FILE}"
)


# ============================================================
# OPEN DUCKDB
# ============================================================

print("\nOpening DuckDB database...")

con = duckdb.connect(
    DB_FILE
)

con.execute(
    f"PRAGMA memory_limit='{MEMORY_LIMIT}'"
)

con.execute(
    f"PRAGMA threads={THREADS}"
)

con.execute(
    f"PRAGMA temp_directory='{TEMP_DIR}'"
)

con.execute(
    f"PRAGMA max_temp_directory_size='{MAX_TEMP_SIZE}'"
)

con.execute(
    "SET preserve_insertion_order=false"
)


print(
    f"Memory limit       : {MEMORY_LIMIT}"
)

print(
    f"Threads             : {THREADS}"
)

print(
    f"Temp directory      : {TEMP_DIR}"
)

print(
    f"Temp disk limit     : {MAX_TEMP_SIZE}"
)


# ============================================================
# CHECK / RESTORE TABLES
# ============================================================

print("\nChecking normalized dataset...")

tables = con.execute(
    "SHOW TABLES"
).fetchall()

table_names = {
    row[0]
    for row in tables
}


# ============================================================
# RESTORE all_entities
# ============================================================

if "all_entities" not in table_names:

    print(
        "\n'all_entities' table not found."
    )

    print(
        "Restoring all_entities from Parquet..."
    )

    start = time.time()

    con.execute(
        f"""
        CREATE OR REPLACE TABLE all_entities AS

        SELECT *

        FROM read_parquet(
            '{NORMALIZED_PARQUET}'
        )
        """
    )

    elapsed = time.time() - start

    print(
        f"all_entities restored successfully "
        f"in {elapsed:.2f} seconds."
    )

else:

    print(
        "all_entities table found."
    )


# ============================================================
# RESTORE ground_truth
# ============================================================

tables = con.execute(
    "SHOW TABLES"
).fetchall()

table_names = {
    row[0]
    for row in tables
}


if "ground_truth" not in table_names:

    print(
        "\n'ground_truth' table not found."
    )

    print(
        "Restoring ground_truth from Parquet..."
    )

    start = time.time()

    con.execute(
        f"""
        CREATE OR REPLACE TABLE ground_truth AS

        SELECT *

        FROM read_parquet(
            '{GROUND_TRUTH_PARQUET}'
        )
        """
    )

    elapsed = time.time() - start

    print(
        f"ground_truth restored successfully "
        f"in {elapsed:.2f} seconds."
    )

else:

    print(
        "ground_truth table found."
    )


print(
    "\nRequired data is available."
)


# ============================================================
# VERIFY SCHEMA
# ============================================================

print(
    "\nVerifying all_entities schema..."
)

columns = con.execute(
    """
    DESCRIBE all_entities
    """
).fetchdf()


required_columns = {
    "entity_id",
    "source",
    "business_name",
    "business_address",
    "country",
    "name_normalized",
    "address_normalized",
    "country_normalized",
    "address_missing"
}


available_columns = set(
    columns["column_name"].tolist()
)


missing_columns = (
    required_columns -
    available_columns
)


if missing_columns:

    raise RuntimeError(
        "\nMissing required columns:\n"
        +
        "\n".join(
            sorted(missing_columns)
        )
    )


print(
    "Schema verification successful."
)


# ============================================================
# CREATE SOURCE BLOCK TABLES
# ============================================================

print(
    "\nCreating blocking source tables..."
)


def create_block_table(
    source_name,
    table_name
):

    start = time.time()

    con.execute(
        f"""
        CREATE OR REPLACE TABLE {table_name} AS

        SELECT

            entity_id,

            country_normalized,

            name_normalized,

            address_normalized,

            address_missing,

            LEFT(
                name_normalized,
                6
            ) AS name_prefix6,

            LEFT(
                name_normalized,
                4
            ) AS name_prefix4,

            split_part(
                name_normalized,
                ' ',
                1
            ) AS name_first_token,

            LEFT(
                address_normalized,
                8
            ) AS address_prefix8

        FROM all_entities

        WHERE source = '{source_name}'
        """
    )

    elapsed = time.time() - start

    count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM {table_name}
        """
    ).fetchone()[0]

    print(
        f"{table_name}: "
        f"{count:,} rows "
        f"({elapsed:.2f}s)"
    )


create_block_table(
    "source1",
    "block_source1"
)

create_block_table(
    "source2",
    "block_source2"
)

create_block_table(
    "source3",
    "block_source3"
)


print(
    "\nBlocking tables created."
)


# ============================================================
# CREATE CANDIDATE TABLE
# ============================================================

print(
    "\nCreating candidate table..."
)

con.execute(
    """
    DROP TABLE IF EXISTS candidate_pairs
    """
)

con.execute(
    """
    CREATE TABLE candidate_pairs (

        source1_entity_id VARCHAR,

        matched_entity_id VARCHAR,

        matched_source VARCHAR,

        blocking_method VARCHAR

    )
    """
)


# ============================================================
# VALID BLOCK CREATION
# ============================================================

def create_valid_blocks(
    table_name,
    column_name,
    block_table_name
):

    start = time.time()

    con.execute(
        f"""
        CREATE OR REPLACE TABLE
        {block_table_name}
        AS

        SELECT

            country_normalized,

            {column_name}
                AS block_key,

            COUNT(*) AS block_size

        FROM {table_name}

        WHERE

            {column_name} IS NOT NULL

            AND {column_name} <> ''

        GROUP BY

            country_normalized,

            {column_name}

        HAVING

            COUNT(*) <= {MAX_BLOCK_SIZE}
        """
    )

    elapsed = time.time() - start

    count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM {block_table_name}
        """
    ).fetchone()[0]

    print(
        f"  Valid blocks: "
        f"{count:,} "
        f"({elapsed:.2f}s)"
    )


# ============================================================
# NAME BLOCKING
# ============================================================

def run_name_block(
    target_table,
    target_source,
    column_name,
    method_name
):

    print(
        "\n" + "-" * 70
    )

    print(
        f"BLOCKING METHOD: {method_name}"
    )

    print(
        f"block_source1 -> {target_table}"
    )

    print(
        "-" * 70
    )

    start = time.time()

    valid_blocks = (
        f"valid_"
        f"{target_source}_"
        f"{column_name}"
    )

    print(
        "Creating valid blocks..."
    )

    create_valid_blocks(
        target_table,
        column_name,
        valid_blocks
    )

    print(
        "Generating candidates..."
    )

    con.execute(
        f"""
        INSERT INTO candidate_pairs

        SELECT

            s.entity_id
                AS source1_entity_id,

            t.entity_id
                AS matched_entity_id,

            '{target_source}'
                AS matched_source,

            '{method_name}'
                AS blocking_method

        FROM block_source1 s

        INNER JOIN {valid_blocks} b

            ON

                s.country_normalized =
                b.country_normalized

            AND

                s.{column_name} =
                b.block_key

        INNER JOIN {target_table} t

            ON

                t.country_normalized =
                b.country_normalized

            AND

                t.{column_name} =
                b.block_key

        WHERE

            s.{column_name} IS NOT NULL

            AND

            s.{column_name} <> ''
        """
    )

    elapsed = time.time() - start

    count = con.execute(
        """
        SELECT COUNT(*)
        FROM candidate_pairs
        """
    ).fetchone()[0]

    print(
        f"Completed in "
        f"{elapsed:.2f} seconds"
    )

    print(
        f"Candidate rows so far: "
        f"{count:,}"
    )


# ============================================================
# NAME PREFIX 6
# ============================================================

run_name_block(
    "block_source2",
    "source2",
    "name_prefix6",
    "country_name_prefix6_s2"
)

run_name_block(
    "block_source3",
    "source3",
    "name_prefix6",
    "country_name_prefix6_s3"
)


# ============================================================
# NAME PREFIX 4
# ============================================================

run_name_block(
    "block_source2",
    "source2",
    "name_prefix4",
    "country_name_prefix4_s2"
)

run_name_block(
    "block_source3",
    "source3",
    "name_prefix4",
    "country_name_prefix4_s3"
)


# ============================================================
# FIRST TOKEN
# ============================================================

run_name_block(
    "block_source2",
    "source2",
    "name_first_token",
    "country_first_token_s2"
)

run_name_block(
    "block_source3",
    "source3",
    "name_first_token",
    "country_first_token_s3"
)


# ============================================================
# ADDRESS BLOCKING
# ============================================================

def run_address_block(
    target_table,
    target_source,
    method_name
):

    print(
        "\n" + "-" * 70
    )

    print(
        f"BLOCKING METHOD: {method_name}"
    )

    print(
        f"block_source1 -> {target_table}"
    )

    print(
        "-" * 70
    )

    start = time.time()

    valid_blocks = (
        f"valid_"
        f"{target_source}_"
        f"address_prefix8"
    )

    print(
        "Creating valid address blocks..."
    )

    con.execute(
        f"""
        CREATE OR REPLACE TABLE
        {valid_blocks}
        AS

        SELECT

            country_normalized,

            address_prefix8
                AS block_key,

            COUNT(*) AS block_size

        FROM {target_table}

        WHERE

            address_prefix8 IS NOT NULL

            AND address_prefix8 <> ''

            AND address_missing = 0

        GROUP BY

            country_normalized,

            address_prefix8

        HAVING

            COUNT(*) <= {MAX_BLOCK_SIZE}
        """
    )

    valid_count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM {valid_blocks}
        """
    ).fetchone()[0]

    print(
        f"Valid address blocks: "
        f"{valid_count:,}"
    )

    print(
        "Generating address candidates..."
    )

    con.execute(
        f"""
        INSERT INTO candidate_pairs

        SELECT

            s.entity_id
                AS source1_entity_id,

            t.entity_id
                AS matched_entity_id,

            '{target_source}'
                AS matched_source,

            '{method_name}'
                AS blocking_method

        FROM block_source1 s

        INNER JOIN {valid_blocks} b

            ON

                s.country_normalized =
                b.country_normalized

            AND

                s.address_prefix8 =
                b.block_key

        INNER JOIN {target_table} t

            ON

                t.country_normalized =
                b.country_normalized

            AND

                t.address_prefix8 =
                b.block_key

        WHERE

            s.address_prefix8 IS NOT NULL

            AND

            s.address_prefix8 <> ''

            AND

            s.address_missing = 0

            AND

            t.address_missing = 0
        """
    )

    elapsed = time.time() - start

    count = con.execute(
        """
        SELECT COUNT(*)
        FROM candidate_pairs
        """
    ).fetchone()[0]

    print(
        f"Completed in "
        f"{elapsed:.2f} seconds"
    )

    print(
        f"Candidate rows so far: "
        f"{count:,}"
    )


# ============================================================
# ADDRESS PREFIX 8
# ============================================================

run_address_block(
    "block_source2",
    "source2",
    "country_address_prefix8_s2"
)

run_address_block(
    "block_source3",
    "source3",
    "country_address_prefix8_s3"
)


# ============================================================
# DEDUPLICATION
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "DEDUPLICATING CANDIDATE PAIRS"
)

print(
    "=" * 70
)

start = time.time()

con.execute(
    """
    CREATE OR REPLACE TABLE final_candidates AS

    SELECT DISTINCT

        source1_entity_id,

        matched_entity_id,

        matched_source

    FROM candidate_pairs
    """
)

elapsed = time.time() - start

print(
    f"Deduplication completed in "
    f"{elapsed:.2f} seconds"
)


# ============================================================
# CANDIDATE STATISTICS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "CANDIDATE STATISTICS"
)

print(
    "=" * 70
)


source1_count = con.execute(
    """
    SELECT COUNT(*)
    FROM block_source1
    """
).fetchone()[0]


source2_count = con.execute(
    """
    SELECT COUNT(*)
    FROM block_source2
    """
).fetchone()[0]


source3_count = con.execute(
    """
    SELECT COUNT(*)
    FROM block_source3
    """
).fetchone()[0]


candidate_count = con.execute(
    """
    SELECT COUNT(*)
    FROM final_candidates
    """
).fetchone()[0]


print(
    f"\nSource1 records : "
    f"{source1_count:,}"
)

print(
    f"Source2 records : "
    f"{source2_count:,}"
)

print(
    f"Source3 records : "
    f"{source3_count:,}"
)

print(
    f"\nCandidate pairs : "
    f"{candidate_count:,}"
)


# ============================================================
# CANDIDATES BY SOURCE
# ============================================================

print(
    "\nCandidate pairs by source:"
)

stats = con.execute(
    """
    SELECT

        matched_source,

        COUNT(*) AS candidate_pairs

    FROM final_candidates

    GROUP BY
        matched_source

    ORDER BY
        matched_source
    """
).fetchdf()

print(
    stats.to_string(
        index=False
    )
)


# ============================================================
# SEARCH SPACE
# ============================================================

possible_s1_s2 = (
    source1_count *
    source2_count
)

possible_s1_s3 = (
    source1_count *
    source3_count
)

possible_pairs = (
    possible_s1_s2 +
    possible_s1_s3
)


reduction = (
    1 -
    (
        candidate_count /
        possible_pairs
    )
) * 100


print(
    f"\nS1 x S2 possible pairs : "
    f"{possible_s1_s2:,}"
)

print(
    f"S1 x S3 possible pairs : "
    f"{possible_s1_s3:,}"
)

print(
    f"Naive possible pairs   : "
    f"{possible_pairs:,}"
)

print(
    f"Blocked candidate pairs: "
    f"{candidate_count:,}"
)

print(
    f"Search-space reduction : "
    f"{reduction:.6f}%"
)


# ============================================================
# GROUND TRUTH
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "BLOCKING RECALL EVALUATION"
)

print(
    "=" * 70
)

print(
    "\nExpanding ground truth..."
)

start = time.time()

con.execute(
    """
    CREATE OR REPLACE TABLE truth_pairs AS

    SELECT

        source1_entity_id,

        TRIM(
            UNNEST(
                STRING_SPLIT(
                    matched_entity_ids,
                    ','
                )
            )
        ) AS matched_entity_id

    FROM ground_truth

    WHERE

        matched_entity_ids IS NOT NULL

        AND

        TRIM(
            matched_entity_ids
        ) <> ''
    """
)

elapsed = time.time() - start

print(
    f"Ground truth expanded "
    f"in {elapsed:.2f} seconds"
)


true_pairs = con.execute(
    """
    SELECT COUNT(*)
    FROM truth_pairs
    """
).fetchone()[0]


print(
    f"\nGround-truth pairs : "
    f"{true_pairs:,}"
)


# ============================================================
# FIND TRUE PAIRS
# ============================================================

print(
    "\nChecking true pairs..."
)

start = time.time()

found_pairs = con.execute(
    """
    SELECT COUNT(*)

    FROM truth_pairs t

    INNER JOIN final_candidates c

        ON

            t.source1_entity_id =
            c.source1_entity_id

        AND

            t.matched_entity_id =
            c.matched_entity_id
    """
).fetchone()[0]

elapsed = time.time() - start


recall = (
    (
        found_pairs /
        true_pairs
    ) * 100

    if true_pairs > 0

    else 0
)


print(
    f"Found by blocking  : "
    f"{found_pairs:,}"
)

print(
    f"Blocking recall    : "
    f"{recall:.4f}%"
)

print(
    f"Recall evaluation time: "
    f"{elapsed:.2f} seconds"
)


# ============================================================
# SAVE FINAL CANDIDATES
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "SAVING CANDIDATES"
)

print(
    "=" * 70
)

start = time.time()

con.execute(
    f"""
    COPY final_candidates

    TO '{CANDIDATE_FILE}'

    (
        FORMAT PARQUET,
        COMPRESSION ZSTD
    )
    """
)

elapsed = time.time() - start

print(
    f"Saved in {elapsed:.2f} seconds"
)

print(
    "\nFile:"
)

print(
    CANDIDATE_FILE
)


# ============================================================
# METHOD CONTRIBUTION
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "BLOCKING METHOD CONTRIBUTION"
)

print(
    "=" * 70
)


method_stats = con.execute(
    """
    SELECT

        blocking_method,

        COUNT(*) AS candidate_pairs

    FROM candidate_pairs

    GROUP BY

        blocking_method

    ORDER BY

        candidate_pairs DESC
    """
).fetchdf()


print(
    method_stats.to_string(
        index=False
    )
)


# ============================================================
# CLEANUP
# ============================================================

print(
    "\nCleaning intermediate candidate table..."
)

con.execute(
    """
    DROP TABLE IF EXISTS candidate_pairs
    """
)


# ============================================================
# CLOSE
# ============================================================

con.close()


# ============================================================
# COMPLETE
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "BLOCKING COMPLETED SUCCESSFULLY"
)

print(
    "=" * 70
)

print(
    "\nCandidate file:"
)

print(
    CANDIDATE_FILE
)

print(
    "\nNext stage:"
)

print(
    "Candidate pairs -> similarity features -> ML model"
)