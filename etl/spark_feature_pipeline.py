"""
Clinical Policy & Risk Engine (CPRE)
Module: PySpark Feature Engineering Pipeline
Description: Ingests CMS SynPUF Beneficiary & Claims data, calculates longitudinal
             utilization metrics, joins CDC SVI county metrics, and generates a
             modeling-ready Feature Store dataset.
"""

import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType


def create_spark_session(app_name: str = "CPRE_Feature_Engineering") -> SparkSession:
    """Initializes a local Spark Session optimized for multicore execution."""
    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )


def build_feature_store(
    beneficiary_path: str,
    inpatient_path: str,
    outpatient_path: str,
    svi_path: str,
    output_parquet_path: str,
):
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("ERROR")

    print("[1/5] Loading raw datasets...")

    # 1. Load Beneficiary Summary (Demographics & Chronic Conditions)
    df_ben = spark.read.csv(beneficiary_path, header=True, inferSchema=True)

    # 2. Load Inpatient Claims
    df_inp = spark.read.csv(inpatient_path, header=True, inferSchema=True)

    # 3. Load Outpatient Claims
    df_out = spark.read.csv(outpatient_path, header=True, inferSchema=True)

    # 4. Load CDC SVI Data
    df_svi = spark.read.csv(svi_path, header=True, inferSchema=True)

    print("[2/5] Cleaning & Standardizing Beneficiary Demographics...")

    # Standardize CMS Chronic Conditions: In SynPUF, 1 = Yes, 2 = No -> Recode to 1 and 0
    # Calculate age baseline (assuming reference year 2008)
    chronic_conditions = [
        "SP_ALZHDMTA", "SP_CHF", "SP_CHRNKIDN", "SP_CNCR", 
        "SP_COPD", "SP_DEPRESSN", "SP_DIABETES", "SP_ISCHMCHT", 
        "SP_OSTEOPRS", "SP_RA_OA", "SP_STRKETIA"
    ]

    ben_clean = (
        df_ben.withColumn("gender", F.when(F.col("BENE_SEX_IDENT_CD") == 1, "M").otherwise("F"))
        .withColumn("race", F.col("BENE_RACE_CD").cast(StringType()))
        .withColumn(
            "age",
            (2008 - F.substring(F.col("BENE_BIRTH_DT").cast(StringType()), 1, 4).cast(IntegerType())),
        )
        .withColumn("fips_state", F.lpad(F.col("SP_STATE_CODE").cast(StringType()), 2, "0"))
        .withColumn("fips_county", F.lpad(F.col("BENE_COUNTY_CD").cast(StringType()), 3, "0"))
        .withColumn("county_fips", F.concat(F.col("fips_state"), F.col("fips_county")))
    )

    # Apply 1/0 binary transformation to chronic conditions and compute Comorbidity Count
    comorbidity_expr = []
    for cond in chronic_conditions:
        clean_col = cond.lower().replace("sp_", "has_")
        ben_clean = ben_clean.withColumn(
            clean_col,
            F.when(F.col(cond) == 1, 1).otherwise(0)
        )
        comorbidity_expr.append(F.col(clean_col))

    ben_clean = ben_clean.withColumn(
        "comorbidity_count",
        sum(comorbidity_expr)
    )

    print("[3/5] Aggregating Longitudinal Inpatient & Outpatient Utilization...")

    # Inpatient Metrics: Total Admissions, Total Days (LOS), Total Paid, Max Single LOS
    inp_summary = df_inp.groupBy("DESYNPUF_ID").agg(
        F.count("CLM_ID").alias("total_inpatient_admissions"),
        F.sum("CLM_UTLZTN_DAY_CNT").alias("total_inpatient_days"),
        F.sum("CLM_PMT_AMT").alias("total_inpatient_paid"),
        F.max("CLM_UTLZTN_DAY_CNT").alias("max_single_los"),
        F.countDistinct("PRVDR_NUM").alias("distinct_inpatient_providers"),
    )

    # Outpatient Metrics: Total Encounters, Total Outpatient Paid
    out_summary = df_out.groupBy("DESYNPUF_ID").agg(
        F.count("CLM_ID").alias("total_outpatient_visits"),
        F.sum("CLM_PMT_AMT").alias("total_outpatient_paid"),
    )

    print("[4/5] Preparing CDC Social Vulnerability Index (SDoH)...")

    # Select core percentile ranking columns from CDC SVI
    # FIPS is standardized to 5-character string
    svi_clean = (
        df_svi.withColumn("fips_str", F.lpad(F.col("FIPS").cast(StringType()), 5, "0"))
        .select(
            F.col("fips_str").alias("svi_county_fips"),
            F.col("RPL_THEME1").cast(DoubleType()).alias("svi_socioeconomic_pctile"),
            F.col("RPL_THEME2").cast(DoubleType()).alias("svi_household_pctile"),
            F.col("RPL_THEME3").cast(DoubleType()).alias("svi_minority_pctile"),
            F.col("RPL_THEME4").cast(DoubleType()).alias("svi_housing_transp_pctile"),
            F.col("RPL_THEMES").cast(DoubleType()).alias("svi_overall_vulnerability"),
        )
        # Filter out invalid missing flags (-999 in CDC data)
        .filter(F.col("svi_overall_vulnerability") >= 0)
    )

    print("[5/5] Executing Distributed Joins & Generating Final Targets...")

    # Join Beneficiary with Utilization Summaries
    feature_df = (
        ben_clean.join(inp_summary, on="DESYNPUF_ID", how="left")
        .join(out_summary, on="DESYNPUF_ID", how="left")
        .fillna(
            {
                "total_inpatient_admissions": 0,
                "total_inpatient_days": 0,
                "total_inpatient_paid": 0.0,
                "max_single_los": 0,
                "distinct_inpatient_providers": 0,
                "total_outpatient_visits": 0,
                "total_outpatient_paid": 0.0,
            }
        )
    )

    # Spatial Join with CDC SVI
    feature_df = feature_df.join(
        svi_clean,
        feature_df["county_fips"] == svi_clean["svi_county_fips"],
        how="left",
    ).fillna(
        {
            "svi_socioeconomic_pctile": 0.5,
            "svi_household_pctile": 0.5,
            "svi_minority_pctile": 0.5,
            "svi_housing_transp_pctile": 0.5,
            "svi_overall_vulnerability": 0.5,
        }
    )

    # Compute Target Columns for Modeling:
    # 1. Prolonged LOS Flag (> 5 days inpatient stay)
    # 2. High Cost Outlier Flag (Inpatient Paid > $10,000)
    # 3. Readmission / Frequent Utilizer Flag (>= 2 Admissions)
    final_feature_store = (
        feature_df.withColumn(
            "target_prolonged_los",
            F.when(F.col("max_single_los") > 5, 1).otherwise(0),
        )
        .withColumn(
            "target_high_cost_outlier",
            F.when(F.col("total_inpatient_paid") > 10000, 1).otherwise(0),
        )
        .withColumn(
            "target_frequent_utilizer",
            F.when(F.col("total_inpatient_admissions") >= 2, 1).otherwise(0),
        )
        .select(
            "DESYNPUF_ID",
            "age",
            "gender",
            "race",
            "county_fips",
            "has_alzhdmta",
            "has_chf",
            "has_chrnkidn",
            "has_cncr",
            "has_copd",
            "has_depressn",
            "has_diabetes",
            "has_ischmcht",
            "has_osteoprs",
            "has_ra_oa",
            "has_strketia",
            "comorbidity_count",
            "total_inpatient_admissions",
            "total_inpatient_days",
            "total_inpatient_paid",
            "max_single_los",
            "distinct_inpatient_providers",
            "total_outpatient_visits",
            "total_outpatient_paid",
            "svi_socioeconomic_pctile",
            "svi_household_pctile",
            "svi_minority_pctile",
            "svi_housing_transp_pctile",
            "svi_overall_vulnerability",
            "target_prolonged_los",
            "target_high_cost_outlier",
            "target_frequent_utilizer",
        )
    )

    print("\nFeature Store Schema:")
    final_feature_store.printSchema()

    # Export to Parquet
    os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
    print(f"\nWriting Feature Store to Parquet at: {output_parquet_path}...")
    final_feature_store.write.mode("overwrite").parquet(output_parquet_path)

    record_count = spark.read.parquet(output_parquet_path).count()
    print(f"Pipeline completed successfully. Total processed records: {record_count:,}")

    spark.stop()


if __name__ == "__main__":
    # Define relative paths based on repository structure
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    BENEFICIARY_FILE = os.path.join(
        BASE_DIR, "data/raw/synpuf/DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv"
    )
    INPATIENT_FILE = os.path.join(
        BASE_DIR, "data/raw/synpuf/DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv"
    )
    OUTPATIENT_FILE = os.path.join(
        BASE_DIR, "data/raw/synpuf/DE1_0_2008_to_2010_Outpatient_Claims_Sample_1.csv"
    )
    SVI_FILE = os.path.join(
        BASE_DIR, "data/raw/svi/SVI_2022_US_COUNTY.csv"
    )
    OUTPUT_PARQUET = os.path.join(BASE_DIR, "data/processed/feature_store.parquet")

    # Verify input files exist before running
    for file_path in [BENEFICIARY_FILE, INPATIENT_FILE, OUTPATIENT_FILE, SVI_FILE]:
        if not os.path.exists(file_path):
            print(f"Error: Required file not found: {file_path}")
            print("Please ensure you downloaded the datasets into the data/raw/ folders.")
            sys.exit(1)

    build_feature_store(
        beneficiary_path=BENEFICIARY_FILE,
        inpatient_path=INPATIENT_FILE,
        outpatient_path=OUTPATIENT_FILE,
        svi_path=SVI_FILE,
        output_parquet_path=OUTPUT_PARQUET,
    )