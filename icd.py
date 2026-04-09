"""
ICD-Based Hemoglobinopathy Phenotyping
======================================

Source-of-truth implementation for classifying patients based on ICD codes.

Phenotype Categories:
- SCD: Sickle Cell Disease (>2 encounters AND >5% of IP+OP)
- SCDX: Possible SCD (>2 encounters but ≤5%)
- THAL: Thalassemia (dominant ICD category)
- TRAIT: Sickle Cell Trait (dominant ICD category)
- OTHER: Other Hemoglobinopathies (dominant ICD category)
- UNK: Unknown/Unclassified (insufficient evidence)
"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import duckdb

from . import config as cfg
from . import utils

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_icd_phenotyping(
    icd_df: pd.DataFrame,
    enc_df: pd.DataFrame,
    column_map: Optional[Dict[str, str]] = None,
    enc_type_values: Optional[List[str]] = None,
    merge_encounters: bool = True,
    output_dir: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run ICD-based hemoglobinopathy phenotyping.
    
    Parameters
    ----------
    icd_df : pd.DataFrame
        DataFrame containing ICD codes with columns for person ID, ICD code, and date
    enc_df : pd.DataFrame
        DataFrame containing encounters with columns for person ID, type, start, end
    column_map : dict, optional
        Mapping of logical column names to actual column names in the data.
        Keys: 'id', 'encType', 'encStart', 'encEnd', 'icdCode', 'icdTime'
    enc_type_values : list, optional
        List of encounter type values in order: [OP, IP, ER, OB]
    merge_encounters : bool, default True
        If True, merge overlapping IP/ER/OB encounters. If False, simple cleaning only.
    output_dir : str, optional
        Directory to write intermediate CSV files for debugging
        
    Returns
    -------
    tuple of (eval_df, enc_icd_df, enc_clean_df)
        eval_df: Final phenotype assignments per person
        enc_icd_df: Encounters linked to ICD categories
        enc_clean_df: Cleaned encounter data
    """
    # Apply defaults
    col = {**cfg.DEFAULT_COLUMN_MAP, **(column_map or {})}
    enc_types = enc_type_values or cfg.DEFAULT_ENC_TYPE_VALUES
    [OP, IP, ER, OB] = enc_types
    
    # Encounter types that count toward phenotype (ER excluded - coding unreliable)
    phenotype_enc_types = cfg.get_phenotype_enc_types(enc_types)
    phenotype_enc_types_sql = ", ".join([f"'{t}'" for t in phenotype_enc_types])
    
    # Create DuckDB connection
    con = utils.get_con()
    
    # -------------------------------------------------------------------------
    # Step 1: Categorize ICD codes by disease group
    # -------------------------------------------------------------------------
    logger.info("Step 1: Categorizing ICD codes by disease group")
    
    # Build CASE statement from regex config
    case_clauses = []
    for category, regex in cfg.ICD_REGEX.items():
        case_clauses.append(f"WHEN regexp_matches(k.{col['icdCode']}, '{regex}') THEN '{category}'")
    case_statement = " ".join(case_clauses)
    
    sql_categorize = f"""
        SELECT DISTINCT
            k.{col['id']}
            , k.{col['icdTime']}
            , CASE {case_statement} END AS IcdCategory
        FROM icd_df k
        WHERE CASE {case_statement} END IS NOT NULL
        ORDER BY k.{col['id']}, k.{col['icdTime']}
    """
    icd_cat_df = con.execute(sql_categorize).df()
    
    logger.info(f"ICD codes categorized: {len(icd_df)} -> {len(icd_cat_df)} (by disease group)")
    
    # -------------------------------------------------------------------------
    # Step 2: Clean encounters
    # -------------------------------------------------------------------------
    logger.info("Step 2: Cleaning encounters")
    
    if merge_encounters:
        enc_clean_df = utils.clean_merge_encounters(
            con, enc_df, col['id'], col['encType'], col['encStart'], col['encEnd'],
            IP, ER, OB, OP
        )
    else:
        enc_clean_df = utils.clean_encounters(
            con, enc_df, col['id'], col['encType'], col['encStart'], col['encEnd'],
            IP, ER, OB, OP
        )
    
    logger.info(f"Encounters cleaned: {len(enc_df)} -> {len(enc_clean_df)}")
    
    # -------------------------------------------------------------------------
    # Step 3: Link ICD codes to encounters
    # -------------------------------------------------------------------------
    logger.info("Step 3: Linking ICD codes to encounters")
    
    sql_link = f"""
        SELECT DISTINCT
            k.{col['id']}
            , k.{col['encType']}
            , k.{col['encStart']}
            , k.{col['encEnd']}
            , MAX(k.ScdYN) AS ScdYN
            , MAX(k.ThalYN) AS ThalYN
            , MAX(k.TraitYN) AS TraitYN
            , MAX(k.OtherYN) AS OtherYN
        FROM (
            SELECT DISTINCT
                e.*
                , CASE WHEN i.IcdCategory = 'SCD' THEN 'Y' ELSE 'N' END AS ScdYN
                , CASE WHEN i.IcdCategory = 'THAL' THEN 'Y' ELSE 'N' END AS ThalYN
                , CASE WHEN i.IcdCategory = 'TRAIT' THEN 'Y' ELSE 'N' END AS TraitYN
                , CASE WHEN i.IcdCategory = 'OTHER' THEN 'Y' ELSE 'N' END AS OtherYN
            FROM enc_clean_df e
            LEFT JOIN icd_cat_df i ON i.{col['id']} = e.{col['id']} 
                AND (i.{col['icdTime']} BETWEEN e.{col['encStart']} AND e.{col['encEnd']} 
                     OR i.{col['icdTime']} = e.{col['encStart']})
        ) k
        GROUP BY k.{col['id']}, k.{col['encType']}, k.{col['encStart']}, k.{col['encEnd']}
        ORDER BY k.{col['id']}, k.{col['encStart']}, k.{col['encEnd']}
    """
    enc_icd_df = con.execute(sql_link).df()
    
    logger.info(f"ICD codes linked to {len(enc_icd_df)} encounter records")
    
    # -------------------------------------------------------------------------
    # Step 4: Final phenotype evaluation
    # -------------------------------------------------------------------------
    logger.info("Step 4: Computing final phenotype assignments")
    
    sql_eval = f"""
        SELECT DISTINCT
            CASE
                WHEN k.ThalCount > k.ScdCount AND k.ThalCount > k.TraitCount AND k.ThalCount > k.OtherCount THEN 'THAL'
                WHEN k.TraitCount > k.ScdCount AND k.TraitCount > k.ThalCount AND k.TraitCount > k.OtherCount THEN 'TRAIT'
                WHEN k.OtherCount > k.ScdCount AND k.OtherCount > k.ThalCount AND k.OtherCount > k.TraitCount THEN 'OTHER'
                WHEN k.ScdCount > k.ThalCount AND k.ScdCount > k.TraitCount AND k.ScdCount > k.OtherCount 
                     AND k.ScdCount > 2 AND k.ScdPercent > 5 THEN 'SCD'
                WHEN k.ScdCount > k.ThalCount AND k.ScdCount > k.TraitCount AND k.ScdCount > k.OtherCount 
                     AND k.ScdCount > 2 THEN 'SCDX'
                ELSE 'UNK'
            END AS IcdPheno
            , k.*
        FROM (
            SELECT DISTINCT
                g.*
                , CASE 
                    WHEN (g.IpCount + g.OpCount) > 0 
                    THEN CAST(g.ScdCount AS REAL) / CAST((g.IpCount + g.OpCount) AS REAL) * 100 
                    ELSE 0 
                  END AS ScdPercent
            FROM (
                SELECT DISTINCT
                    e.{col['id']}
                    , CAST(COUNT(*) AS INTEGER) AS EncCount
                    , CAST(COUNT(CASE WHEN e.{col['encType']} = '{IP}' THEN 1 END) AS INTEGER) AS IpCount
                    , CAST(COUNT(CASE WHEN e.{col['encType']} = '{ER}' THEN 1 END) AS INTEGER) AS ErCount
                    , CAST(COUNT(CASE WHEN e.{col['encType']} = '{OB}' THEN 1 END) AS INTEGER) AS ObCount
                    , CAST(COUNT(CASE WHEN e.{col['encType']} = '{OP}' THEN 1 END) AS INTEGER) AS OpCount
                    , CAST(COUNT(CASE WHEN e.ScdYN = 'Y' AND e.{col['encType']} IN ({phenotype_enc_types_sql}) THEN 1 END) AS INTEGER) AS ScdCount
                    , CAST(COUNT(CASE WHEN e.ThalYN = 'Y' AND e.{col['encType']} IN ({phenotype_enc_types_sql}) THEN 1 END) AS INTEGER) AS ThalCount
                    , CAST(COUNT(CASE WHEN e.TraitYN = 'Y' AND e.{col['encType']} IN ({phenotype_enc_types_sql}) THEN 1 END) AS INTEGER) AS TraitCount
                    , CAST(COUNT(CASE WHEN e.OtherYN = 'Y' AND e.{col['encType']} IN ({phenotype_enc_types_sql}) THEN 1 END) AS INTEGER) AS OtherCount
                FROM enc_icd_df e
                GROUP BY e.{col['id']}
            ) g
        ) k
    """
    eval_df = con.execute(sql_eval).df()
    eval_df = eval_df.sort_values(col['id'])
    
    logger.info(f"Phenotyping complete: {len(eval_df)} persons evaluated")
    
    # Log phenotype distribution
    if len(eval_df) > 0:
        pheno_counts = eval_df['IcdPheno'].value_counts()
        logger.info(f"Phenotype distribution:\n{pheno_counts.to_string()}")
    
    # -------------------------------------------------------------------------
    # Optional: Write intermediate files
    # -------------------------------------------------------------------------
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        date_str = datetime.today().strftime('%Y%m%d')
        
        enc_clean_df.to_csv(os.path.join(output_dir, f'enc_cleaned_{date_str}.csv'), index=False)
        enc_icd_df.to_csv(os.path.join(output_dir, f'enc_icd_linked_{date_str}.csv'), index=False)
        eval_df.to_csv(os.path.join(output_dir, f'phenotype_results_{date_str}.csv'), index=False)
        
        logger.info(f"Intermediate files written to {output_dir}")
    
    con.close()
    return eval_df, enc_icd_df, enc_clean_df


def get_icd_regex() -> Dict[str, str]:
    """Return the ICD regex patterns used for categorization."""
    return cfg.ICD_REGEX.copy()


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hemoglobinopathy ICD Phenotyping")
    parser.add_argument("--icd", required=True, help="Path to ICD CSV file")
    parser.add_argument("--enc", required=True, help="Path to Encounters CSV file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--no-merge", action="store_true", help="Disable encounter merging")
    
    args = parser.parse_args()
    
    icd_df = pd.read_csv(args.icd, parse_dates=True, low_memory=False)
    enc_df = pd.read_csv(args.enc, parse_dates=True, low_memory=False)
    
    eval_df, _, _ = run_icd_phenotyping(
        icd_df, enc_df,
        merge_encounters=not args.no_merge,
        output_dir=args.output
    )
    
    print(f"\nPhenotyping complete. Results written to {args.output}")
    print(f"\nPhenotype distribution:")
    print(eval_df['IcdPheno'].value_counts())


if __name__ == "__main__":
    main()

