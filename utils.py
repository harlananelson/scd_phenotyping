"""
Utility Functions for Phenotyping
=================================

Encounter cleaning, merging, and DuckDB connection management.
"""

import logging
from typing import Optional

import pandas as pd
import duckdb

logger = logging.getLogger(__name__)


def get_con() -> duckdb.DuckDBPyConnection:
    """Return a fresh in-memory DuckDB connection."""
    return duckdb.connect()


def clean_encounters(
    con: duckdb.DuckDBPyConnection,
    enc_df: pd.DataFrame,
    id_col: str,
    enc_type: str,
    enc_start: str,
    enc_end: str,
    IP: str, ER: str, OB: str, OP: str
) -> pd.DataFrame:
    """
    Simple encounter cleaning: deduplicate and remove OP encounters during IP/ER/OB.
    
    Parameters
    ----------
    con : DuckDBPyConnection
        Active DuckDB connection
    enc_df : pd.DataFrame
        Raw encounter data
    id_col, enc_type, enc_start, enc_end : str
        Column names
    IP, ER, OB, OP : str
        Encounter type values
        
    Returns
    -------
    pd.DataFrame
        Cleaned encounters
    """
    # Extract relevant columns and deduplicate
    enc_df = enc_df[[id_col, enc_type, enc_start, enc_end]].drop_duplicates()
    
    # Multi-day encounters (IP, ER, OB)
    sql_multi = f"""
        SELECT DISTINCT e.*
        FROM enc_df e
        WHERE e.{enc_type} IN ('{IP}', '{ER}', '{OB}')
            AND e.{enc_end} IS NOT NULL
        ORDER BY e.{id_col}, e.{enc_start}
    """
    enc_multi = con.execute(sql_multi).df()
    
    # Filter out stays > 365 days
    sql_days = f"""
        WITH enc_with_days AS (
            SELECT *,
                CASE 
                    WHEN TRY_CAST({enc_start} AS DATE) IS NOT NULL 
                    THEN date_diff('day', CAST({enc_start} AS DATE), CAST({enc_end} AS DATE))
                    ELSE (CAST({enc_end} AS DECIMAL) - CAST({enc_start} AS DECIMAL)) * 365.25
                END AS enc_days
            FROM enc_multi
        )
        SELECT {id_col}, {enc_type}, {enc_start}, {enc_end}
        FROM enc_with_days
        WHERE enc_days < 365
    """
    enc_multi = con.execute(sql_days).df()
    
    long_stays = len(enc_df[(enc_df[enc_type].isin([IP, ER, OB]))]) - len(enc_multi)
    if long_stays > 0:
        logger.info(f"Removed {long_stays} encounters > 365 days")
    
    # Outpatient encounters (set end to NULL per convention)
    sql_op = f"""
        SELECT DISTINCT {id_col}, {enc_type}, {enc_start}, NULL AS {enc_end}
        FROM enc_df
        WHERE {enc_type} = '{OP}'
    """
    enc_op = con.execute(sql_op).df()
    
    # Remove OP encounters that overlap with multi-day stays
    sql_clean = f"""
        SELECT * FROM enc_multi
        UNION
        SELECT o.*
        FROM enc_op o
        LEFT JOIN enc_multi m ON m.{id_col} = o.{id_col}
            AND o.{enc_start} BETWEEN m.{enc_start} AND m.{enc_end}
        WHERE m.{id_col} IS NULL
        ORDER BY {id_col}, {enc_start}
    """
    return con.execute(sql_clean).df()


def clean_merge_encounters(
    con: duckdb.DuckDBPyConnection,
    enc_df: pd.DataFrame,
    id_col: str,
    enc_type: str,
    enc_start: str,
    enc_end: str,
    IP: str, ER: str, OB: str, OP: str
) -> pd.DataFrame:
    """
    Clean encounters and merge overlapping IP/ER/OB encounters.
    
    Encounter type hierarchy for merged encounters: IP > ER > OB
    
    Parameters
    ----------
    con : DuckDBPyConnection
        Active DuckDB connection
    enc_df : pd.DataFrame
        Raw encounter data
    id_col, enc_type, enc_start, enc_end : str
        Column names
    IP, ER, OB, OP : str
        Encounter type values
        
    Returns
    -------
    pd.DataFrame
        Cleaned and merged encounters
    """
    # First do simple cleaning
    enc_clean = clean_encounters(con, enc_df, id_col, enc_type, enc_start, enc_end, IP, ER, OB, OP)
    
    # Separate multi-day and OP
    sql_multi = f"""
        SELECT * FROM enc_clean WHERE {enc_type} IN ('{IP}', '{ER}', '{OB}')
    """
    enc_multi = con.execute(sql_multi).df()
    
    sql_op = f"""
        SELECT * FROM enc_clean WHERE {enc_type} = '{OP}'
    """
    enc_op = con.execute(sql_op).df()
    
    if len(enc_multi) == 0:
        return enc_clean
    
    # Iteratively merge overlapping encounters
    enc_loop = enc_multi.copy()
    change_count = 1
    iteration = 0
    max_iterations = 10  # Safety limit
    
    while change_count > 0 and iteration < max_iterations:
        iteration += 1
        start_count = len(enc_loop)
        logger.debug(f"Merge iteration {iteration}: {start_count} multi-day encounters")
        
        # Merge overlapping encounters
        sql_merge = f"""
            WITH extended AS (
                SELECT DISTINCT
                    e1.{id_col},
                    e1.{enc_start},
                    COALESCE(MAX(e2.{enc_end}), e1.{enc_end}) AS {enc_end}
                FROM enc_loop e1
                LEFT JOIN enc_loop e2 ON e1.{id_col} = e2.{id_col}
                    AND e2.{enc_start} BETWEEN e1.{enc_start} AND e1.{enc_end}
                    AND e2.{enc_end} > e1.{enc_end}
                GROUP BY e1.{id_col}, e1.{enc_start}, e1.{enc_end}
            ),
            aggregated AS (
                SELECT {id_col}, MIN({enc_start}) AS {enc_start}, {enc_end}
                FROM (
                    SELECT {id_col}, {enc_start}, MAX({enc_end}) AS {enc_end}
                    FROM extended
                    GROUP BY {id_col}, {enc_start}
                )
                GROUP BY {id_col}, {enc_end}
            ),
            no_eclipse AS (
                SELECT a.* FROM aggregated a
                WHERE NOT EXISTS (
                    SELECT 1 FROM aggregated b
                    WHERE b.{id_col} = a.{id_col}
                        AND a.{enc_start} > b.{enc_start}
                        AND a.{enc_end} < b.{enc_end}
                )
            )
            SELECT * FROM no_eclipse ORDER BY {id_col}, {enc_start}
        """
        merged = con.execute(sql_merge).df()
        
        # Assign encounter type by hierarchy: IP > ER > OB
        sql_type = f"""
            SELECT DISTINCT
                m.{id_col},
                CASE 
                    WHEN MAX(CASE WHEN e.{enc_type} = '{IP}' THEN 1 ELSE 0 END) = 1 THEN '{IP}'
                    WHEN MAX(CASE WHEN e.{enc_type} = '{ER}' THEN 1 ELSE 0 END) = 1 THEN '{ER}'
                    ELSE '{OB}'
                END AS {enc_type},
                m.{enc_start},
                m.{enc_end}
            FROM merged m
            LEFT JOIN enc_multi e ON e.{id_col} = m.{id_col}
                AND e.{enc_start} BETWEEN m.{enc_start} AND m.{enc_end}
            GROUP BY m.{id_col}, m.{enc_start}, m.{enc_end}
        """
        enc_loop = con.execute(sql_type).df()
        
        change_count = start_count - len(enc_loop)
    
    if iteration >= max_iterations:
        logger.warning(f"Encounter merging hit iteration limit ({max_iterations})")
    
    # Combine with OP (excluding those during merged stays)
    sql_final = f"""
        SELECT * FROM enc_loop
        UNION
        SELECT o.{id_col}, o.{enc_type}, o.{enc_start}, NULL AS {enc_end}
        FROM enc_op o
        LEFT JOIN enc_loop m ON m.{id_col} = o.{id_col}
            AND o.{enc_start} BETWEEN m.{enc_start} AND m.{enc_end}
        WHERE m.{id_col} IS NULL
        ORDER BY {id_col}, {enc_start}
    """
    return con.execute(sql_final).df()

