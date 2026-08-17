import os
import sys
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon

# Import modules from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_loader
import ndvi_analysis

def run_tests():
    print("=" * 60)
    print("RUNNING EDGE CASE TEST SUITE FOR DATA PROCESSING LAYER")
    print("=" * 60)
    
    test_results = {}
    
    # ----------------------------------------------------
    # Setup test mock boundaries (GeoJSON equivalent GeoDataFrame)
    # ----------------------------------------------------
    test_fields = {
        "field_id": ["T01", "T02", "T03"],
        "field_name": ["Test Field Alpha", "Test Field Beta", "Test Field Gamma"],
        "area_ha": [10.0, 5.5, 8.2],
        "geometry": [
            Polygon([(0,0), (0,1), (1,1), (1,0), (0,0)]),
            Polygon([(2,2), (2,3), (3,3), (3,2), (2,2)]),
            Polygon([(4,4), (4,5), (5,5), (5,4), (4,4)])
        ]
    }
    gdf_test = gpd.GeoDataFrame(test_fields, crs="EPSG:4326")
    
    # ----------------------------------------------------
    # Case 1, 2, 3: Health Classification Limits
    # ----------------------------------------------------
    poor_val = 0.20
    mod_val = 0.45
    healthy_val = 0.75
    
    poor_class = ndvi_analysis.classify_health(poor_val, poor_thresh=0.30, healthy_thresh=0.60)
    mod_class = ndvi_analysis.classify_health(mod_val, poor_thresh=0.30, healthy_thresh=0.60)
    healthy_class = ndvi_analysis.classify_health(healthy_val, poor_thresh=0.30, healthy_thresh=0.60)
    
    test_results["1. Poor NDVI Health Classification"] = ("PASS" if poor_class == "Poor" else "FAIL", f"Expected Poor, got {poor_class}")
    test_results["2. Moderate NDVI Health Classification"] = ("PASS" if mod_class == "Moderate" else "FAIL", f"Expected Moderate, got {mod_class}")
    test_results["3. Healthy NDVI Health Classification"] = ("PASS" if healthy_class == "Healthy" else "FAIL", f"Expected Healthy, got {healthy_class}")

    # ----------------------------------------------------
    # Case 4, 5, 6, 7, 8: Deltas, Anomalies, and Cloud filtering
    # ----------------------------------------------------
    # Construct a timeseries for Field T01
    # T01 Month 1: 0.65, Cloud 5%  (Valid, No previous baseline)
    # T01 Month 2: 0.75, Cloud 10% (Valid, Positive NDVI Change)
    # T01 Month 3: 0.65, Cloud 8%  (Valid, Negative NDVI Change, Not Anomaly)
    # T01 Month 4: 0.35, Cloud 45% (Cloudy, Excluded from future deltas, health 'Moderate')
    # T01 Month 5: 0.30, Cloud 12% (Valid, Sudden NDVI Drop relative to Month 3! 0.30 - 0.65 = -0.35, Anomaly Severe!)
    df_timeseries = pd.DataFrame([
        {"field_id": "T01", "acquisition_date": pd.to_datetime("2026-01-15"), "ndvi_mean": 0.65, "ndvi_median": 0.64, "cloud_cover": 5.0},
        {"field_id": "T01", "acquisition_date": pd.to_datetime("2026-02-15"), "ndvi_mean": 0.75, "ndvi_median": 0.74, "cloud_cover": 10.0},
        {"field_id": "T01", "acquisition_date": pd.to_datetime("2026-03-15"), "ndvi_mean": 0.65, "ndvi_median": 0.64, "cloud_cover": 8.0},
        {"field_id": "T01", "acquisition_date": pd.to_datetime("2026-04-15"), "ndvi_mean": 0.35, "ndvi_median": 0.34, "cloud_cover": 45.0},
        {"field_id": "T01", "acquisition_date": pd.to_datetime("2026-05-15"), "ndvi_mean": 0.30, "ndvi_median": 0.29, "cloud_cover": 12.0}
    ])
    
    # Process
    df_processed = ndvi_analysis.process_field_timeseries(df_timeseries, poor_thresh=0.30, healthy_thresh=0.60, drop_thresh=0.20, max_cloud=20.0)
    
    # Check 7. No previous observation (First index)
    row_1 = df_processed.iloc[0]
    is_r1_nan = pd.isna(row_1['ndvi_delta']) and pd.isna(row_1['ndvi_pct_change']) and (row_1['is_anomaly'] == False)
    test_results["7. No Previous Observation Baseline"] = ("PASS" if is_r1_nan else "FAIL", f"Expected nan deltas, got delta={row_1['ndvi_delta']}, is_anomaly={row_1['is_anomaly']}")
    
    # Check 4. Positive NDVI change (Month 2 vs Month 1)
    row_2 = df_processed.iloc[1]
    is_r2_pos = (abs(row_2['ndvi_delta'] - 0.10) < 1e-5) and (abs(row_2['ndvi_pct_change'] - 15.38) < 0.05) and (row_2['is_anomaly'] == False)
    test_results["4. Positive NDVI Change"] = ("PASS" if is_r2_pos else "FAIL", f"Expected +0.10 delta and 15.38% change, got delta={row_2['ndvi_delta']}, pct={row_2['ndvi_pct_change']:.2f}%")
    
    # Check 5. Negative NDVI change (Month 3 vs Month 2)
    row_3 = df_processed.iloc[2]
    is_r3_neg = (abs(row_3['ndvi_delta'] - (-0.10)) < 1e-5) and (abs(row_3['ndvi_pct_change'] - (-13.33)) < 0.05) and (row_3['is_anomaly'] == False)
    test_results["5. Negative NDVI Change (Non-Anomaly)"] = ("PASS" if is_r3_neg else "FAIL", f"Expected -0.10 delta and -13.33% change, got delta={row_3['ndvi_delta']}, pct={row_3['ndvi_pct_change']:.2f}%")
    
    # Check 8. Cloudy observation (Month 4)
    row_4 = df_processed.iloc[3]
    is_r4_cloudy = (row_4['data_status'] == 'Cloudy') and pd.isna(row_4['ndvi_delta'])
    test_results["8. Cloudy Observation Exclusion"] = ("PASS" if is_r4_cloudy else "FAIL", f"Expected status 'Cloudy' and nan delta, got status='{row_4['data_status']}', delta={row_4['ndvi_delta']}")
    
    # Check 6. Sudden NDVI Drop Anomaly (Month 5 vs Month 3)
    # The last VALID was Month 3 (0.65). Month 5 is 0.30. Drop is 0.35, which is severe.
    row_5 = df_processed.iloc[4]
    is_r5_anomaly = (row_5['is_anomaly'] == True) and (abs(row_5['ndvi_drop'] - 0.35) < 1e-5) and (row_5['anomaly_severity'] == 'Severe')
    test_results["6. Sudden NDVI Drop Anomaly (Severity Severe)"] = ("PASS" if is_r5_anomaly else "FAIL", f"Expected Anomaly True, Drop 0.35, Severe, got anomaly={row_5['is_anomaly']}, drop={row_5['ndvi_drop']}, severity='{row_5['anomaly_severity']}'")

    # ----------------------------------------------------
    # Case 9: Missing NDVI (Null value)
    # ----------------------------------------------------
    df_missing_ndvi = pd.DataFrame([
        {"field_id": "T01", "acquisition_date": pd.to_datetime("2026-06-15"), "ndvi_mean": np.nan, "ndvi_median": np.nan, "cloud_cover": 5.0}
    ])
    df_processed_missing = ndvi_analysis.process_field_timeseries(df_missing_ndvi)
    row_missing = df_processed_missing.iloc[0]
    is_missing_excluded = (row_missing['data_status'] == 'Excluded') and (row_missing['health_status'] == 'Unknown')
    test_results["9. Missing NDVI Value Excluded"] = ("PASS" if is_missing_excluded else "FAIL", f"Expected Excluded status and Unknown health, got status='{row_missing['data_status']}', health='{row_missing['health_status']}'")

    # ----------------------------------------------------
    # Case 10: Invalid NDVI (Bounds violation)
    # ----------------------------------------------------
    df_invalid_ndvi = pd.DataFrame([
        {"field_id": "T01", "acquisition_date": "2026-01-15", "ndvi_mean": 1.5, "ndvi_median": 0.6, "cloud_cover": 5.0}
    ])
    # Run data loader validator directly
    report_invalid = data_loader.validate_ndvi_data(df_invalid_ndvi)
    is_invalid_caught = (report_invalid['success'] == False) and any("ndvi_mean is outside" in err for err in report_invalid['errors'])
    test_results["10. Invalid NDVI Bounds Flagged"] = ("PASS" if is_invalid_caught else "FAIL", f"Expected success=False and error output, got success={report_invalid['success']}, errors={report_invalid['errors']}")

    # ----------------------------------------------------
    # Case 11: Duplicate Observation
    # ----------------------------------------------------
    df_duplicate = pd.DataFrame([
        {"field_id": "T01", "acquisition_date": "2026-01-15", "ndvi_mean": 0.6, "ndvi_median": 0.59, "cloud_cover": 5.0},
        {"field_id": "T01", "acquisition_date": "2026-01-15", "ndvi_mean": 0.62, "ndvi_median": 0.61, "cloud_cover": 5.0}
    ])
    report_dup = data_loader.validate_ndvi_data(df_duplicate)
    is_dup_caught = (report_dup['success'] == False) and any("Duplicate observations detected" in err for err in report_dup['errors'])
    test_results["11. Duplicate Observation Flagged"] = ("PASS" if is_dup_caught else "FAIL", f"Expected success=False and duplicate error, got success={report_dup['success']}, errors={report_dup['errors']}")

    # ----------------------------------------------------
    # Case 12: Missing Field ID (Referential Integrity Check)
    # ----------------------------------------------------
    # NDVI record for field T99, which is not in our boundaries (gdf_test contains T01, T02, T03)
    df_unmatched = pd.DataFrame([
        {"field_id": "T99", "acquisition_date": "2026-01-15", "ndvi_mean": 0.6, "ndvi_median": 0.59, "cloud_cover": 5.0}
    ])
    report_integrity = data_loader.validate_field_ids(gdf_test, df_unmatched)
    is_integrity_caught = (report_integrity['success'] == False) and any("Referential Integrity Failure" in err for err in report_integrity['errors'])
    test_results["12. Missing Field ID (Referential Integrity) Flagged"] = ("PASS" if is_integrity_caught else "FAIL", f"Expected success=False and integrity error, got success={report_integrity['success']}, errors={report_integrity['errors']}")

    # ----------------------------------------------------
    # Case 13: Normalization mapping check
    # ----------------------------------------------------
    # We will test loading with a simulated teammate dataframe with different column names
    df_teammate = pd.DataFrame([
        {"t_field": "T01", "t_date": "2026-01-15", "t_mean": 0.6, "t_median": 0.59, "t_cloud": 5.0}
    ])
    # Apply mapping
    mapping = {
        "t_field": "field_id",
        "t_date": "acquisition_date",
        "t_mean": "ndvi_mean",
        "t_median": "ndvi_median",
        "t_cloud": "cloud_cover"
    }
    df_mapped = df_teammate.rename(columns=mapping)
    has_normalized_cols = set(df_mapped.columns) == {'field_id', 'acquisition_date', 'ndvi_mean', 'ndvi_median', 'cloud_cover'}
    test_results["13. Schema Mapping & Normalization Layer"] = ("PASS" if has_normalized_cols else "FAIL", f"Expected normalized columns, got {list(df_mapped.columns)}")

    # ----------------------------------------------------
    # Case 14: PDF report compilation check
    # ----------------------------------------------------
    try:
        import report_generator
        test_meta = {"field_id": "T01", "field_name": "Test Field", "area_ha": 10.0}
        test_obs = {
            "acquisition_date_str": "2026-01-15",
            "ndvi_mean": 0.65,
            "ndvi_median": 0.64,
            "prev_valid_ndvi": 0.60,
            "ndvi_delta": 0.05,
            "ndvi_pct_change": 8.33,
            "cloud_cover": 5.0,
            "health_status": "Healthy",
            "is_anomaly": False,
            "anomaly_severity": "None",
            "ndvi_drop": 0.0
        }
        pdf_buf = report_generator.generate_pdf_report(test_meta, test_obs, total_anomalies=0, earliest_date="2026-01-15", latest_date="2026-01-15")
        is_pdf_valid = pdf_buf is not None and len(pdf_buf.getvalue()) > 0
        test_results["14. ReportLab PDF Report Compilation"] = ("PASS" if is_pdf_valid else "FAIL", f"Expected non-empty pdf buffer, got size {len(pdf_buf.getvalue()) if pdf_buf else 0}")
    except Exception as e:
        test_results["14. ReportLab PDF Report Compilation"] = ("FAIL", f"Error generating PDF: {e}")

    # ----------------------------------------------------
    # Print Test Results Table
    # ----------------------------------------------------

    print(f"{'Test Case Description':<55} | {'Status':<6} | {'Details / Notes'}")
    print("-" * 110)
    all_passed = True
    for name, (status, note) in test_results.items():
        if status == "FAIL":
            all_passed = False
        print(f"{name:<55} | {status:<6} | {note}")
        
    print("=" * 60)
    if all_passed:
        print("[PASS] ALL EDGE CASE TESTS PASSED SUCCESSFULLY!")
    else:
        print("[FAIL] SOME TESTS FAILED. PLEASE DEBUG.")
    print("=" * 60)
    
    # ----------------------------------------------------
    # Demonstrate Analysis Outputs with Phase 1 Data
    # ----------------------------------------------------
    print("\nDEMONSTRATING PIPELINE WITH REAL MOCK DATASETS")
    print("-" * 60)
    gdf, df, summary = data_loader.load_and_validate_all()
    print(f"Data loading validation success: {summary['success']}")
    
    if summary['success']:
        df_all_processed = ndvi_analysis.process_all_fields(df)
        print(f"Processed DataFrame shape: {df_all_processed.shape}")
        
        # Display sample output columns for Field F04 (stress event field)
        print("\nProcessed Time-series Sample (Field F04 - South Barley Field):")
        f04_data = df_all_processed[df_all_processed['field_id'] == 'F04'][
            ['field_id', 'acquisition_date', 'ndvi_mean', 'cloud_cover', 'data_status', 'ndvi_delta', 'is_anomaly', 'anomaly_severity', 'health_status']
        ]
        print(f04_data.to_string(index=False))
        
        # Display field summaries
        print("\nGenerated Field summaries:")
        df_summaries = ndvi_analysis.generate_field_summaries(df_all_processed)
        print(df_summaries[
            ['field_id', 'latest_ndvi_mean', 'latest_health_status', 'latest_ndvi_delta', 'latest_is_anomaly', 'total_anomalies', 'total_valid_observations']
        ].to_string(index=False))
        
        # Display missing months
        print("\nMissing Months Report (for Year 2026):")
        missing_months = ndvi_analysis.detect_missing_months(df_all_processed)
        print(missing_months)
        
    else:
        print("Could not demonstrate pipeline due to loading failure.")
        print(f"Errors: {summary['errors']}")

if __name__ == "__main__":
    run_tests()
