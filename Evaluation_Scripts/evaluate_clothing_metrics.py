import pandas as pd
import numpy as np

def calculate_iou(gt_items, pred_items):
    """Calculates Intersection over Union (IoU) for two sets of items."""
    set_gt = set(gt_items)
    set_pred = set(pred_items)

    if len(set_gt) == 0 and len(set_pred) == 0:
        return 1.0  
        
    intersection = set_gt.intersection(set_pred)
    union = set_gt.union(set_pred)

    if len(union) == 0: return 0.0
    return len(intersection) / len(union)

def calculate_clothing_metrics(input_csv, models):
    df = pd.read_csv(input_csv)
    results = []
    
    gt_col_name = 'extracted_ground_truth_clothing'
    gt_clo_col_name = 'extracted_ground_truth_clo'
    
    if gt_col_name not in df.columns:
        print("Error: Ground truth clothing column not found.")
        return

    for model in models:
        pred_col = f"{model}_all_clothing_items"
        pred_clo_col = f"{model}_total_clo"

        if pred_col not in df.columns:
            continue

        ious = []
        clo_diffs = []

        for _, row in df.iterrows():
            # Calculate IoU
            gt_str = str(row[gt_col_name])
            pred_str = str(row[pred_col])

            gt_list = [x.strip() for x in gt_str.split(';') if x.strip()] if pd.notna(row[gt_col_name]) else []
            pred_list = [x.strip() for x in pred_str.split(';') if x.strip()] if pd.notna(row[pred_col]) else []

            ious.append(calculate_iou(gt_list, pred_list))

            # Calculate CLO Error
            try:
                gt_clo = float(row[gt_clo_col_name])
                pred_clo = float(row[pred_clo_col])
                clo_diffs.append(pred_clo - gt_clo)
            except (ValueError, KeyError):
                pass 

        mean_iou = np.mean(ious) * 100
        clo_mae = np.mean(np.abs(clo_diffs))
        clo_rmse = np.sqrt(np.mean(np.array(clo_diffs) ** 2))

        results.append({
            'Model': model,
            'Intersection over Union (IoU %)': round(mean_iou, 2),
            'CLO MAE': round(clo_mae, 3),
            'CLO RMSE': round(clo_rmse, 3)
        })

    return pd.DataFrame(results)

if __name__ == "__main__":
    target_models = ['gemini3pro', 'gemini_3flash', 'slowfast']
    df_results = calculate_clothing_metrics("processed_numeric_predictions.csv", target_models)
    print("\n--- Clothing Insulation (CLO) Metrics ---")
    print(df_results.to_string(index=False))