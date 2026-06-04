import numpy as np
import pandas as pd

def calculate_activity_metrics(input_csv, models):
    df = pd.read_csv(input_csv)
    results = []

    for model in models:
        col_names = f"{model}_all_action_names"
        col_met = f"{model}_top_action_met"

        # Filter valid rows
        temp_df = df[['extracted_ground_truth', 'extracted_ground_truth_met', col_names, col_met]].dropna()
        total = len(temp_df)

        if total == 0:
            continue

        top1_correct = 0
        top5_correct = 0

        for _, row in temp_df.iterrows():
            gt = int(row['extracted_ground_truth'])
            preds_str = str(row[col_names])
            preds = [int(x) for x in preds_str.split(';') if x.strip().isdigit()]

            # Top-1 Accuracy
            if len(preds) > 0 and preds[0] == gt:
                top1_correct += 1

            # Top-5 Accuracy
            if gt in preds[:5]:
                top5_correct += 1

        # Calculate Final Metrics
        top1_acc = (top1_correct / total) * 100
        top5_acc = (top5_correct / total) * 100
        met_mae = np.mean(np.abs(temp_df[col_met] - temp_df['extracted_ground_truth_met']))
        met_rmse = np.sqrt(np.mean((temp_df[col_met] - temp_df['extracted_ground_truth_met']) ** 2))

        results.append({
            'Model': model,
            'Top-1 Accuracy (%)': round(top1_acc, 2),
            'Top-5 Accuracy (%)': round(top5_acc, 2),
            'MET MAE': round(met_mae, 3),
            'MET RMSE': round(met_rmse, 3)
        })

    return pd.DataFrame(results)

if __name__ == "__main__":
    target_models = ['gemini3pro', 'gemini_3flash', 'slowfast']
    df_results = calculate_activity_metrics('processed_numeric_predictions.csv', target_models)
    print("\n--- Activity & Metabolic Rate (MET) Metrics ---")
    print(df_results.to_string(index=False))