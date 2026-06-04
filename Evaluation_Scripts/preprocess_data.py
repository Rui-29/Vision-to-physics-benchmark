import pandas as pd
import re

# --- 1. Define Mappings ---
ACTION_KEYS = [
    "Sit", "Stand", "Touch Face", "Arm/Hand Movements", "Social Gestures",
    "Leg/Foot Movements", "Bend", "Stretch", "Pick Up", "Step/Stomp",
    "T Pose", "Drink", "Balance", "Kneel", "Sneak", "Walk", "Tiptoe",
    "Lift", "Squat", "Crouch", "Dance", "Jump/Hop", "Exercise/Acrobatics",
    "Throw/Catch", "Crawl", "Run/Jog", "Punch/Kick/Martial Arts"
]

MET_VALUES = [
    1.0, 1.2, 1.2, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 
    1.4, 1.2, 1.4, 1.7, 1.7, 2.0, 1.7, 3.0, 2.7, 2.7, 
    3.4, 3.5, 3.5, 3.5, 3.5, 3.8, 7.85
]

CLOTHING_KEYS = [
    "Bra", "Panties", "Men's briefs", "T-shirt", "Long underwear bottoms",
    "Long underwear top", "Sleeveless/scoop-neck blouse", "Short-sleeve knit sport shirt",
    "Short-sleeve dress shirt", "Long-sleeve dress shirt", "Long-sleeve flannel shirt",
    "Long-sleeve sweatshirt", "Short shorts", "Walking shorts", "Straight trousers",
    "Sweatpants", "Overalls", "Skirt", "Sleeveless, scoop neck (dress)",
    "Short-sleeve shirtdress", "Long-sleeve shirtdress", "Sleeveless vest (sweater)",
    "Long-sleeve sweater", "Sleeveless vest (jacket)", "Single-breasted jacket",
    "Double-breasted jacket", "Sleeveless short gown", "Sleeveless long gown",
    "Short-sleeve hospital gown", "Short-sleeve short robe", "Short-sleeve pajamas",
    "Long-sleeve long gown", "Long-sleeve short wrap robe", "Long-sleeve pajamas",
    "Long-sleeve long wrap robe"
]

GT_FILENAME_TO_CANONICAL = {
    'foot_movement': 'Leg/Foot Movements', 'legs_movement': 'Leg/Foot Movements',
    'walk': 'Walk', 'bend': 'Bend', 'squat': 'Squat', 'jog': 'Run/Jog',
    'run': 'Run/Jog', 'lift_something': 'Lift', 'crouch': 'Crouch',
    'arm_movements': 'Arm/Hand Movements', 'write': 'Arm/Hand Movements',
    'stretch': 'Stretch', 'take_or_pick_something_up': 'Pick Up',
    'dance': 'Dance', 'bow': 'Social Gestures',
    'exercise_or_training': 'Exercise/Acrobatics', 'sit': 'Sit', 'stand_up': 'Stand'
}

# Create Lookup Dictionaries
ACTION_TO_ID = {name.lower(): i for i, name in enumerate(ACTION_KEYS)}
CLOTHING_TO_ID = {name.lower(): i for i, name in enumerate(CLOTHING_KEYS)}
ID_TO_MET = {i: met for i, met in enumerate(MET_VALUES)}

def extract_action_from_filename(filename):
    """Extracts the action string from filenames."""
    if not isinstance(filename, str): return None
    match = re.search(r"_\d+_([a-zA-Z_]+)_gemini", filename)
    return match.group(1) if match else None

def preprocess_csv(input_csv, output_csv):
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: {input_csv} not found.")
        return

    # Extract and map ground truth
    ground_truth_raw = df['filename'].apply(extract_action_from_filename)
    ground_truth_canonical = ground_truth_raw.map(GT_FILENAME_TO_CANONICAL)
    
    ground_truth_ids = ground_truth_canonical.str.lower().map(ACTION_TO_ID)
    ground_truth_met = ground_truth_ids.map(ID_TO_MET)

    df.insert(1, 'extracted_ground_truth', ground_truth_ids)
    df.insert(2, 'extracted_ground_truth_met', ground_truth_met)

    # Convert Predictions to IDs
    for col in df.columns:
        if "top_action_name" in col:
            df[col] = df[col].astype(str).str.lower().str.strip().map(ACTION_TO_ID)
        elif "top_clothing_item" in col:
            df[col] = df[col].astype(str).str.lower().str.strip().map(CLOTHING_TO_ID)

    df.to_csv(output_csv, index=False)
    print(f"Success! Preprocessed data saved to {output_csv}")

if __name__ == "__main__":
    preprocess_csv("raw_model_predictions.csv", "processed_numeric_predictions.csv")