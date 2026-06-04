import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

try:
    client = genai.Client()
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    exit()

# Use Flash for speed/cost, or Pro for accuracy
GEMINI_MODEL = "gemini-3-pro-preview"

# --- METADATA DICTIONARIES ---
ACTION_TO_MET = {
    # 1.0 MET
    "Sit": 1.0,
    # 1.2 MET
    "Stand": 1.2,
    # 1.4 MET
    "Arm/Hand Movements": 1.4,
    "Social Gestures": 1.4,
    "Leg/Foot Movements": 1.4,
    "Bend": 1.4,
    "Stretch": 1.4,
    "Pick Up": 1.4,
    "Step/Stomp": 1.4,
    "Drink": 1.4,
    # 2.0 MET
    "Walk": 2.0,
    # 2.1 MET
    "Lift": 2.1,
    # 2.7 MET
    "Squat": 2.7,
    "Crouch": 2.7,
    # 3.4 MET
    "Dance": 3.4,
    # 3.5 MET
    "Jump/Hop": 3.5,
    "Exercise/Acrobatics": 3.5,
    # 3.8 MET
    "Run/Jog": 3.8,
}

CLOTHING_TO_CLO = {
    # --- Underwear ---
    "Bra": 0.01,
    "Panties": 0.03,
    "Men's briefs": 0.04,
    "T-shirt": 0.08,
    "Long underwear bottoms": 0.15,
    "Long underwear top": 0.20,
    # --- Shirts and Blouses ---
    "Sleeveless/scoop-neck blouse": 0.12,
    "Short-sleeve knit sport shirt": 0.17,
    "Short-sleeve dress shirt": 0.19,
    "Long-sleeve dress shirt": 0.25,
    "Long-sleeve flannel shirt": 0.34,
    "Long-sleeve sweatshirt": 0.34,
    # --- Trousers and Coveralls ---
    "Short shorts": 0.06,
    "Walking shorts": 0.08,
    "Straight trousers": 0.20,
    "Sweatpants": 0.28,
    "Overalls": 0.30,
    # --- Dress and Skirts ---
    "Skirt": 0.19,
    "Sleeveless, scoop neck (dress)": 0.25,
    "Short-sleeve shirtdress": 0.29,
    "Long-sleeve shirtdress": 0.40,
    # --- Sweaters ---
    "Sleeveless vest (sweater)": 0.18,
    "Long-sleeve sweater": 0.31,
    # --- Suit Jackets and Vests ---
    "Sleeveless vest (jacket)": 0.14,
    "Single-breasted jacket": 0.40,
    "Double-breasted jacket": 0.45,
    # --- Sleepwear and Robes ---
    "Sleeveless short gown": 0.18,
    "Sleeveless long gown": 0.20,
    "Short-sleeve hospital gown": 0.31,
    "Short-sleeve short robe": 0.34,
    "Short-sleeve pajamas": 0.42,
    "Long-sleeve long gown": 0.46,
    "Long-sleeve short wrap robe": 0.48,
    "Long-sleeve pajamas": 0.57,
    "Long-sleeve long wrap robe": 0.69
}

# --- PROMPT CONSTRUCTION ---
action_list_str = ", ".join([f'{k} ({v})' for k, v in ACTION_TO_MET.items()])
clothing_list_str = ", ".join([f'{k} ({v})' for k, v in CLOTHING_TO_CLO.items()])

SYSTEM_PROMPT = (
    "You are a deterministic annotation assistant. "
    "Respond ONLY with valid JSON, no explanations."
)

USER_TASK_TEXT = f"""The following video shows one person performing an activity.

TASK 1 – Activity (Metabolic Rate)
Use the provided list of {len(ACTION_TO_MET)} coarse actions with their specific MET values:
[{action_list_str}]

1. Select EXACTLY 5 actions from the list above that best describe the activity.
2. For each action, provide:
   - "prob": probability ∈ [0,1]
   - "met": The EXACT MET value provided in parentheses above (do not guess).
3. Compute confidence-weighted MET = Σ(prob × met).

TASK 2 – Clothing (Clothing Insulation)
Use the provided list of {len(CLOTHING_TO_CLO)} clothing items with their specific CLO values:
[{clothing_list_str}]

1. Identify all visible items from the list above.
2. For each item, provide:
   - "conf": confidence ∈ [0,1]
   - "clo": The EXACT CLO value provided in parentheses above.
3. Compute total_clo = Σclo.

Return ONLY a JSON object."""

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "actions": {
            "type": "array",
            "items": {"type": "object", "properties": {
                "name": {"type": "string"},
                "prob": {"type": "number"},
                "met": {"type": "number"}
            }, "required": ["name", "prob", "met"]}
        },
        "confidence_weighted_met": {"type": "number"},
        "clothing": {
            "type": "array",
            "items": {"type": "object", "properties": {
                "item": {"type": "string"},
                "conf": {"type": "number"},
                "clo": {"type": "number"}
            }, "required": ["item", "conf", "clo"]}
        },
        "total_clo": {"type": "number"}
    },
    "required": ["actions", "confidence_weighted_met", "clothing", "total_clo"]
}

# --- HELPER FUNCTIONS ---
def annotate_video(video_path: Path):
    start_time = time.time()

    print(f"Reading {video_path.name}...")
    with open(video_path, "rb") as f:
        video_bytes = f.read()

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=JSON_SCHEMA,
        system_instruction=SYSTEM_PROMPT,
        temperature=0.0,
        seed=7,
    )

    print("Generating annotation with Inline Data...")
    try:
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(
                            data=video_bytes,
                            mime_type="video/mp4"
                        ),
                        types.Part.from_text(text=USER_TASK_TEXT),
                    ],
                ),
            ],
            config=config,
        )
    except Exception as e:
        print(f"API Error: {e}")
        raise

    elapsed = time.time() - start_time
    msg = resp.text
    print(f"Time spent: {elapsed:.2f} seconds")
    return msg