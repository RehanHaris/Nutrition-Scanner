import os
import io
import re
from typing import List, Tuple, Dict

import pandas as pd
import streamlit as st
from rapidfuzz import process, fuzz
from fpdf import FPDF  # fpdf2

# ----------------------------- App config -----------------------------
APP_TITLE = "Smart Nutrition Scanner"
DB_FILENAME = "merged_food_db.csv"
CUSTOM_DB_FILENAME = "user_custom_db.csv"

# A compact demo database (kept for bootstrap when no CSV present)
DEFAULT_DATA_CSV = """name,synonyms,serving,calories_kcal,protein_g,carbs_g,fat_g,image_url
Chapati (Roti),"roti|phulka|chapathi",1 medium (40 g),120,3.7,18,3.7,unsplash:chapati roti flatbread
Paratha (Plain),"parantha|parotta",1 piece (70 g),260,5,34,10,unsplash:paratha
Chicken Biryani,"biryani|chicken biryani rice",1 cup (200 g),290,15,38,8,unsplash:biryani
Chicken Shawarma Wrap,"shawarma|shaurma|shwerma|shawarama",1 wrap (250 g),430,30,45,15,unsplash:shawarma
Beef Burger,"burger|cheeseburger|hamburger",1 sandwich,303,17,30,14,unsplash:burger
Egg (Chicken),"egg|boiled egg|fried egg",1 large (50 g),72,6.3,0.4,4.8,unsplash:egg
Grilled Chicken Breast,"chicken breast|bbq chicken",100 g,165,31,0,3.6,unsplash:grilled chicken
White Rice (Cooked),"rice|steamed rice|boiled rice",1 cup (158 g),205,4.3,45,0.4,unsplash:white rice
Banana (Medium),"banana fruit|kela",1 medium (118 g),105,1.3,27,0.3,unsplash:banana
Paneer (Fresh),"cottage cheese|paneer cubes",100 g,265,18,3.4,21,unsplash:paneer
"""

# ----------------------------- Helpers -----------------------------
@st.cache_data
def load_database(csv_bytes: bytes | None = None) -> pd.DataFrame:
    """Load base database. If csv_bytes provided, load from upload.
    Otherwise prefer local CSV, else bootstrap with DEFAULT_DATA_CSV."""
    if csv_bytes is not None:
        df = pd.read_csv(io.BytesIO(csv_bytes))
        return sanitize_df(df)

    # Try local DB, else create it from default
    if os.path.exists(DB_FILENAME):
        df = pd.read_csv(DB_FILENAME)
        return sanitize_df(df)
    else:
        with open(DB_FILENAME, "w", encoding="utf-8") as f:
            f.write(DEFAULT_DATA_CSV)
        df = pd.read_csv(io.StringIO(DEFAULT_DATA_CSV))
        return sanitize_df(df)


def sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    expected_cols = [
        "name",
        "synonyms",
        "serving",
        "calories_kcal",
        "protein_g",
        "carbs_g",
        "fat_g",
        "image_url",
    ]
    for c in expected_cols:
        if c not in df.columns:
            df[c] = "" if c in ["name", "synonyms", "serving", "image_url"] else 0.0
    df = df[expected_cols].copy()
    # Fill NA, ensure numeric
    for col in ["calories_kcal", "protein_g", "carbs_g", "fat_g"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    for col in ["name", "synonyms", "serving", "image_url"]:
        df[col] = df[col].fillna("")
    return df


def make_unsplash_url(term: str) -> str:
    term = term.replace(" ", ",")
    return f"https://source.unsplash.com/600x400/?{term}"


def pick_image_url(raw_url: str, fallback_term: str) -> str:
    if isinstance(raw_url, str) and raw_url:
        if raw_url.startswith("unsplash:"):
            term = raw_url.split("unsplash:", 1)[1].strip()
            return make_unsplash_url(term or fallback_term)
        return raw_url
    return make_unsplash_url(fallback_term)


def normalize(s: str) -> str:
    s = str(s or "").lower().strip()
    # lightweight normalization
    table = str.maketrans({",": " ", "-": " ", "_": " ", "/": " "})
    return " ".join(s.translate(table).split())


def build_search_index(df: pd.DataFrame) -> List[Tuple[str, str]]:
    """Return list of (search_key, canonical_name)."""
    keys = []
    for _, row in df.iterrows():
        name = str(row["name"]).strip()
        keys.append((normalize(name), name))
        syns = (
            str(row.get("synonyms", "")).split("|")
            if pd.notna(row.get("synonyms", ""))
            else []
        )
        for s in syns:
            s = s.strip()
            if s:
                keys.append((normalize(s), name))
    # de-duplicate
    keys = list(dict.fromkeys(keys))
    return keys


def fuzzy_top_matches(
    query: str, search_index: List[Tuple[str, str]], limit: int = 3
) -> List[Tuple[str, float]]:
    """Return top canonical names with best score."""
    if not query.strip():
        return []
    choices = [k for k, _ in search_index]
    results = process.extract(normalize(query), choices, scorer=fuzz.WRatio, limit=20)
    name_to_score: Dict[str, float] = {}
    for (matched_key, score, idx) in results:
        name = search_index[idx][1]
        name_to_score[name] = max(name_to_score.get(name, 0.0), float(score))
    ranked = sorted(name_to_score.items(), key=lambda x: x[1], reverse=True)
    return ranked[:limit]


def get_rows_by_names(df: pd.DataFrame, names: List[str]) -> pd.DataFrame:
    mask = df["name"].isin(names)
    return df[mask].set_index("name").loc[names].reset_index()


def printable_macros(row: pd.Series) -> str:
    return (
        f"Calories: {row['calories_kcal']:.0f} kcal  |  Protein: {row['protein_g']:.1f} g  |  "
        f"Carbs: {row['carbs_g']:.1f} g  |  Fat: {row['fat_g']:.1f} g"
    )


# ----------------------------- Grams parsing & scaling -----------------------------

def parse_serving_grams(serving: str) -> float | None:
    """Extract grams from strings like '1 medium (40 g)' or '100 g'."""
    if not serving:
        return None
    match = re.search(r"(\d+(?:\.\d+)?)\s*(g|grams)\b", serving, flags=re.I)
    if match:
        try:
            return float(match.group(1))
        except Exception:
            return None
    return None


def scale_macros_by_grams(row: pd.Series, grams: float) -> Dict[str, float]:
    """Scale macros for given grams. If serving grams present, scale by grams/serving_grams.
    Otherwise assume stored macros are per 100 g (fallback)."""
    serving = str(row.get("serving", ""))
    stored_kcal = float(row.get("calories_kcal", 0.0))
    stored_pro = float(row.get("protein_g", 0.0))
    stored_car = float(row.get("carbs_g", 0.0))
    stored_fat = float(row.get("fat_g", 0.0))

    serving_grams = parse_serving_grams(serving)
    factor = (grams / serving_grams) if (serving_grams and serving_grams > 0) else (grams / 100.0)
    return {
        "calories_kcal": stored_kcal * factor,
        "protein_g": stored_pro * factor,
        "carbs_g": stored_car * factor,
        "fat_g": stored_fat * factor,
    }


# ----------------------------- PDF & report helpers -----------------------------

def generate_day_dataframe(meals: Dict[str, List[dict]]) -> pd.DataFrame:
    rows = []
    for meal_name, items in meals.items():
        for it in items:
            rows.append(
                {
                    "meal": meal_name,
                    "name": it["name"],
                    "grams": it["grams"],
                    "calories_kcal": it["calories_kcal"],
                    "protein_g": it["protein_g"],
                    "carbs_g": it["carbs_g"],
                    "fat_g": it["fat_g"],
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["meal", "name", "grams", "calories_kcal", "protein_g", "carbs_g", "fat_g"]
        )
    return pd.DataFrame(rows)


def summarize_totals_from_df(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {"calories_kcal": 0.0, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0}
    return {
        "calories_kcal": float(df["calories_kcal"].sum()),
        "protein_g": float(df["protein_g"].sum()),
        "carbs_g": float(df["carbs_g"].sum()),
        "fat_g": float(df["fat_g"].sum()),
    }


def generate_pdf_report(meals: Dict[str, List[dict]], goals: Dict[str, float], note: str = "") -> bytes:
    df = generate_day_dataframe(meals)
    totals = summarize_totals_from_df(df)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)  # core font
    pdf.cell(0, 8, "Nutrition Report", ln=True)
    pdf.set_font("Helvetica", size=10)
    pdf.ln(2)
    pdf.multi_cell(0, 6, "Summary for the day - Totals vs Goals")
    pdf.ln(2)

    pdf.set_font("Helvetica", size=10)
    pdf.cell(60, 6, "Goal (cal/pro/car/fat):")
    pdf.cell(0, 6, f"{goals['calories']} kcal / {goals['protein']} g / {goals['carbs']} g / {goals['fat']} g", ln=True)
    pdf.cell(60, 6, "Actual (cal/pro/car/fat):")
    pdf.cell(0, 6, f"{totals['calories_kcal']:.0f} kcal / {totals['protein_g']:.1f} g / {totals['carbs_g']:.1f} g / {totals['fat_g']:.1f} g", ln=True)
    pdf.ln(6)

    # Per-meal listing
    for meal_name, items in meals.items():
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 7, meal_name, ln=True)

        pdf.set_font("Helvetica", size=10)
        if not items:
            pdf.cell(0, 6, "  (no items)", ln=True)
        else:
            for it in items:
                line = (
                    f"  - {it['name']} - {it['grams']} g - {it['calories_kcal']:.0f} kcal, "
                    f"P:{it['protein_g']:.1f}g C:{it['carbs_g']:.1f}g F:{it['fat_g']:.1f}g"
                )
                pdf.multi_cell(0, 6, line)
                pdf.ln(1)
        pdf.ln(3)

    if note:
        pdf.ln(4)
        pdf.set_font("Helvetica", style="I", size=10)
        pdf.multi_cell(0, 6, "AI note: " + note)

    return bytes(pdf.output(dest="S"))


# ----------------------------- AI-style summary (simple heuristic) -----------------------------

def ai_guess_summary(meals: Dict[str, List[dict]], totals: Dict[str, float], goals: Dict[str, float]) -> str:
    if totals["calories_kcal"] <= 0:
        return "No foods recorded yet - add items to see an AI-style summary."
    parts = []
    cal = totals["calories_kcal"]
    if cal < 0.8 * goals["calories"]:
        parts.append("Under calorie target - you might need more food today.")
    elif cal > 1.15 * goals["calories"]:
        parts.append("Over calorie target - today is calorie-heavy.")
    else:
        parts.append("Calories are close to target - good job staying on track.")

    prot = totals["protein_g"]
    if goals["protein"] > 0:
        prot_pct = prot / goals["protein"]
        if prot_pct < 0.7:
            parts.append("Protein is a bit low - consider adding lean protein.")
        elif prot_pct > 1.3:
            parts.append("Protein is high compared to your goal - great for muscle goals.")
        else:
            parts.append("Protein is well-balanced.")

    carb = totals["carbs_g"]
    fat = totals["fat_g"]
    parts.append(f"Carbs: {carb:.0f} g, Fat: {fat:.0f} g - adjust based on energy needs.")

    return " ".join(parts)


# ----------------------------- UI -----------------------------

st.set_page_config(page_title=APP_TITLE, page_icon="🍱", layout="wide")

# ✅ Initialize session state keys once
if "current_meal" not in st.session_state:
    st.session_state["current_meal"] = []
if "meals" not in st.session_state:
    st.session_state["meals"] = {"Breakfast": [], "Lunch": [], "Dinner": []}
if "goals" not in st.session_state:
    st.session_state["goals"] = {"calories": 2170, "protein": 163.0, "carbs": 217.0, "fat": 72.0}
if "favorites" not in st.session_state:
    st.session_state["favorites"] = set()

st.title(APP_TITLE)
st.caption("Search foods, scan by grams, build meals, assign to breakfast/lunch/dinner, and export reports.")

# Sidebar: Data + goals + actions
with st.sidebar:
    st.header("Data & Goals")
    uploaded_db = st.file_uploader("Upload custom database CSV", type=["csv"])
    df = load_database(uploaded_db.getvalue() if uploaded_db else None)

    # Merge user_custom_db if exists
    if os.path.exists(CUSTOM_DB_FILENAME):
        try:
            user_df = pd.read_csv(CUSTOM_DB_FILENAME)
            df = pd.concat([df, sanitize_df(user_df)], ignore_index=True)
        except Exception:
            st.warning("Couldn't read your custom DB; please re-upload or re-add items.")

    st.download_button(
        "⬇️ Download current DB as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="nutrition_db_export.csv",
    )

    st.markdown("---")
    st.subheader("Daily goals (change anytime)")
    g_cal = st.number_input(
        "Calories (kcal)", min_value=0, value=int(st.session_state["goals"]["calories"]), step=50, key="goal_calories"
    )
    g_pro = st.number_input(
        "Protein (g)", min_value=0.0, value=float(st.session_state["goals"]["protein"]), step=1.0, key="goal_protein"
    )
    g_carb = st.number_input(
        "Carbs (g)", min_value=0.0, value=float(st.session_state["goals"]["carbs"]), step=1.0, key="goal_carbs"
    )
    g_fat = st.number_input(
        "Fat (g)", min_value=0.0, value=float(st.session_state["goals"]["fat"]), step=1.0, key="goal_fat"
    )
    st.session_state["goals"] = {"calories": int(g_cal), "protein": float(g_pro), "carbs": float(g_carb), "fat": float(g_fat)}

    st.markdown("---")
    st.subheader("Quick exports")

    # Always-rendered export buttons (no extra click needed)
    df_day_sidebar = pd.DataFrame(columns=["meal", "name", "grams", "calories_kcal", "protein_g", "carbs_g", "fat_g"])  # placeholder
    ai_note_sidebar = ""
    if "meals" in st.session_state:
        df_day_sidebar = generate_day_dataframe(st.session_state["meals"])
        ai_note_sidebar = ai_guess_summary(
            st.session_state["meals"], summarize_totals_from_df(df_day_sidebar), st.session_state["goals"]
        )

    st.download_button(
        "📄 Download today's PDF report",
        data=generate_pdf_report(st.session_state["meals"], st.session_state["goals"], note=ai_note_sidebar),
        file_name="nutrition_report.pdf",
    )

    st.download_button(
        "🧾 Download today's CSV",
        data=df_day_sidebar.to_csv(index=False).encode("utf-8"),
        file_name="nutrition_day.csv",
    )

    if st.button("Clear today's meals and current meal"):
        st.session_state["current_meal"] = []
        st.session_state["meals"] = {"Breakfast": [], "Lunch": [], "Dinner": []}
        st.success("Cleared today's tracker.")

    st.markdown("---")
    st.write("Tips:")
    st.write(
        "• If a food serving shows grams (e.g. '100 g'), the app will scale using that.\n"
        "• If serving grams are missing, the app assumes values are per 100 g for grams scaling."
    )

# Build search index
search_index = build_search_index(df)

# Main layout: Left = scanner + meal builder, Right = daily tracker + summary
left, right = st.columns([2, 1])

with left:
    st.subheader("Food Scanner")
    query = st.text_input("🔎 Food name", placeholder="e.g., shawarma, chapati, chicken biryani...")
    qty_grams = st.number_input("Enter grams (g)", min_value=1.0, max_value=2000.0, value=100.0, step=5.0, key="scanner_grams")
    go = st.button("Scan")

    if go and not query.strip():
        st.warning("Type a food name to search.")
    elif go and query.strip():
        matches = fuzzy_top_matches(query, search_index, limit=5)
        if not matches:
            st.error("No matches found. Try a simpler word or add a custom item from the sidebar.")
        else:
            names = [m[0] for m in matches]
            top_df = get_rows_by_names(df, names)

            st.markdown("### Top matches - choose grams and add to meal")
            for i, (_, row) in enumerate(top_df.iterrows()):
                safe_name = re.sub(r"[^0-9a-zA-Z]+", "_", row["name"]) + f"_{i}"
                with st.expander(f"{row['name']} - {row['serving']}"):
                    img_url = pick_image_url(row["image_url"], row["name"])
                    st.image(img_url, use_container_width=True)
                    st.write(printable_macros(row))

                    serving_grams = parse_serving_grams(row.get("serving", ""))
                    if serving_grams:
                        st.caption(
                            f"Serving weight parsed: {serving_grams} g - scaling will use grams / {serving_grams}."
                        )
                    else:
                        st.caption(
                            "Serving grams not found for this item - scaling will assume per-100 g fallback."
                        )

                    grams_key = f"grams_{safe_name}"
                    grams_val = st.number_input(
                        "Grams (g)", min_value=1.0, max_value=5000.0, value=float(qty_grams), step=1.0, key=grams_key
                    )

                    scaled = scale_macros_by_grams(row, grams_val)
                    st.write(
                        f"Scaled: {scaled['calories_kcal']:.0f} kcal | P: {scaled['protein_g']:.1f} g | "
                        f"C: {scaled['carbs_g']:.1f} g | F: {scaled['fat_g']:.1f} g"
                    )

                    def add_current_meal(item: dict):
                        st.session_state["current_meal"] = st.session_state.get("current_meal", []) + [item]

                    def add_to_meal(meal_name: str, item: dict):
                        st.session_state["meals"][meal_name].append(item)
                        st.success(f"Added to {meal_name}!")

                    new_item = {
                        "name": row["name"],
                        "grams": float(grams_val),
                        "calories_kcal": float(scaled["calories_kcal"]),
                        "protein_g": float(scaled["protein_g"]),
                        "carbs_g": float(scaled["carbs_g"]),
                        "fat_g": float(scaled["fat_g"]),
                        "serving": row.get("serving", ""),
                    }

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.button("Add to Current Meal", key=f"add_current_{safe_name}", on_click=add_current_meal, args=(new_item,))
                    with c2:
                        st.button("Add → Breakfast", key=f"add_bf_{safe_name}", on_click=add_to_meal, args=("Breakfast", new_item))
                        st.button("Add → Lunch", key=f"add_lunch_{safe_name}", on_click=add_to_meal, args=("Lunch", new_item))
                    with c3:
                        st.button("Add → Dinner", key=f"add_dinner_{safe_name}", on_click=add_to_meal, args=("Dinner", new_item))

    # Current meal builder area
    st.markdown("---")
    st.subheader("Current Meal (build here)")
    if st.session_state["current_meal"]:
        df_current = pd.DataFrame(st.session_state["current_meal"])
        st.dataframe(df_current[["name", "grams", "calories_kcal", "protein_g", "carbs_g", "fat_g"]])
        totals = summarize_totals_from_df(df_current)
        st.write(
            f"Meal totals - {totals['calories_kcal']:.0f} kcal | P: {totals['protein_g']:.1f} g | "
            f"C: {totals['carbs_g']:.1f} g | F: {totals['fat_g']:.1f} g"
        )

        cola, colb, colc = st.columns(3)
        with cola:
            if st.button("Save Current Meal → Breakfast"):
                st.session_state["meals"]["Breakfast"].extend(st.session_state["current_meal"])
                st.session_state["current_meal"] = []
                st.success("Saved meal to Breakfast.")
        with colb:
            if st.button("Save Current Meal → Lunch"):
                st.session_state["meals"]["Lunch"].extend(st.session_state["current_meal"])
                st.session_state["current_meal"] = []
                st.success("Saved meal to Lunch.")
        with colc:
            if st.button("Save Current Meal → Dinner"):
                st.session_state["meals"]["Dinner"].extend(st.session_state["current_meal"])
                st.session_state["current_meal"] = []
                st.success("Saved meal to Dinner.")

        if st.button("Clear Current Meal"):
            st.session_state["current_meal"] = []
            st.info("Current meal cleared.")
    else:
        st.info("Current meal is empty - add items from search results above.")

with right:
    st.subheader("Daily Tracker")
    meals = st.session_state["meals"]
    day_df = generate_day_dataframe(meals)
    day_totals = summarize_totals_from_df(day_df)

    # Show per-meal details
    for meal_name in ["Breakfast", "Lunch", "Dinner"]:
        with st.expander(f"{meal_name} - {len(meals[meal_name])} item(s)"):
            if meals[meal_name]:
                df_m = pd.DataFrame(meals[meal_name])
                st.dataframe(df_m[["name", "grams", "calories_kcal", "protein_g", "carbs_g", "fat_g"]])
                tot = summarize_totals_from_df(df_m)
                st.write(
                    f"Totals: {tot['calories_kcal']:.0f} kcal | P: {tot['protein_g']:.1f} g | "
                    f"C: {tot['carbs_g']:.1f} g | F: {tot['fat_g']:.1f} g"
                )

                # Allow removing individual item
                for idx, r in list(enumerate(meals[meal_name])):
                    remove_key = f"rm_{meal_name}_{idx}_" + re.sub(r"[^0-9a-zA-Z]+", "_", r["name"]) 

                    def remove_from_meal(meal_name_local: str, idx_local: int):
                        try:
                            st.session_state["meals"][meal_name_local].pop(idx_local)
                        except IndexError:
                            pass

                    st.button(
                        f"Remove '{r['name']}' from {meal_name}",
                        key=remove_key,
                        on_click=remove_from_meal,
                        args=(meal_name, idx),
                    )
            else:
                st.write("(no items)")

    st.markdown("---")
    st.subheader("Day Summary")
    st.write(f"Total calories: **{day_totals['calories_kcal']:.0f} kcal**")
    st.write(
        f"Protein: **{day_totals['protein_g']:.1f} g**  •  Carbs: **{day_totals['carbs_g']:.1f} g**  •  Fat: **{day_totals['fat_g']:.1f} g**"
    )

    # Calorie calculator + wow factor
    goals = st.session_state["goals"]

    def score_against_goal(actual, target):
        if target <= 0:
            return 100.0 if actual <= 0 else 0.0
        diff = abs(actual - target) / target
        score = max(0.0, 100.0 * (1 - diff))
        return score

    calorie_score = score_against_goal(day_totals["calories_kcal"], goals["calories"])
    prot_score = score_against_goal(day_totals["protein_g"], goals["protein"])
    carb_score = score_against_goal(day_totals["carbs_g"], goals["carbs"])
    fat_score = score_against_goal(day_totals["fat_g"], goals["fat"])

    macro_avg = (prot_score + carb_score + fat_score) / 3.0
    final_score = int(round(0.6 * calorie_score + 0.4 * macro_avg))

    st.progress(final_score / 100.0)
    st.metric(
        "Day Wow Factor",
        f"{final_score}%",
        delta=f"Cal score: {int(calorie_score)}% | Macro avg: {int(macro_avg)}%",
    )

    st.markdown("---")
    st.subheader("AI-style Summary")
    ai_note = ai_guess_summary(meals, day_totals, goals)
    st.info(ai_note)

    st.markdown("---")
    st.subheader("Exports")
    csv_bytes = day_df.to_csv(index=False).encode("utf-8")
    pdf_bytes = generate_pdf_report(meals, goals, note=ai_note)

    st.download_button("Download day CSV", data=csv_bytes, file_name="nutrition_day.csv")
    st.download_button("Download day PDF", data=pdf_bytes, file_name="nutrition_report.pdf")

    st.markdown("---")
    st.caption(
        "Made with ❤ - try adding items, building a meal, then saving the meal to Breakfast/Lunch/Dinner. Export using the buttons above."
    )

# Footer notes
st.markdown("---")
st.write(
    "**Developer notes:** This is a single-file Streamlit app. If you want tabs or persistent user DB (login + history), we can extend this next."
)
