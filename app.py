import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os

# --- データ取得 ---
@st.cache_data(ttl=600)
def load_data():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    if os.path.exists("credentials.json"):
        creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
    elif "gcp_service_account" in st.secrets and st.secrets["gcp_service_account"].get("private_key"):
        creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["gcp_service_account"], scope)
    else:
        st.error("認証情報がありません。Cloudはsecrets.toml、ローカルはcredentials.jsonが必要です。")
        st.stop()
    client = gspread.authorize(creds)
    spreadsheet = client.open_by_key("1ISs5mqSRdZfF3NVOt60VFtY8p8HsM0ZkM3sfu3cPzVE")  # ←差し替えて！

    student_df = pd.DataFrame(spreadsheet.worksheet("スクール生情報").get_all_records())
    mentor_df = pd.DataFrame(spreadsheet.worksheet("メンター情報").get_all_records())
    game_df = pd.DataFrame(spreadsheet.worksheet("ゲーム一覧").get_all_values())
    return student_df, mentor_df, game_df

# --- ゲーム正規名マッピング ---
def get_game_word_map(game_df):
    word_to_canonical = {}
    for i, row in game_df.iterrows():
        canonical = row[0].strip()
        aliases = [canonical]
        if len(row) > 1 and row[1]:
            aliases += [w.strip() for w in row[1].split(",")]
        for w in aliases:
            if w:
                word_to_canonical[w] = canonical
    game_list_words = list(word_to_canonical.keys())
    return word_to_canonical, game_list_words

# --- 時間帯足切り判定（曜日・時間カラム名設計前提） ---
def is_time_slot_match(student, mentor):
    possible_slots = []
    for col in student.index:
        if "定期的" in col and "[" in col and "]" in col:
            value = student[col]
            if isinstance(value, str) and value.strip():
                days = [d.strip() for d in value.split(",")]
                try:
                    hour = col.split("[")[-1].split("〜")[0].replace("：", ":").strip()
                    hour = hour.replace(":", "")  # "17:00" → "1700"
                    for day in days:
                        if day in ["月", "火", "水", "木", "金", "土", "日"]:
                            slot = f"1on1可能時間_{day}_{hour}-"
                            possible_slots.append(slot)
                except:
                    continue
    # メンターの可能時間カラムに1つでも"TRUE"があれば一致とみなす
    return any(str(mentor.get(slot, "")).strip().lower() == "true" for slot in possible_slots)

# --- マッチングスコア計算 ---
def calculate_matching_score(student, mentor, game_list_words, word_to_canonical):
    # === 必須足切り ===
    # (1) 時間帯
    if not is_time_slot_match(student, mentor):
        return 0, "時間帯が一致しない"

    # (2) 担当枠
    if int(mentor.get("追加可能人数", 0)) < 1:
        return 0, "担当枠が空いていない"

    # (3) 性別希望の足切り・加点
    mentor_gender = mentor.get("属性_性別", "").strip()
    student_gender = student.get("お子さまの性別", "").strip()
    student_gender_pref = student.get("メンターの性別のご希望", "").strip()
    score = 0
    reasons = []

    if student_gender_pref and student_gender_pref not in ["指定なし", "", None]:
        # 希望があれば一致しなければ除外、一致しても加点はしない
        if student_gender_pref != mentor_gender:
            return 0, "性別希望に一致しない"
        # 一致しても加点なし
    elif student_gender and student_gender == mentor_gender:
        # 性別希望未指定なら「本人性別=メンター性別」で+10点
        score += 10
        reasons.append("性別一致（本人と同じ）")

    # === ゲームマッチ加点 ===
    student_text = (
        str(student.get("お子さまの得意なこと、好きなことを教えてください", ""))
        + str(student.get("興味がある分野をお答えください", ""))
        + str(student.get("お子さまがSOZOWスクールに期待していること、楽しみにしていることなどを教えてください", ""))
    )
    matched_games_canonical = set()
    max_game_point = 0

    # 個別ゲームカラム（レベル付き）
    for col in mentor.index:
        if col.startswith("ゲーム_") and col != "ゲーム_その他":
            game_name = col.replace("ゲーム_", "")
            game_value = pd.to_numeric(mentor[col], errors="coerce")
            if game_value >= 2:
                for g_word in game_list_words:
                    if (game_name.lower() in g_word.lower() or g_word.lower() in game_name.lower()) and g_word in student_text:
                        canonical = word_to_canonical.get(g_word, g_word)
                        matched_games_canonical.add(canonical)
                        point = int(game_value) * 5
                        if point > max_game_point:
                            max_game_point = point

    # ゲーム_その他（自由記述）
    other_games = mentor.get("ゲーム_その他", "")
    other_words = re.split(r"[、,/\s\n]+", other_games)
    for g_word in game_list_words:
        if g_word in student_text and any(g_word in o for o in other_words):
            canonical = word_to_canonical.get(g_word, g_word)
            matched_games_canonical.add(canonical)
            if 15 > max_game_point:
                max_game_point = 15

    score += max_game_point
    if max_game_point > 0 and matched_games_canonical:
        reasons.append(f"ゲームマッチ（{','.join(sorted(matched_games_canonical))}）{max_game_point}点")

    # === 趣味マッチ（テキスト類似度×30点） ===
    mentor_hobby_text = (
        str(mentor.get("得意なこと趣味興味のあること", ""))
        + " " + str(mentor.get("特にどんなスクール生のサポートが得意か", ""))
    )
    def calculate_text_similarity(text1, text2):
        if not text1 or not text2:
            return 0.0
        vectorizer = CountVectorizer().fit_transform([str(text1), str(text2)])
        vectors = vectorizer.toarray()
        return cosine_similarity([vectors[0]], [vectors[1]])[0][0]

    similarity_score = calculate_text_similarity(student_text, mentor_hobby_text)
    hobby_point = similarity_score * 30
    score += hobby_point
    if hobby_point > 0:
        reasons.append(f"趣味・興味マッチ {hobby_point:.1f}点")

    if not reasons:
        reasons.append("最低条件は満たしています")
    return score, "＋".join(reasons)

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("SOZOW メンターマッチングアプリ")

student_df, mentor_df, game_df = load_data()
word_to_canonical, game_list_words = get_game_word_map(game_df)

if len(student_df) == 0:
    st.warning("スクール生情報が空です。")
    st.stop()

student_ids = student_df["スクールID"].tolist()
selected_id = st.selectbox("スクール生IDを選択", student_ids)

if selected_id:
    selected_student = student_df[student_df["スクールID"] == selected_id].iloc[0]
    mentor_df["追加可能人数"] = pd.to_numeric(mentor_df["追加可能人数"], errors="coerce").fillna(0).astype(int)
    scores = []
    reasons = []
    for _, row in mentor_df.iterrows():
        score, reason = calculate_matching_score(selected_student, row, game_list_words, word_to_canonical)
        scores.append(score)
        reasons.append(reason)

    mentor_df["マッチングスコア"] = scores
    mentor_df["おすすめ理由"] = reasons

    matched = mentor_df[mentor_df["マッチングスコア"] > 0].sort_values("マッチングスコア", ascending=False)

    if matched.empty:
        st.warning("条件に一致するメンターがいません。")
    else:
        st.markdown("### 🎯 おすすめメンター一覧（上位10人）")
        st.dataframe(
            matched[["ニックネーム", "マッチングスコア", "追加可能人数", "属性_性別", "おすすめ理由"]].head(10)
        )
else:
    st.info("スクール生を選択してください。")
