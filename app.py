import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# --- データ取得を関数化 ---
@st.cache_data(ttl=600)
def load_data():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    # ローカルなら credentials.json
    if os.path.exists("credentials.json"):
        creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
    # Cloud（Streamlit Cloud）の場合
    elif "gcp_service_account" in st.secrets and st.secrets["gcp_service_account"].get("private_key"):
        creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["gcp_service_account"], scope)
    else:
        st.error("認証情報がありません。CloudはSecrets、ローカルはcredentials.jsonが必要です。")
        st.stop()
    client = gspread.authorize(creds)
    spreadsheet = client.open_by_key("1ISs5mqSRdZfF3NVOt60VFtY8p8HsM0ZkM3sfu3cPzVE")
    student_df = pd.DataFrame(spreadsheet.worksheet("スクール生情報").get_all_records())
    mentor_df = pd.DataFrame(spreadsheet.worksheet("メンター情報").get_all_records())
    st.write("メンター情報：", mentor_df)
    return student_df, mentor_df

# --- データ取得 ---
student_df, mentor_df = load_data()

# --- テキスト類似度 ---
def calculate_text_similarity(text1, text2):
    if not text1 or not text2:
        return 0.0
    vectorizer = CountVectorizer().fit_transform([str(text1), str(text2)])
    vectors = vectorizer.toarray()
    return cosine_similarity([vectors[0]], [vectors[1]])[0][0]

# --- スロット抽出 ---
def extract_possible_slots(student):
    slots = []
    for col in student.index:
        if "定期的" in col and "[" in col and "]" in col:
            value = student[col]
            if isinstance(value, str) and value.strip():
                days = [d.strip() for d in value.split(",")]
                try:
                    hour = col.split("[")[-1].split("〜")[0].replace("：", ":").strip()
                    for day in days:
                        if day in ["月", "火", "水", "木", "金", "土", "日"]:
                            slot = f"1on1可能時間_{day}_{hour}～"
                            slots.append(slot)
                except:
                    continue
    return slots

# --- マッチングスコア計算 ---
def calculate_matching_score(student, mentor):
    reasons = []
    possible_slots = extract_possible_slots(student)
    matched = any(str(mentor.get(slot, "")).strip().lower() == "true" for slot in possible_slots)
    if not matched:
        return 0, "時間帯が一致しない"
    try:
        if int(mentor.get("追加可能人数", 0)) < 1:
            return 0, "担当枠が空いていない"
    except:
        return 0, "追加可能人数の取得エラー"
    score = 50
    game_matching_score = 0
    matched_keywords = []
    student_text = student.get("お子さまの得意なこと、好きなことを教えてください", "") + \
                   student.get("興味がある分野をお答えください", "")
    for col in mentor.index:
        if col.startswith("ゲーム_"):
            game_name = col.replace("ゲーム_", "")
            game_value = pd.to_numeric(mentor[col], errors="coerce")
            if game_value >= 2 and game_name in student_text:
                matched_keywords.append(game_name)
                game_matching_score += (game_value - 1)
    score += game_matching_score
    preferred_keywords = ["マイクラ", "Minecraft"]
    game_info_columns = [col for col in mentor.index if col.startswith("ゲーム_") and pd.to_numeric(mentor[col], errors="coerce") >= 2]
    game_info = ", ".join([col.replace("ゲーム_", "") for col in game_info_columns])
    bonus = 0
    preferred_matched = []
    for keyword in preferred_keywords:
        if keyword.lower() in game_info.lower():
            bonus += 10
            preferred_matched.append(keyword)
    score += bonus
    mentor_gender = mentor.get("属性_性別", "").strip()
    student_gender = student.get("お子さまの性別", "").strip()
    student_gender_pref = student.get("メンターの性別のご希望", "").strip()

    # ① 性別希望がある場合 → 一致しなければスコア0で終了
    if student_gender_pref and student_gender_pref not in ["指定なし", "", None]:
        if student_gender_pref != mentor_gender:
            return 0, "性別希望に一致しない"
        else:
            score += 30
            reasons.append("性別一致（希望通り）")

    # ② 性別希望がない場合 → 本人と同じ性別なら少し加点
    elif student_gender and student_gender == mentor_gender:
        score += 10
        reasons.append("性別一致（本人と同じ）")
    sim1 = calculate_text_similarity(student.get("お子さまの得意なこと、好きなことを教えてください", ""),
                                     mentor.get("得意なこと・趣味・興味のあること", "") + " " + game_info)
    sim2 = calculate_text_similarity(student.get("興味がある分野をお答えください", ""),
                                     mentor.get("得意なこと・趣味・興味のあること", "") + " " + game_info)
    sim3 = calculate_text_similarity(student.get("お子さまがSOZOWスクールに期待していること、楽しみにしていることなどを教えてください", ""),
                                     mentor.get("特にどんなスクール生のサポートが得意か", "") + " " + game_info)
    similarity_score = (sim1 + sim2 + sim3) / 3
    score += similarity_score * 20
    interest_reason = []
    if sim1 > 0.15: interest_reason.append("得意・好きなことが近い")
    if sim2 > 0.15: interest_reason.append("興味分野が似ている")
    if sim3 > 0.15: interest_reason.append("スクールへの期待が似ている")
    student_relation_pref = student.get("お子さまが今まで関わった大人の中で、良好なコミュニケーションがとれていた方に共通する特徴を教えてください", "") + \
                            student.get("相性の良い大人", "")
    mentor_personality = mentor.get("性格・コミュニケーション特性", "") + " " + mentor.get("特にどんなスクール生のサポートが得意か", "")
    sim_relation = calculate_text_similarity(student_relation_pref, mentor_personality)
    if sim_relation > 0.2:
        score += 10
        reasons.append("相性が良さそう（関わり方の特徴が近い）")
    if preferred_matched:
        reasons.append(f"希望のゲームに対応（{', '.join(preferred_matched)}）")
    if matched_keywords:
        reasons.append(f"得意ゲーム：{', '.join(matched_keywords)}")
    if interest_reason:
        reasons.append("＋".join(interest_reason))
    if sim_relation > 0.2:
        reasons.append("関わり方の相性が良い")
    if not reasons:
        reasons.append("最低条件は満たしています")
    return score, "＋".join(reasons)

# --- Streamlit アプリ ---
st.set_page_config(layout="wide")
st.title("SOZOWメンターマッチングアプリ")

# ★データが空のときも安全
if len(student_df) == 0:
    st.warning("スクール生情報が空です。")
    st.stop()

selected_id = st.selectbox("スクール生を選択", student_df["(編集不可)スクールID"].unique())

# --- 選択肢が1件以上ある場合だけ処理（Cloudで初期化エラーを100%防ぐ） ---
if selected_id:
    selected_student = student_df[student_df["(編集不可)スクールID"] == selected_id].iloc[0]
    mentor_df["追加可能人数"] = pd.to_numeric(mentor_df["追加可能人数"], errors="coerce").fillna(0).astype(int)
    scores = []
    reasons = []
    for _, row in mentor_df.iterrows():
        score, reason = calculate_matching_score(selected_student, row)
        scores.append(score)
        reasons.append(reason)
    st.write("計算されたマッチングスコア一覧:", scores)
    mentor_df["マッチングスコア"] = scores
    mentor_df["おすすめ理由"] = reasons

    matched = mentor_df[mentor_df["マッチングスコア"] > 0].sort_values("マッチングスコア", ascending=False)

    st.markdown("### 🎯 おすすめメンター一覧")
    st.dataframe(matched[["ニックネーム\n（自動）", "マッチングスコア", "追加可能人数", "属性_性別", "おすすめ理由"]].head(10))
else:
    st.info("スクール生を選択してください。")



