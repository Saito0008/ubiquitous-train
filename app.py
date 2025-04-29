import streamlit as st
from newspaper import Article
import openai
import time
import tiktoken
import requests
from datetime import datetime
import librosa
import soundfile as sf
import numpy as np
import os
import uuid
import json

# secretsからAPIキーを取得
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# 履歴ファイルのパス
HISTORY_FILE = "audio_history.json"

# アプリケーションのバージョン
APP_VERSION = "1.0.1"

def load_history():
    """履歴をファイルから読み込む"""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_history(history):
    """履歴をファイルに保存する"""
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

@st.cache_data(ttl=3600)  # 1時間キャッシュ
def get_exchange_rate() -> float:
    """USD/JPYの為替レートを取得"""
    try:
        response = requests.get("https://api.exchangerate-api.com/v4/latest/USD")
        data = response.json()
        return data["rates"]["JPY"]
    except Exception as e:
        st.warning("為替レートの取得に失敗しました。固定レート(145円)を使用します。")
        return 145.0

def count_tokens(text: str) -> int:
    """テキストのトークン数を計算"""
    encoding = tiktoken.encoding_for_model("gpt-4")
    return len(encoding.encode(text))

def format_cost_jpy(usd_cost: float) -> str:
    """USDのコストを日本円に変換"""
    rate = get_exchange_rate()
    jpy_cost = usd_cost * rate
    return f"¥{jpy_cost:,.0f}"

def format_cost_usd(tokens: int) -> float:
    """トークン数から概算コストを計算（USD）"""
    # GPT-4の料金: $0.03/1K tokens (input), $0.06/1K tokens (output)
    return (tokens * 0.03) / 1000

def get_article_text(url):
    """記事を取得する"""
    article = Article(url, language='ja')
    article.download()
    article.parse()
    return {
        'text': article.text,
        'title': article.title,
        'images': article.images
    }

def summarize_article(article_info):
    """記事を要約する"""
    # 記事の本文と画像情報を組み合わせる
    article_content = f"記事タイトル: {article_info['title']}\n\n"
    
    # 画像の説明を追加
    if article_info['images']:
        article_content += "記事内の画像:\n"
        for i, img_url in enumerate(article_info['images'], 1):
            article_content += f"画像{i}: {img_url}\n"
        article_content += "\n"
    
    article_content += f"記事本文:\n{article_info['text']}"
    
    prompt = (
        "以下の記事（本文と画像を含む）の内容を、重要なポイントを逃さないように要約してください。\n"
        "特に画像については、その意図や伝えたいメッセージを分析して要約してください。\n\n"
        "【記事内容】\n"
        f"{article_content}\n\n"
        "【要約】"
    )
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    
    return response.choices[0].message.content.strip()

def combine_audio_files(teacher_file, student_file, output_file="output_combined.mp3"):
    """音声ファイルを結合する"""
    try:
        # 音声ファイルを読み込む
        teacher_audio, sr = librosa.load(teacher_file, sr=None)
        student_audio, _ = librosa.load(student_file, sr=sr)
        
        # 0.5秒の無音を作成
        silence = np.zeros(int(0.5 * sr))
        
        # 音声を結合（正規化してから結合）
        teacher_audio = librosa.util.normalize(teacher_audio)
        student_audio = librosa.util.normalize(student_audio)
        combined = np.concatenate([teacher_audio, silence, student_audio])
        
        # 結合した音声を保存（正規化してから保存）
        combined = librosa.util.normalize(combined)
        sf.write(output_file, combined, sr)
        
        return output_file
    except Exception as e:
        st.error(f"音声の結合中にエラーが発生しました: {str(e)}")
        return None

def split_script_by_speaker(script):
    """台本をA（先生）とB（生徒）のパートに分割"""
    lines = script.split('\n')
    dialogues = []
    
    for line in lines:
        line = line.strip()
        if line.startswith('A:'):
            dialogues.append({
                'speaker': 'teacher',
                'text': line.replace('A:', '').strip()
            })
        elif line.startswith('B:'):
            dialogues.append({
                'speaker': 'student',
                'text': line.replace('B:', '').strip()
            })
    
    return dialogues

def convert_to_ssml(text):
    """テキストをSSML形式に変換する"""
    # 基本的なSSMLタグを追加
    ssml = f"""<speak>
    <prosody rate="1.0">
        {text}
    </prosody>
    </speak>"""
    
    # 句読点の後に間を追加
    ssml = ssml.replace("。", "。<break time='500ms'/>")
    ssml = ssml.replace("、", "、<break time='300ms'/>")
    ssml = ssml.replace("？", "？<break time='500ms'/>")
    ssml = ssml.replace("！", "！<break time='500ms'/>")
    
    # 文節の区切りを追加
    ssml = ssml.replace("「", "<break time='200ms'/>「")
    ssml = ssml.replace("」", "」<break time='200ms'/>")
    
    # 余分な空白を削除
    ssml = ssml.replace("  ", " ")
    ssml = ssml.replace("\n", " ")
    
    return ssml

def generate_tts(script):
    """音声を生成して結合する"""
    # 台本をセリフごとに分割
    dialogues = split_script_by_speaker(script)
    
    # 各セリフの音声を生成して結合
    combined_audio = None
    sr = None
    
    for i, dialogue in enumerate(dialogues):
        # テキストをSSML形式に変換
        ssml_text = convert_to_ssml(dialogue['text'])
        
        try:
            # 音声を生成
            response = client.audio.speech.create(
                model="tts-1",
                voice=st.session_state.teacher_voice if dialogue['speaker'] == 'teacher' else st.session_state.student_voice,
                input=ssml_text,
                response_format="mp3"
            )
            
            # 一時ファイルとして保存
            temp_file = f"temp_{dialogue['speaker']}_{i}.mp3"
            with open(temp_file, "wb") as f:
                f.write(response.content)
            
            # 音声を読み込む
            audio, current_sr = librosa.load(temp_file, sr=None)
            if sr is None:
                sr = current_sr
            
            # 音声を正規化
            audio = librosa.util.normalize(audio)
            
            # 音声を結合
            if combined_audio is None:
                combined_audio = audio
            else:
                # 0.5秒の無音を追加
                silence = np.zeros(int(0.5 * sr))
                combined_audio = np.concatenate([combined_audio, silence, audio])
            
            # 一時ファイルを削除
            os.remove(temp_file)
            
        except Exception as e:
            st.error(f"音声生成中にエラーが発生しました: {str(e)}")
            return None, 0
    
    # 結合した音声を保存
    output_file = "output_combined.mp3"
    combined_audio = librosa.util.normalize(combined_audio)
    sf.write(output_file, combined_audio, sr)
    
    # 音声生成のコストを計算（$0.015/1K characters）
    total_chars = sum(len(d['text']) for d in dialogues)
    tts_cost_usd = (total_chars * 0.015) / 1000
    
    return output_file, tts_cost_usd

def generate_script(article_info):
    total_cost_usd = 0
    
    # 進捗表示用のコンテナを作成
    progress_container = st.container()
    with progress_container:
        st.markdown("### 進捗状況")
        progress_bar = st.progress(0)
        status_text = st.empty()
        time_text = st.empty()
    
    # ステップ1: 記事の要約
    with progress_container:
        status_text.markdown("**ステップ1: 記事を要約中...**")
        time_text.markdown("残り時間: 約30秒")
        progress_bar.progress(20)
    
    summary = summarize_article(article_info)
    
    # ステップ2: 台本の生成
    with progress_container:
        status_text.markdown("**ステップ2: 台本を生成中...**")
        time_text.markdown("残り時間: 約1分")
        progress_bar.progress(50)
    
    prompt = (
        "以下の要約された記事内容を基に、テーマや結論がしっかり伝わるように、"
        "聞き手が理解しやすい長さ（最大20分、ベストな長さはお任せします）で、"
        "日本語のポッドキャスト台本にしてください。\n\n"
        "【台本の形式】\n"
        "- プロフェッショナルなホストA（先生役）と、初学者のホストB（生徒役）による対話形式\n"
        "- 各発言の前に「A:」「B:」をつけて、誰の発言かを明確にする\n"
        "- 会話の間は「...」ではなく「、」や「。」を使って自然な間を表現\n"
        "- 最後に記事の重要なポイントをまとめて締めくくってください\n"
        "- BGMや効果音などの演出指示は含めない\n\n"
        "【台本の内容について】\n"
        "- 専門用語が出てきたら、必ず身近な例を使って説明してください\n"
        "- 「なぜそうなるのか」という理由や背景を丁寧に説明してください\n"
        "- 抽象的な概念は具体的な例を使って説明してください\n"
        "- 重要なポイントは繰り返し説明してください\n"
        "- 生徒役（B）は適度に質問や疑問を投げかけ、理解を深めるようにしてください\n"
        "- 先生役（A）は生徒の理解度を確認しながら、必要に応じて補足説明をしてください\n\n"
        "【日本語の表現について】\n"
        "- 自然な日本語のイントネーションになるように、適切な句読点を使用してください\n"
        "- 重要な部分は強調するように、文の構造を工夫してください\n"
        "- 会話の流れを考慮して、適切な間を取るようにしてください\n"
        "- 文末表現は「です・ます」調を基本とし、必要に応じて「だ・である」調も使用してください\n\n"
        "【画像の扱いについて】\n"
        "- 画像の内容を単に説明するのではなく、その意図や伝えたいメッセージを会話の中で自然に伝えてください\n"
        "- 例えば、象とりんごの大きさを比較する画像がある場合、\n"
        "  ×「象とりんごの画像があります」\n"
        "  ○「りんごは象と比較すると何倍も小さいです」\n"
        "  のように、画像の意図を会話の中で自然に説明してください\n"
        "- 画像の視覚的な要素（色、形、配置など）が重要な場合は、その効果や意図を説明してください\n"
        "- 複数の画像がある場合は、それらの関連性やストーリー性を活かして説明してください\n\n"
        "【記事タイトル】\n"
        f"{article_info['title']}\n\n"
        "【要約された内容】\n"
        f"{summary}\n\n"
        "【ポッドキャスト台本】"
    )
    
    # トークン数を計算
    input_tokens = count_tokens(prompt)
    input_cost_usd = format_cost_usd(input_tokens)
    total_cost_usd += input_cost_usd
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4000,
        temperature=0.8
    )
    
    generated_text = response.choices[0].message.content
    
    # 出力トークン数からコストを計算
    output_tokens = count_tokens(generated_text)
    output_cost_usd = format_cost_usd(output_tokens)
    total_cost_usd += output_cost_usd
    
    # ステップ3: 音声の生成
    with progress_container:
        status_text.markdown("**ステップ3: 音声を生成中...**")
        time_text.markdown("残り時間: 約2分")
        progress_bar.progress(80)
    
    combined_file, tts_cost_usd = generate_tts(generated_text.strip())
    total_cost_usd += tts_cost_usd
    
    # 完了
    with progress_container:
        status_text.markdown("**✅ 処理が完了しました！**")
        time_text.empty()
        progress_bar.progress(100)
    
    # 結果表示
    st.markdown("### 📝 生成された台本")
    st.text_area("", generated_text.strip(), height=300)
    
    st.markdown("### 🔊 生成された音声")
    st.audio(combined_file)
    
    return combined_file, total_cost_usd

# 音声の種類を定義
VOICE_OPTIONS = {
    "男性の声": {
        "alloy": "標準的な男性の声",
        "echo": "落ち着いた男性の声",
        "fable": "若々しい男性の声"
    },
    "女性の声": {
        "nova": "標準的な女性の声",
        "shimmer": "落ち着いた女性の声",
        "onyx": "力強い女性の声"
    }
}

# 音声履歴を初期化
if 'audio_history' not in st.session_state:
    st.session_state.audio_history = load_history()

# バージョン情報を表示
st.markdown("""
<div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 20px;'>
    <h1 style='color: #262730; margin-bottom: 0;'>記事URLからポッドキャスト風音声生成アプリ</h1>
    <p style='color: #666; margin-top: 5px;'>バージョン: {}</p>
</div>
""".format(APP_VERSION), unsafe_allow_html=True)

# サイドバーに音声選択を追加
with st.sidebar:
    st.markdown("### 🎤 音声設定")
    
    # 先生役の音声選択
    st.markdown("#### 先生役の音声")
    teacher_voice = st.selectbox(
        "先生役の音声を選択",
        options=list(VOICE_OPTIONS["男性の声"].keys()),
        format_func=lambda x: f"{x} - {VOICE_OPTIONS['男性の声'][x]}",
        key="teacher_voice"
    )
    
    # 生徒役の音声選択
    st.markdown("#### 生徒役の音声")
    student_voice = st.selectbox(
        "生徒役の音声を選択",
        options=list(VOICE_OPTIONS["女性の声"].keys()),
        format_func=lambda x: f"{x} - {VOICE_OPTIONS['女性の声'][x]}",
        key="student_voice"
    )
    
    st.markdown("---")
    
    # 料金目安の表示
    st.markdown("### 💰 料金目安")
    rate = get_exchange_rate()
    st.markdown(f"**現在の為替レート: $1 = ¥{rate:.2f}**")
    
    st.markdown("""
    #### 1記事あたりの目安料金
    - **短い記事（約1000文字）**: ¥50〜¥100
    - **中程度の記事（約3000文字）**: ¥100〜¥200
    - **長い記事（約5000文字）**: ¥200〜¥300
    
    #### 内訳
    - **GPT-4入力**: ¥4.35/1K tokens
    - **GPT-4出力**: ¥8.70/1K tokens
    - **音声生成**: ¥2.18/1K文字
    
    #### 注意事項
    - 料金は記事の長さや内容によって変動します
    - 画像が多い記事は処理に時間がかかり、料金が高くなる可能性があります
    - 音声の長さによっても料金が変動します
    """)
    st.markdown("---")
    
    # 履歴リストの表示
    if st.session_state.audio_history:
        st.markdown("### 📚 生成履歴")
        for item in reversed(st.session_state.audio_history):
            with st.expander(f"{item['title']} - {item['timestamp']}"):
                st.audio(item['file'])
                with open(item['file'], "rb") as f:
                    st.download_button(
                        "音声をダウンロード",
                        f,
                        file_name=f"podcast_{item['timestamp']}.mp3",
                        mime="audio/mp3",
                        key=f"dl_{item['id']}"
                    )
                
                # 履歴から削除するボタン
                if st.button("この履歴を削除", key=f"delete_{item['id']}"):
                    st.session_state.audio_history = [h for h in st.session_state.audio_history if h['id'] != item['id']]
                    save_history(st.session_state.audio_history)
                    st.rerun()

url = st.text_input("記事のURLを入力してください")

if st.button("台本生成＆音声化"):
    if not url:
        st.error("URLを入力してください。")
    else:
        try:
            article_info = get_article_text(url)
            script, text_cost_usd = generate_script(article_info)
            
            # 音声を再生
            st.audio(script)
            
            # 音声ファイルのダウンロードボタン
            with open(script, "rb") as f:
                st.download_button("音声をダウンロード", f, file_name="podcast.mp3", mime="audio/mp3")
            
            # 音声履歴に追加
            history_item = {
                'id': str(uuid.uuid4()),
                'title': article_info['title'],
                'file': script,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state.audio_history.append(history_item)
            
            # 履歴をファイルに保存
            save_history(st.session_state.audio_history)
            
            # 総コストを表示
            total_cost_usd = text_cost_usd
            st.success(f"処理が完了しました！ 総コスト: {format_cost_jpy(total_cost_usd)}")
            
        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")