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

# secretsからAPIキーを取得
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = openai.OpenAI(api_key=OPENAI_API_KEY)

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
        "特に画像の内容も含めて要約してください。\n\n"
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
    # 音声ファイルを読み込む
    teacher_audio, sr = librosa.load(teacher_file, sr=None)
    student_audio, _ = librosa.load(student_file, sr=sr)
    
    # 0.5秒の無音を作成
    silence = np.zeros(int(0.5 * sr))
    
    # 音声を結合
    combined = np.concatenate([teacher_audio, silence, student_audio])
    
    # 結合した音声を保存
    sf.write(output_file, combined, sr)
    return output_file

def split_script_by_speaker(script):
    """台本をA（先生）とB（生徒）のパートに分割"""
    lines = script.split('\n')
    a_lines = []
    b_lines = []
    
    for line in lines:
        if line.startswith('A:'):
            a_lines.append(line.replace('A:', '').strip())
        elif line.startswith('B:'):
            b_lines.append(line.replace('B:', '').strip())
    
    return {
        'teacher': ' '.join(a_lines),
        'student': ' '.join(b_lines)
    }

def generate_script(article_info):
    start_time = time.time()
    total_cost_usd = 0
    
    with st.status("台本を生成中...", expanded=True) as status:
        # ステップ1のステータス表示
        status.update(label="Step 1: 記事を要約中...")
        summary = summarize_article(article_info)
        
        # ステップ2のステータス表示
        status.update(label="Step 2: 台本を生成中...")
        prompt = (
            "以下の要約された記事内容を基に、テーマや結論がしっかり伝わるように、"
            "聞き手が理解しやすい長さ（最大20分、ベストな長さはお任せします）で、"
            "日本語のポッドキャスト台本にしてください。\n\n"
            "【台本の形式】\n"
            "- プロフェッショナルなホストA（先生役）と、初学者のホストB（生徒役）による対話形式\n"
            "- 各発言の前に「A:」「B:」をつけて、誰の発言かを明確にする\n"
            "- 会話の間は「...」ではなく「、」や「。」を使って自然な間を表現\n"
            "- 記事内の画像についても詳しく説明してください\n"
            "- 最後に記事の重要なポイントをまとめて締めくくってください\n"
            "- BGMや効果音などの演出指示は含めない\n\n"
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
        
        progress_bar = st.progress(0)
        time_placeholder = st.empty()
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000,
            temperature=0.8,
            stream=True
        )
        
        generated_text = ""
        estimated_time = 60  # 約1分を想定
        
        for chunk in response:
            if chunk.choices[0].delta.content:
                generated_text += chunk.choices[0].delta.content
                progress = min(0.95, (time.time() - start_time) / estimated_time)
                progress_bar.progress(progress)
                
                elapsed_time = time.time() - start_time
                remaining_time = max(0, estimated_time - elapsed_time)
                time_placeholder.text(f"残り時間: 約{int(remaining_time)}秒")
                
                status.update(label=f"台本を生成中... ({int(progress * 100)}%)")
        
        progress_bar.progress(1.0)
        time_placeholder.text("生成完了！")
        
        # 出力トークン数からコストを計算
        output_tokens = count_tokens(generated_text)
        output_cost_usd = format_cost_usd(output_tokens)
        total_cost_usd += output_cost_usd
        
        status.update(label="台本の生成が完了しました！", state="complete")
        
        return generated_text.strip(), total_cost_usd

def generate_tts(script):
    """音声を生成して結合する"""
    with st.status("音声を生成中...", expanded=True) as status:
        # 台本を話者ごとに分割
        parts = split_script_by_speaker(script)
        
        # 先生役（Alloy）と生徒役（Nova）の音声を生成
        status.update(label="先生役の音声を生成中...")
        teacher_response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",  # プロフェッショナルな声
            input=parts['teacher']
        )
        
        status.update(label="生徒役の音声を生成中...")
        student_response = client.audio.speech.create(
            model="tts-1",
            voice="nova",  # フレンドリーな声
            input=parts['student']
        )
        
        # 一時ファイルとして保存
        teacher_file = "temp_teacher.mp3"
        student_file = "temp_student.mp3"
        
        with open(teacher_file, "wb") as f:
            f.write(teacher_response.content)
        with open(student_file, "wb") as f:
            f.write(student_response.content)
        
        # 音声を結合
        status.update(label="音声を結合中...")
        combined_file = combine_audio_files(teacher_file, student_file)
        
        # 一時ファイルを削除
        os.remove(teacher_file)
        os.remove(student_file)
        
        # 音声生成のコストを計算（$0.015/1K characters）
        total_chars = len(parts['teacher']) + len(parts['student'])
        tts_cost_usd = (total_chars * 0.015) / 1000
        
        status.update(label="音声の生成が完了しました！", state="complete")
        return combined_file, tts_cost_usd

st.title("記事URLからポッドキャスト風音声生成アプリ")

# サイドバーに料金目安を表示
with st.sidebar:
    st.markdown("### 💰 料金目安")
    rate = get_exchange_rate()
    st.markdown(f"**現在の為替レート: $1 = ¥{rate:.2f}**")
    st.markdown("""
    - GPT-4入力: ¥4.35/1K tokens
    - GPT-4出力: ¥8.70/1K tokens
    - 音声生成: ¥2.18/1K文字
    """)
    st.markdown("---")

url = st.text_input("記事のURLを入力してください")

if st.button("台本生成＆音声化"):
    if not url:
        st.error("URLを入力してください。")
    else:
        try:
            article_info = get_article_text(url)
            script, text_cost_usd = generate_script(article_info)
            st.text_area("生成された台本", script, height=300)
            
            combined_file, tts_cost_usd = generate_tts(script)
            
            # 音声を再生
            st.audio(combined_file)
            
            # 音声ファイルのダウンロードボタン
            with open(combined_file, "rb") as f:
                st.download_button("音声をダウンロード", f, file_name="podcast.mp3", mime="audio/mp3")
            
            # 一時ファイルを削除
            os.remove(combined_file)
            
            # 総コストを表示
            total_cost_usd = text_cost_usd + tts_cost_usd
            st.success(f"処理が完了しました！ 総コスト: {format_cost_jpy(total_cost_usd)}")
            
        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")