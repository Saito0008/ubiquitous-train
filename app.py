import streamlit as st
from newspaper import Article
import openai

# secretsからAPIキーを取得
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def get_article_text(url):
    article = Article(url, language='ja')
    article.download()
    article.parse()
    return article.text

def generate_script(text):
    prompt = (
        "以下の記事本文を、テーマや結論がしっかり伝わるように、聞き手が理解しやすい長さ（最大20分、ベストな長さはお任せします）で、"
        "日本語のポッドキャスト台本にしてください。"
        "無駄に長くせず、要点を押さえて、二人のパーソナリティによる対話形式で、親しみやすい雰囲気でお願いします。"
        "特に、聞き手の頭に内容が入りやすいように、文章と文章の間や話すスピード、言い回し、間の取り方なども工夫してください。"
        "また、BGMや効果音など本来読み上げない演出指示（例：『オープニングBGM』など）は台本に含めず、純粋に話すべき内容だけを出力してください。\n\n"
        "【記事本文】\n"
        f"{text}\n\n"
        "【ポッドキャスト台本】"
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4000,
        temperature=0.8,
    )
    return response.choices[0].message.content.strip()

def generate_tts(script, filename="output.mp3", voice="nova"):
    response = client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=script
    )
    with open(filename, "wb") as f:
        f.write(response.content)
    return filename

st.title("記事URLからポッドキャスト風音声生成アプリ")
url = st.text_input("記事のURLを入力してください")

if st.button("台本生成＆音声化"):
    if not url:
        st.error("URLを入力してください。")
    else:
        with st.spinner("記事を取得中..."):
            try:
                text = get_article_text(url)
            except Exception as e:
                st.error(f"記事取得エラー: {e}")
                st.stop()
        with st.spinner("台本を生成中..."):
            try:
                script = generate_script(text)
            except Exception as e:
                st.error(f"台本生成エラー: {e}")
                st.stop()
        st.success("台本生成完了！")
        st.text_area("生成された台本", script, height=300)
        with st.spinner("音声を生成中..."):
            try:
                filename = generate_tts(script)
            except Exception as e:
                st.error(f"音声生成エラー: {e}")
                st.stop()
        st.audio(filename)
        with open(filename, "rb") as f:
            st.download_button("音声をダウンロード", f, file_name="output.mp3", mime="audio/mp3")