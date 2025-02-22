import streamlit as st
import re
import os
from datetime import datetime
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
import pysbd
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Set page configuration
st.set_page_config(
    page_title="Youtube Assistant",
    page_icon="🤖",
    layout="wide"
)

seg = pysbd.Segmenter(language='en', clean=True)

def extract_youtube_video_id(url: str) -> str:
    found = re.search(r"(?:youtu\.be\/|watch\?v=)([\w-]+)", url)
    if found:
        return found.group(1)
    return None

def get_video_transcript(video_id: str) -> str | None:
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['pt', 'en'])
    except TranscriptsDisabled:
        return None
    text_version = ' '.join([line["text"] for line in transcript])
    detailed_version = [{"start": line["start"], "duration": line["duration"], "text": line["text"]} for line in transcript]
    return text_version, detailed_version

def chunk_large_text(text_list: list, max_size: int) -> list[str]:
    txts = []
    para = ''
    for s in text_list:
        s_len = len(s)
        if para and len(para) + s_len > max_size:
            txts.append(para)
            para = ''
        if s_len <= max_size:
            para += s + '\n'
        else:
            if para:
                txts.append(para)
                para = ''
            cut = s_len // max_size
            chunk = s_len // (cut + 1)
            i = 0
            while i < s_len:
                if s_len - i <= chunk:
                    txts.append('… ' + s[i:] + ' …')
                    break
                clip_i = s.find(' ', i + chunk)
                txts.append('… ' + s[i:clip_i] + ' …')
                i = clip_i + 1
    if para:
        txts.append(para)
    return txts

def summarize_large_text(text_list: list, max_size: int) -> str:
    summaries = ""
    txts = chunk_large_text(text_list, max_size)
    summaries = summaries.join(txts)
    return summaries


def generate_summary(question, data=None, data_context=""):
    """Generate summary using Groq """
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    system_prompt = """You are an expert YouTube analyst with extensive experience in channel growth, video content optimization, audience engagement, and data analysis. 
    Your goal is to provide accurate, actionable, and strategic recommendations to YouTube creators to help them grow their channels and improve their content performance. 
    You should focus on leveraging data and trends to offer insightful guidance."""

    # user_message = f"I have a YouTube video, and I need a concise and engaging summary. Start by providing a brief summary in 2-3 sentences that captures the essence of the video. Then, list 5 to 10 key points or takeaways from the video in bullet format. These should include important facts, insights, or actionable steps discussed in the content.Point out the step-by-step guide to achieve similar results.Ensure the language is clear, easy to understand, and highlights the main ideas without unnecessary details.\n\nSummarize the text: {question}"
    
    if len(question) > 6000:
        print(f"Question with: {len(question)}")
        question = question[:22800]

    # Question with: 44427
    # 520
    # Limit 6000, Requested 11373 - 44427
    # Limit 6000, Requested 6130 - 23500
    # Limit 6000, Requested 6005 - 23000

    user_message = f"""{data_context}\n\nHere are text: {question}"""
    print(len(data_context))

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        model="deepseek-r1-distill-llama-70b",
    )

    response = chat_completion.choices[0].message.content
    print(chat_completion.usage)
    # return response
    # Include thinking
    return clean_code_response(response)


def clean_code_response(response):
    """Clean the code response from Groq"""
    # Remove thinking section
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)

    # Extract code from markdown if present
    code_match = re.search(r'''```python(.*?)```''', response, flags=re.DOTALL)
    if code_match:
        return code_match.group(1).strip()

    return response.strip()


# Streamlit UI
st.title('YouTube Transcript and Summarizer using Deepseek')


# model_names
model_names = ['deepseek-r1-distill-llama-70b']
selected_model = st.selectbox("Choose a LLM model", model_names, index=model_names.index("deepseek-r1-distill-llama-70b"))

url = st.text_input('Enter YouTube URL')

# prompt = st.text_area(
#     "Prompt",
#     '''You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior.
# You job is summarize texts. Please do not say anything else and do not start a conversation. 
# The text can be in either English or Brazilian Portuguese. If the text is in English, please summarize and respond in English. If the text is in Portuguese, please summarize and respond in Portuguese.
# Summarize the content of this YouTube video transcript into 5 to 10 concise and easy-to-read bullet points. Focus on capturing the most important information, key takeaways, and highlights of the video. Ensure the summary is accurate, avoids unnecessary details, and presents the points in a logical order.
# Summarize the text:''',200
# )
# prompt = st.text_area(
#     "Prompt",
#     '''You are a cautious assistant. You carefully follow instructions. You are helpful and harmless, and you follow ethical guidelines and promote positive behavior.
# You job is summarize texts. Please do not say anything else and do not start a conversation. 
# The text can be in either English or Brazilian Portuguese. If the text is in English, please summarize and respond in English. If the text is in Portuguese, please summarize and respond in Portuguese.
# I have a YouTube video, and I need a concise and engaging summary. Start by providing a brief summary in 2-3 sentences that captures the essence of the video. Then, list 5 to 10 key points or takeaways from the video in bullet format. These should include important facts, insights, or actionable steps discussed in the content. Ensure the language is clear, easy to understand, and highlights the main ideas without unnecessary details.
# Summarize the text:''',180
# )

# If the text is in Portuguese, please summarize and translate to Portuguese.
prompt = st.text_area(
    "Prompt",
'''I have a YouTube video, and I need a concise and engaging summary. Start by providing a brief summary in 2-3 sentences that captures the essence of the video. 
Then, list 5 to 10 key points or takeaways from the video in bullet format. These should include important facts, insights, or actionable steps discussed in the content.
Point out the step-by-step guide to achieve similar results.

Ensure the language is clear, easy to understand, and highlights the main ideas without unnecessary details.
Summarize the text:''',180
)

st.write(f"You wrote {len(prompt)} characters.")

submit = st.button('Submit')

col1, col2 = st.columns(2)

with col1:
    if submit and url and selected_model:
        video_id = extract_youtube_video_id(url)
        if video_id:
            transcript, transcript_detailed = get_video_transcript(video_id)
            if transcript and transcript_detailed:
                text_list = seg.segment(transcript)
                summary = summarize_large_text(text_list, max_size=2048)
                #print(summary)
                st.subheader("Extracted Text from Video")
                # st.code(summary, language=None,wrap_lines=True)
                st.code(transcript_detailed, language="python",wrap_lines=True)

                if summary:
                    with col2:
                        summary = prompt + "\n" + summary
                        st.subheader('Summary')
                        summary_response = generate_summary(summary, data_context=prompt)
                        st.code(summary_response, language=None,wrap_lines=True)
            else:
                st.write("No transcript found for this video.")
        else:
            st.write("Invalid YouTube URL.")





#prompt = "Summarize the following content within 150 tokens : "
# prompt = '''You are a cautious assistant. You carefully follow instructions. 
# You are helpful and harmless and you follow ethical guidelines and promote positive behavior.
# You job is summarize texts.
# Please do not say anything else and do not start a conversation. 
# The text can be in either English or Brazilian Portuguese. If the text is in English, please summarize and respond in English. If the text is in Portuguese, please summarize and respond in Portuguese.
# Summarize the content of this YouTube video transcript into 5 to 10 concise and easy-to-read bullet points. Focus on capturing the most important information, key takeaways, and highlights of the video. Ensure the summary is accurate, avoids unnecessary details, and presents the points in a logical order.
# Sumarize the text:'''