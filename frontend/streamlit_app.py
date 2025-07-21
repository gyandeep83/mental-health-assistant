# frontend/streamlit_app.py

from datetime import datetime, timedelta, timezone
import streamlit as st
import requests
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
from datetime import datetime, timedelta, timezone
import plotly.express as px
import os
from dotenv import load_dotenv, find_dotenv

# --- Environment Variables ---
load_dotenv(find_dotenv())
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

# --- Enhanced Remedy Recommendation Setup ---
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load Enhanced Remedy Bank
with open("/Users/gyandeep/remedies_bank/enhanced_remedy_bank.pkl", "rb") as f:
    remedy_bank = pickle.load(f)

remedy_texts = remedy_bank["remedy_texts"]
remedy_embeddings = remedy_bank["remedy_embeddings"]
keyword_embeddings = remedy_bank["keyword_embeddings"]

# Load models
embedder = SentenceTransformer('all-MiniLM-L6-v2')
sentiment_analyzer = SentimentIntensityAnalyzer()

def detect_emotion_category(entry_text):
    """Detect emotion category from journal entry"""
    sentiment_score = sentiment_analyzer.polarity_scores(entry_text)
    compound = sentiment_score['compound']
    
    if compound >= 0.1:
        return "positive"
    elif compound <= -0.1:
        return "negative"
    else:
        return "neutral"

def extract_emotional_keywords(entry_text):
    """Extract potential emotional keywords from entry"""
    entry_lower = entry_text.lower()
    
    stress_keywords = ["stress", "overwhelm", "pressure", "busy", "hectic", "chaotic"]
    exhaustion_keywords = ["tired", "exhaust", "fatigue", "drain", "worn out", "burned out"]
    anxiety_keywords = ["anxious", "worry", "nervous", "panic", "fear", "scared"]
    sadness_keywords = ["sad", "depress", "down", "low", "blue", "empty", "hopeless"]
    anger_keywords = ["angry", "mad", "frustrated", "irritated", "annoyed", "furious"]
    happiness_keywords = ["happy", "joy", "excited", "great", "wonderful", "amazing"]
    
    found_keywords = []
    for word in entry_lower.split():
        clean_word = word.strip('.,!?;:"()[]{}')
        for category, keywords in [
            ("stress", stress_keywords),
            ("exhaustion", exhaustion_keywords), 
            ("anxiety", anxiety_keywords),
            ("sadness", sadness_keywords),
            ("anger", anger_keywords),
            ("happiness", happiness_keywords)
        ]:
            if any(keyword in clean_word for keyword in keywords):
                found_keywords.append(category)
    
    return found_keywords

def recommend_remedies(entry_text, top_n=2):
    """Enhanced remedy recommendation with emotion awareness"""
    
    # Detect emotion category
    emotion_category = detect_emotion_category(entry_text)
    detected_keywords = extract_emotional_keywords(entry_text)
    
    # Embed the journal entry
    entry_embedding = embedder.encode([entry_text])
    
    # Calculate similarities
    text_similarities = cosine_similarity(entry_embedding, remedy_embeddings)[0]
    keyword_similarities = cosine_similarity(entry_embedding, keyword_embeddings)[0]
    
    # Create combined score with emotion filtering
    combined_scores = []
    
    for i, (text_sim, keyword_sim) in enumerate(zip(text_similarities, keyword_similarities)):
        remedy_row = remedy_texts.iloc[i]
        
        # Base score: weighted combination
        base_score = (0.4 * text_sim) + (0.6 * keyword_sim)
        
        # Emotion target bonus/penalty
        emotion_bonus = 0
        if remedy_row['emotion_target'] == emotion_category:
            emotion_bonus = 0.3
        elif remedy_row['emotion_target'] != emotion_category and emotion_category != "neutral":
            emotion_bonus = -0.2
        
        # Keyword category bonus
        keyword_bonus = 0
        remedy_keywords = remedy_row['keywords']
        for detected_keyword in detected_keywords:
            if any(detected_keyword in keyword.lower() for keyword in remedy_keywords):
                keyword_bonus += 0.1
        
        # Final score
        final_score = base_score + emotion_bonus + keyword_bonus
        combined_scores.append(final_score)
    
    # Get top N remedies
    combined_scores = np.array(combined_scores)
    top_indices = combined_scores.argsort()[-top_n:][::-1]
    
    # Return recommendations
    recommendations = []
    for idx in top_indices:
        remedy_row = remedy_texts.iloc[idx]
        recommendations.append((remedy_row['text'], combined_scores[idx]))
    
    return recommendations


# --- Helper function to clean Unicode characters ---
def clean_unicode_text(text):
    """Clean problematic Unicode characters that cause encoding issues"""
    if not isinstance(text, str):
        return str(text)
    
    # Replace problematic Unicode characters
    replacements = {
        '\u2028': ' ',  # Line separator
        '\u2029': ' ',  # Paragraph separator
        '\u00a0': ' ',  # Non-breaking space
        '\u200b': '',   # Zero-width space
        '\u200c': '',   # Zero-width non-joiner
        '\u200d': '',   # Zero-width joiner
        '\ufeff': '',   # Byte order mark
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove any remaining non-printable characters
    text = ''.join(char for char in text if char.isprintable() or char.isspace())
    
    return text

# --- Groq API Integration ---
def analyze_weekly_summary_with_groq(journal_entries, sentiment_stats):
    """Analyze weekly journal entries using Groq API"""
    if not GROQ_API_KEY:
        return "⚠️ GROQ_API_KEY not found in environment variables. Please set it up to use AI insights."
    
    # Clean journal entries before processing
    cleaned_entries = []
    for entry in journal_entries:
        cleaned_entry = {
            'entry_date': clean_unicode_text(entry.get('entry_date', '')),
            'entry_text': clean_unicode_text(entry.get('entry_text', ''))
        }
        cleaned_entries.append(cleaned_entry)
    
    combined_text = "\n\n".join([
        f"{entry['entry_date']}: {entry['entry_text']}" for entry in cleaned_entries
    ])
    
    prompt = f"""You are a compassionate mental health assistant helping the user reflect on their week.

Here are this week's journaling statistics:
- Positive entries: {sentiment_stats['positive']}
- Neutral entries: {sentiment_stats['neutral']}
- Negative entries: {sentiment_stats['negative']}
- Total entries: {sentiment_stats['total']}

Journal Entries:
{combined_text}

Based on the above, provide a short summary of the emotional trends and offer gentle advice."""

    # Clean the prompt text as well
    cleaned_prompt = clean_unicode_text(prompt)
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json; charset=utf-8"
    }
    
    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are a compassionate mental health assistant helping the user reflect on their week."
            },
            {
                "role": "user", 
                "content": cleaned_prompt
            }
        ],
        "model": GROQ_MODEL,
        "temperature": 0.7,
        "max_tokens": 500
    }
    
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            return f"⚠️ Error calling Groq API: {response.status_code} - {response.text}"
            
    except requests.exceptions.RequestException as e:
        return f"⚠️ Network error calling Groq API: {str(e)}"
    except Exception as e:
        return f"⚠️ Unexpected error: {str(e)}"


API_URL = "http://127.0.0.1:8000/api/journal/entries/"  # Your Django endpoint

# Custom CSS for dark theme consistency
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
    
    .entry-card {
        border: 1px solid rgba(250, 250, 250, 0.1);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        transition: all 0.2s ease;
    }
    
    .entry-card:hover {
        border-color: rgba(250, 250, 250, 0.2);
        transform: translateY(-1px);
    }
    
    .remedy-card {
        background: rgba(30, 30, 30, 0.6);
        border-left: 3px solid rgba(250, 250, 250, 0.3);
        border-radius: 6px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
    
    .sentiment-positive { 
        background-color: rgba(76, 175, 80, 0.1);
        border-left-color: #4CAF50;
    }
    .sentiment-negative { 
        background-color: rgba(244, 67, 54, 0.1);
        border-left-color: #F44336;
    }
    .sentiment-neutral { 
        background-color: rgba(255, 152, 0, 0.1);
        border-left-color: #FF9800;
    }
    
    .compact-entry {
        font-size: 0.9rem;
        line-height: 1.4;
    }
    
    .entry-preview {
        opacity: 0.8;
        max-height: 60px;
        overflow: hidden;
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
    }
    
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
    
    .stat-item {
        text-align: center;
        padding: 0.5rem;
    }
    
    .pagination-info {
        text-align: center;
        color: rgba(250, 250, 250, 0.6);
        margin: 1rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>📝 Mental Health Journal</h1></div>', unsafe_allow_html=True)

# Placeholder for token (we'll improve this later)
token = st.text_input("🔐 Enter your auth token", type="password")

# Clean the token to prevent Unicode encoding issues
clean_token = clean_unicode_text(token) if token else ""

headers = {
    "Authorization": f"Token {clean_token}"
} if clean_token else {}

# Journal Entry Section
with st.container():
    st.subheader("✍️ New Journal Entry")
    entry_text = st.text_area("How are you feeling today?", height=120, placeholder="Share your thoughts, feelings, and experiences...")

# Utility: sentiment emoji
def get_sentiment_emoji(label):
    label = label.upper()
    if label == "POSITIVE":
        return "😊"
    elif label == "NEGATIVE":
        return "😔"
    elif label == "NEUTRAL":
        return "😐"
    else:
        return "❓"

# Enhanced Entry Submission Section
if st.button("Submit Entry", type="primary"):
    if not clean_token:
        st.warning("Please enter your auth token.")
    elif not entry_text.strip():
        st.warning("Journal entry cannot be empty.")
    else:
        # Clean the entry text before sending
        cleaned_entry_text = clean_unicode_text(entry_text)
        
        payload = {
            "entry_text": cleaned_entry_text
        }
        response = requests.post(API_URL, headers=headers, json=payload)
        if response.status_code == 201:
            st.success("Entry submitted successfully!")
            
            # Enhanced Remedy Recommendation
            st.subheader("🌟 Personalized Remedies")
            
            # Show detected emotion for transparency
            detected_emotion = detect_emotion_category(cleaned_entry_text)
            detected_keywords = extract_emotional_keywords(cleaned_entry_text)
            
            # Display emotion detection (optional - you can remove this)
            with st.expander("🧠 Emotion Analysis", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Detected Emotion:** {detected_emotion.title()}")
                with col2:
                    if detected_keywords:
                        st.write(f"**Key Themes:** {', '.join(set(detected_keywords))}")
            
            # Get and display recommendations
            recommendations = recommend_remedies(cleaned_entry_text, top_n=2)
            
            for i, (remedy_text, score) in enumerate(recommendations, 1):
                clean_remedy_text = clean_unicode_text(remedy_text)
                
                sentiment_class = f"sentiment-{detected_emotion}"
                
                st.markdown(f"""
                <div class="remedy-card {sentiment_class}">
                    <strong>💡 Remedy {i}:</strong> {clean_remedy_text}
                </div>
                """, unsafe_allow_html=True)

            # Sentiment prediction with transformers
            try:
                from transformers import pipeline
                sentiment_pipeline = pipeline("sentiment-analysis")
                prediction = sentiment_pipeline(cleaned_entry_text)[0]
                predicted_sentiment = prediction['label']

                st.info(f"**Mood Detected:** {predicted_sentiment} {get_sentiment_emoji(predicted_sentiment)}")
            except Exception as e:
                st.warning(f"Could not load sentiment analysis model: {str(e)}")

        else:
            st.error(f"Failed to submit entry: {response.status_code}")
            
            
def format_timestamp(iso_string):
    try:
        dt = datetime.fromisoformat(iso_string.replace("Z", "+00:00"))
        return dt.strftime("%b %d, %Y - %I:%M %p")  # e.g., Apr 22, 2025 - 11:06 AM
    except Exception as e:
        return iso_string  # fallback in case of bad format

def format_date_short(iso_string):
    try:
        dt = datetime.fromisoformat(iso_string.replace("Z", "+00:00"))
        return dt.strftime("%m/%d")  # e.g., 04/22
    except Exception as e:
        return iso_string

if clean_token:
    response = requests.get(API_URL, headers=headers)
    if response.status_code == 200:
        entries = response.json()

        # --- Enhanced Journal Entries Section ---
        st.markdown("---")
        st.subheader("📓 Your Journal Entries")
        
        # Initialize session state for pagination and view mode
        if 'entries_per_page' not in st.session_state:
            st.session_state.entries_per_page = 5
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 0
        if 'view_mode' not in st.session_state:
            st.session_state.view_mode = 'compact'

        # Control panel
        col1, col2, col3 = st.columns([3, 2.5, 2])
        
        with col1:
            selected_sentiments = st.multiselect(
                "🎭 Filter by Sentiment",
                options=["POSITIVE", "NEGATIVE", "NEUTRAL"],
                default=["POSITIVE", "NEGATIVE", "NEUTRAL"],
                help="Select which sentiment types to display"
            )
        
        with col2:
                    # Add some spacing before view mode
                    st.markdown("<div style='margin: 0 1rem;'>", unsafe_allow_html=True)
                    view_mode = st.radio(
                        "👁️ View Mode",
                        options=["compact", "detailed"],
                        index=0 if st.session_state.view_mode == 'compact' else 1,
                        horizontal=True
                    )
                    st.session_state.view_mode = view_mode
                    st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            entries_per_page = st.selectbox(
                "📄 Entries per Page",
                options=[5, 10, 15, 20, 50],
                index=[5, 10, 15, 20, 50].index(st.session_state.entries_per_page) if st.session_state.entries_per_page in [5, 10, 15, 20, 50] else 0
            )
            if entries_per_page != st.session_state.entries_per_page:
                st.session_state.entries_per_page = entries_per_page
                st.session_state.current_page = 0
        
        # Additional filter options
        col1, col2, col3 = st.columns([2.5, 2, 2.5])
        
        with col1:
            date_filter = st.selectbox(
                "📅 Time Period",
                options=["All Time", "Last 7 Days", "Last 30 Days", "Last 3 Months"],
                index=0
            )
        
        with col2:
            st.markdown("<div style='margin-top: 1.5rem;'>", unsafe_allow_html=True)
            if st.button("🔄 Reset All Filters", use_container_width=True):
                st.session_state.current_page = 0
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            sort_order = st.selectbox(
                "🔄 Sort Order",
                options=["Newest First", "Oldest First"],
                index=0
            )

        # Filter entries based on all criteria
        filtered_entries = entries.copy()
        
        # Sentiment filter
        filtered_entries = [
            entry for entry in filtered_entries
            if clean_unicode_text(entry.get('sentiment_label', 'NEUTRAL')).upper() in selected_sentiments
        ]
        
        # Date filter
        if date_filter != "All Time":
            now = datetime.now(timezone.utc)
            if date_filter == "Last 7 Days":
                cutoff = now - timedelta(days=7)
            elif date_filter == "Last 30 Days":
                cutoff = now - timedelta(days=30)
            elif date_filter == "Last 3 Months":
                cutoff = now - timedelta(days=90)
            
            filtered_entries = [
                entry for entry in filtered_entries
                if datetime.fromisoformat(entry["entry_date"].replace("Z", "+00:00")) >= cutoff
            ]
        
        # Sort entries
        if sort_order == "Oldest First":
            filtered_entries = sorted(filtered_entries, key=lambda x: x["entry_date"])
        else:
            filtered_entries = sorted(filtered_entries, key=lambda x: x["entry_date"], reverse=True)

        if not filtered_entries:
            st.info("No entries match the selected sentiment filter.")
        else:
            # Pagination logic
            total_entries = len(filtered_entries)
            total_pages = (total_entries - 1) // entries_per_page + 1
            
            # Pagination controls
            col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
            
            with col1:
                if st.button("⏮️ First") and st.session_state.current_page > 0:
                    st.session_state.current_page = 0
                    st.rerun()
            
            with col2:
                if st.button("◀️ Prev") and st.session_state.current_page > 0:
                    st.session_state.current_page -= 1
                    st.rerun()
            
            with col3:
                st.markdown(f'<div class="pagination-info">Page {st.session_state.current_page + 1} of {total_pages} ({total_entries} entries)</div>', unsafe_allow_html=True)
            
            with col4:
                if st.button("Next ▶️") and st.session_state.current_page < total_pages - 1:
                    st.session_state.current_page += 1
                    st.rerun()
            
            with col5:
                if st.button("Last ⏭️") and st.session_state.current_page < total_pages - 1:
                    st.session_state.current_page = total_pages - 1
                    st.rerun()

            # Display entries for current page
            start_idx = st.session_state.current_page * entries_per_page
            end_idx = min(start_idx + entries_per_page, total_entries)
            page_entries = filtered_entries[start_idx:end_idx]

            for entry in page_entries:
                formatted_date = format_timestamp(entry["entry_date"])
                short_date = format_date_short(entry["entry_date"])
                sentiment_label = clean_unicode_text(entry.get('sentiment_label', 'NEUTRAL'))
                sentiment_score = entry.get('sentiment_score', 0.0)
                clean_entry_text = clean_unicode_text(entry.get('entry_text', ''))
                emoji = get_sentiment_emoji(sentiment_label)

                sentiment_class = f"sentiment-{sentiment_label.lower()}"

                if view_mode == 'compact':
                    # Compact view
                    preview_text = clean_entry_text[:150] + "..." if len(clean_entry_text) > 150 else clean_entry_text
                    
                    st.markdown(f"""
                    <div class="entry-card {sentiment_class} compact-entry">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                            <strong style="font-size: 0.9rem;">{short_date}</strong>
                            <span style="font-size: 0.8rem; opacity: 0.7;">{sentiment_label} {emoji}</span>
                        </div>
                        <div class="entry-preview">{preview_text}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Detailed view
                    st.markdown(f"""
                    <div class="entry-card {sentiment_class}">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                            <strong>{formatted_date}</strong>
                            <span>{sentiment_label} {emoji} ({sentiment_score:.2f})</span>
                        </div>
                        <div>{clean_entry_text}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # --- Weekly Summary Section ---
        st.markdown("---")
        st.subheader("📊 Weekly Summary")
        
        if st.button("Generate Weekly Summary", type="secondary"):
        
            analyzer = SentimentIntensityAnalyzer()
            
            def classify_sentiment(text):
                score = analyzer.polarity_scores(text)
                if score['compound'] >= 0.05:
                    return "POSITIVE"
                elif score['compound'] <= -0.05:
                    return "NEGATIVE"
                else:
                    return "NEUTRAL"

            def get_last_7_days_entries(entries):
                now = datetime.now(timezone.utc)
                seven_days_ago = now - timedelta(days=7)
                return [
                    entry for entry in entries
                    if datetime.fromisoformat(entry["entry_date"].replace("Z", "+00:00")) >= seven_days_ago
                ]

            last_7_days = get_last_7_days_entries(entries)

            if last_7_days:
                for entry in last_7_days:
                    if "sentiment_label" not in entry or not entry["sentiment_label"]:
                        entry["sentiment_label"] = classify_sentiment(entry["entry_text"])

                sentiment_counts = Counter([e["sentiment_label"].upper() for e in last_7_days])
                sentiment_stats = {
                "positive": sentiment_counts.get("POSITIVE", 0),
                "neutral": sentiment_counts.get("NEUTRAL", 0),
                "negative": sentiment_counts.get("NEGATIVE", 0),
                "total": len(last_7_days)
                }

                # Stats display
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("😊 Positive", sentiment_stats["positive"])
                with col2:
                    st.metric("😐 Neutral", sentiment_stats["neutral"])
                with col3:
                    st.metric("😔 Negative", sentiment_stats["negative"])
                with col4:
                    st.metric("📝 Total", sentiment_stats["total"])

                # Chart with dark theme
                import plotly.express as px
                chart_data = {
                    "Sentiment": list(sentiment_counts.keys()),
                    "Count": list(sentiment_counts.values())
                }
                fig = px.pie(
                    chart_data, 
                    names="Sentiment", 
                    values="Count", 
                    title="Sentiment Distribution (Last 7 Days)",
                    color_discrete_map={
                        'POSITIVE': '#4CAF50',
                        'NEGATIVE': '#F44336', 
                        'NEUTRAL': '#FF9800'
                    }
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                st.plotly_chart(fig, use_container_width=True)

                # Call Groq-powered summary
                with st.spinner("Analyzing weekly journals with AI..."):
                    summary = analyze_weekly_summary_with_groq(last_7_days, sentiment_stats)
                    st.subheader("🧘 Weekly AI Insights")
                    st.markdown(f"""
                    <div class="remedy-card">
                        {summary}
                    </div>
                    """, unsafe_allow_html=True)

            else:
                st.info("No journal entries in the last 7 days.")
                

        # --- Email Notification Settings ---
        st.markdown("---")
        st.subheader("📧 Email Reminders")

        if clean_token:
            # Get current preferences
            prefs_response = requests.get(
                "http://127.0.0.1:8000/api/journal/notification-preferences/",
                headers=headers
            )
            
            if prefs_response.status_code == 200:
                prefs_data = prefs_response.json()
                current_enabled = prefs_data.get('email_reminders_enabled', False)
                current_time = prefs_data.get('reminder_time')
                current_timezone = prefs_data.get('timezone', 'UTC')
                available_timezones = prefs_data.get('available_timezones', ['UTC'])
                
                # Display current setting
                if current_enabled and current_time:
                    st.info(f"**Current setting:** ✅ Daily reminder at {current_time} ({current_timezone})")
                else:
                    st.info("**Current setting:** ❌ Email reminders disabled")
                
                # Settings form
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Enable/disable toggle
                    email_enabled = st.checkbox(
                        "Enable daily email reminders",
                        value=current_enabled
                    )
                    
                    if email_enabled:
                        # Time picker
                        col_time, col_tz = st.columns([1, 1])
                        
                        with col_time:
                            # Parse current time or use default
                            default_hour, default_minute = (9, 0)
                            if current_time:
                                try:
                                    time_parts = current_time.split(':')
                                    default_hour = int(time_parts[0])
                                    default_minute = int(time_parts[1])
                                except:
                                    pass
                            
                            reminder_hour = st.selectbox(
                                "Hour",
                                options=list(range(24)),
                                index=default_hour,
                                format_func=lambda x: f"{x:02d}"
                            )
                            
                            reminder_minute = st.selectbox(
                                "Minute", 
                                options=[0, 15, 30, 45],
                                index=[0, 15, 30, 45].index(default_minute) if default_minute in [0, 15, 30, 45] else 0,
                                format_func=lambda x: f"{x:02d}"
                            )
                        
                        with col_tz:
                            selected_timezone = st.selectbox(
                                "Timezone",
                                options=available_timezones,
                                index=available_timezones.index(current_timezone) if current_timezone in available_timezones else 0
                            )
                        
                        # Show preview time
                        reminder_time_str = f"{reminder_hour:02d}:{reminder_minute:02d}"
                        st.write(f"**Reminder time:** {reminder_time_str} ({selected_timezone})")
                
                with col2:
                    st.markdown("<div style='margin-top: 2rem;'>", unsafe_allow_html=True)
                    if st.button("💾 Save Settings", type="primary"):
                        # Prepare data
                        update_data = {
                            'email_reminders_enabled': email_enabled,
                            'timezone': selected_timezone
                        }
                        
                        if email_enabled:
                            update_data['reminder_time'] = f"{reminder_hour:02d}:{reminder_minute:02d}"
                        
                        # Update preferences
                        update_response = requests.post(
                            "http://127.0.0.1:8000/api/journal/notification-preferences/",
                            headers=headers,
                            json=update_data
                        )
                        
                        if update_response.status_code == 200:
                            st.success("✅ Email preferences updated!")
                            st.rerun()
                        else:
                            error_data = update_response.json() if update_response.content else {}
                            error_msg = error_data.get('error', 'Failed to update preferences')
                            st.error(f"❌ {error_msg}")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Information box
                if email_enabled:
                    st.markdown("""
                    <div class="remedy-card">
                        <strong>📮 Email Reminder Info:</strong><br>
                        • You'll receive a gentle daily reminder at your chosen time<br>
                        • The email includes journaling prompts and encouragement<br>
                        • Make sure your email is set in your account settings<br>
                        • You can change or disable these settings anytime<br>
                        • Times are automatically converted to your timezone
                    </div>
                    """, unsafe_allow_html=True)
            
            else:
                st.warning("Could not load notification preferences. Please check your connection.")

        else:
            st.info("Enter your auth token above to configure email reminders.")
                                
    else:
        st.info("Enter your token above to view your entries.")