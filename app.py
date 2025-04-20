import re
import pandas as pd
import numpy as np
import emoji
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
from textblob import TextBlob
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
import pyLDAvis.sklearn
import os
import warnings
warnings.filterwarnings('ignore')

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class WhatsAppChatAnalyzer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words('english'))
        self.additional_stopwords = {'media', 'omitted', 'message', 'deleted', 'hey', 'hi', 'hello', 'ok', 'okay'}
        self.stopwords.update(self.additional_stopwords)
        self.emoji_pattern = re.compile("["
                                        u"\U0001F600-\U0001F64F"  # emoticons
                                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                        u"\U0001F700-\U0001F77F"  # alchemical symbols
                                        u"\U0001F780-\U0001F7FF"  # Geometric Shapes
                                        u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                                        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                                        u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                                        u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                                        u"\U00002702-\U000027B0"  # Dingbats
                                        u"\U000024C2-\U0001F251" 
                                        "]+", flags=re.UNICODE)
        
    def preprocess_chat(self, file_path):
        """
        Preprocess WhatsApp chat export file into a structured DataFrame
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as file:
                content = file.read()
                
        # Extract date, time, user, and message using regex
        pattern = r'(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?::\d{2})?\s?(?:AM|PM|am|pm)?)\s-\s([^:]+):\s(.*)'
        matches = re.findall(pattern, content, re.MULTILINE)
        
        if not matches:
            # Try alternative date format
            pattern = r'(\d{1,2}\s[A-Za-z]+\s\d{4},\s\d{1,2}:\d{2}(?::\d{2})?\s?(?:AM|PM|am|pm)?)\s-\s([^:]+):\s(.*)'
            matches = re.findall(pattern, content, re.MULTILINE)
        
        # Create DataFrame
        data = []
        for match in matches:
            date_str, user, message = match
            
            # Try different date formats
            try:
                # Format: MM/DD/YY, H:MM:SS AM/PM
                date = datetime.strptime(date_str, '%m/%d/%y, %I:%M:%S %p')
            except ValueError:
                try:
                    # Format: MM/DD/YY, H:MM AM/PM
                    date = datetime.strptime(date_str, '%m/%d/%y, %I:%M %p')
                except ValueError:
                    try:
                        # Format: MM/DD/YYYY, H:MM:SS
                        date = datetime.strptime(date_str, '%m/%d/%Y, %H:%M:%S')
                    except ValueError:
                        try:
                            # Format: DD Month YYYY, H:MM - British/European format
                            date = datetime.strptime(date_str, '%d %B %Y, %H:%M')
                        except ValueError:
                            # If all fail, use a placeholder date and log the issue
                            date = datetime(2000, 1, 1)
                            print(f"Could not parse date: {date_str}")
            
            user = user.strip()
            message = message.strip()
            
            data.append({
                'date': date,
                'user': user,
                'message': message,
                'hour': date.hour,
                'day_name': date.strftime('%A'),
                'day': date.day,
                'month': date.month,
                'year': date.year,
                'minute': date.minute
            })
            
        df = pd.DataFrame(data)
        
        # Add more derived features
        df['letter_count'] = df['message'].apply(lambda s: len(s))
        df['word_count'] = df['message'].apply(lambda s: len(s.split()))
        df['message_length_category'] = pd.cut(
            df['letter_count'],
            bins=[0, 10, 50, 200, 1000, 10000],
            labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long']
        )
        
        # Extract media messages
        df['is_media'] = df['message'].str.contains('media omitted')
        
        # Extract URLs
        url_pattern = r'(https?://\S+)'
        df['contains_url'] = df['message'].str.contains(url_pattern)
        
        # Extract emojis
        df['emoji_count'] = df['message'].apply(self._count_emojis)
        df['contains_emoji'] = df['emoji_count'] > 0
        
        # Sentiment analysis
        df['sentiment_score'] = df['message'].apply(self._get_sentiment)
        df['sentiment'] = df['sentiment_score'].apply(
            lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral')
        )
        
        # Clean and preprocess text for NLP tasks
        df['clean_message'] = df['message'].apply(self._clean_message)
        
        return df
    
    def _count_emojis(self, text):
        """Count emojis in a text"""
        return len([c for c in text if c in emoji.EMOJI_DATA])
    
    def _get_sentiment(self, text):
        """Get sentiment score using TextBlob"""
        try:
            return TextBlob(text).sentiment.polarity
        except:
            return 0
    
    def _clean_message(self, text):
        """Clean and preprocess text for NLP tasks"""
        if isinstance(text, str):
            # Remove URLs
            text = re.sub(r'https?://\S+', '', text)
            
            # Remove emojis
            text = self.emoji_pattern.sub(r'', text)
            
            # Remove special characters and digits
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            
            # Convert to lowercase
            text = text.lower()
            
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove stopwords and lemmatize
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stopwords and len(token) > 2]
            
            return ' '.join(tokens)
        return ''
    
    def get_basic_stats(self, df):
        """
        Generate basic statistics with user-friendly explanations
        """
        total_messages = len(df)
        total_participants = df['user'].nunique()
        participants_list = ", ".join(df['user'].unique())
        total_days = (df['date'].max() - df['date'].min()).days + 1
        avg_messages_per_day = total_messages / total_days
        
        total_words = df['word_count'].sum()
        avg_words_per_message = total_words / total_messages
        
        media_count = df['is_media'].sum()
        media_percentage = (media_count / total_messages) * 100
        
        url_count = df['contains_url'].sum()
        
        emoji_messages = df['contains_emoji'].sum()
        emoji_percentage = (emoji_messages / total_messages) * 100
        
        most_active_user = df['user'].value_counts().idxmax()
        most_active_count = df['user'].value_counts().max()
        most_active_percentage = (most_active_count / total_messages) * 100
        
        most_active_day = df['day_name'].value_counts().idxmax()
        most_active_hour = df['hour'].value_counts().idxmax()
        
        stats = {
            'Total Messages': total_messages,
            'Total Participants': total_participants,
            'Participants': participants_list,
            'Duration (days)': total_days,
            'Start Date': df['date'].min().strftime('%B %d, %Y'),
            'End Date': df['date'].max().strftime('%B %d, %Y'),
            'Average Messages Per Day': round(avg_messages_per_day, 1),
            'Total Words': total_words,
            'Average Words Per Message': round(avg_words_per_message, 1),
            'Media Messages': f"{media_count} ({media_percentage:.1f}%)",
            'Messages with URLs': url_count,
            'Messages with Emojis': f"{emoji_messages} ({emoji_percentage:.1f}%)",
            'Most Active User': f"{most_active_user} ({most_active_count} messages, {most_active_percentage:.1f}%)",
            'Most Active Day': most_active_day,
            'Most Active Hour': f"{most_active_hour}:00 - {most_active_hour+1}:00"
        }
        
        friendly_summary = f"""
        📊 <b>Chat Overview:</b><br>
        This conversation includes <b>{total_messages} messages</b> exchanged between <b>{total_participants} participants</b> over <b>{total_days} days</b>.<br>
        The chat started on <b>{df['date'].min().strftime('%B %d, %Y')}</b> and the most recent message was on <b>{df['date'].max().strftime('%B %d, %Y')}</b>.<br>
        
        💬 <b>Messaging Patterns:</b><br>
        • On average, the group exchanged <b>{avg_messages_per_day:.1f} messages per day</b>
        • <b>{most_active_user}</b> was the most active, sending <b>{most_active_percentage:.1f}%</b> of all messages
        • The group is most active on <b>{most_active_day}s</b> around <b>{most_active_hour}:00</b>
        
        📱 <b>Content Breakdown:</b><br>
        • <b>{media_percentage:.1f}%</b> of messages contained media (photos, videos, etc.)
        • <b>{emoji_percentage:.1f}%</b> of messages included emojis
        • The average message contained <b>{avg_words_per_message:.1f} words</b>
        """
        
        return stats, friendly_summary
    
    def generate_user_activity_insights(self, df):
        """
        Generate insights about user activity patterns
        """
        # Messages by user
        user_message_counts = df['user'].value_counts()
        
        # User message length averages
        user_avg_length = df.groupby('user')['word_count'].mean().sort_values(ascending=False)
        
        # User media sharing
        user_media = df[df['is_media']].groupby('user').size()
        if not user_media.empty:
            user_media_percentage = (user_media / df.groupby('user').size() * 100).sort_values(ascending=False)
        else:
            user_media_percentage = pd.Series(dtype=float)
        
        # User emoji usage
        user_emoji = df.groupby('user')['emoji_count'].sum().sort_values(ascending=False)
        
        # User activity by hour
        user_hour_activity = pd.crosstab(df['user'], df['hour'])
        
        # User activity by day
        user_day_activity = pd.crosstab(df['user'], df['day_name'])
        
        # User sentiment
        user_sentiment = df.groupby('user')['sentiment_score'].mean().sort_values(ascending=False)
        
        # Response time analysis (if possible)
        df_sorted = df.sort_values('date')
        df_sorted['next_date'] = df_sorted['date'].shift(-1)
        df_sorted['next_user'] = df_sorted['user'].shift(-1)
        df_sorted['is_response'] = df_sorted['user'] != df_sorted['next_user']
        df_sorted['response_time_seconds'] = (df_sorted['next_date'] - df_sorted['date']).dt.total_seconds()
        
        # Average response time when someone else responds
        response_times = df_sorted[df_sorted['is_response']]
        if not response_times.empty:
            avg_response_times = response_times.groupby('next_user')['response_time_seconds'].mean()
            avg_response_times = avg_response_times.apply(lambda x: str(timedelta(seconds=x)).split('.')[0])
        else:
            avg_response_times = pd.Series(dtype=str)
        
        insights = {
            'message_counts': user_message_counts,
            'avg_message_length': user_avg_length,
            'media_sharing': user_media_percentage,
            'emoji_usage': user_emoji,
            'hour_activity': user_hour_activity,
            'day_activity': user_day_activity,
            'sentiment': user_sentiment,
            'response_times': avg_response_times
        }
        
        # Generate natural language insights
        top_user = user_message_counts.idxmax()
        most_positive = user_sentiment.idxmax() if not user_sentiment.empty else "Unknown"
        most_negative = user_sentiment.idxmin() if not user_sentiment.empty else "Unknown"
        most_talkative = user_avg_length.idxmax() if not user_avg_length.empty else "Unknown"
        quickest_responder = avg_response_times.idxmin() if not avg_response_times.empty else "Unknown"
        
        friendly_summary = f"""
        👥 <b>Participant Insights:</b><br>
        • <b>{top_user}</b> is the most active, sending {user_message_counts[top_user]} messages
        • <b>{most_talkative}</b> typically writes the longest messages (avg {user_avg_length[most_talkative]:.1f} words)
        • <b>{most_positive}</b> tends to send the most positive messages
        • <b>{most_negative}</b> tends to send the most critical messages
        
        ⏱️ <b>Conversation Dynamics:</b><br>
        """
        
        if not avg_response_times.empty:
            friendly_summary += f"• <b>{quickest_responder}</b> responds fastest to others (avg {avg_response_times[quickest_responder]})"
        
        return insights, friendly_summary
    
    def analyze_content(self, df):
        """
        Analyze the content of messages
        """
        # Word frequency analysis (excluding stopwords)
        all_words = ' '.join(df['clean_message'].dropna())
        words_list = all_words.split()
        word_freq = Counter(words_list).most_common(20)
        
        # Emoji analysis
        all_emojis = []
        for message in df['message']:
            if isinstance(message, str):
                all_emojis.extend([c for c in message if c in emoji.EMOJI_DATA])
        emoji_freq = Counter(all_emojis).most_common(10)
        
        # Topic modeling
        topic_df = df[df['clean_message'] != ''].copy()
        if len(topic_df) > 10:  # Only perform if we have enough data
            vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000)
            dtm = vectorizer.fit_transform(topic_df['clean_message'])
            
            # Number of topics
            num_topics = min(5, len(topic_df) // 5)
            if num_topics < 2:
                num_topics = 2
                
            lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
            lda.fit(dtm)
            
            # Get topic keywords
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            for topic_idx, topic in enumerate(lda.components_):
                top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
                topics.append(top_words)
        else:
            topics = []
        
        # Message type analysis
        message_types = {
            'Text only': len(df[~df['is_media'] & ~df['contains_emoji'] & ~df['contains_url']]),
            'With emoji': len(df[df['contains_emoji']]),
            'With media': len(df[df['is_media']]),
            'With URL': len(df[df['contains_url']])
        }
        
        # Sentiment over time
        df_by_date = df.groupby(df['date'].dt.date)['sentiment_score'].mean().reset_index()
        
        # Generate natural language summary
        if word_freq:
            top_words = [word for word, count in word_freq[:5]]
            top_words_str = ", ".join(top_words)
        else:
            top_words_str = "No significant words found"
            
        if emoji_freq:
            top_emojis = [emoji_char for emoji_char, count in emoji_freq[:3]]
            top_emojis_str = " ".join(top_emojis)
        else:
            top_emojis_str = "No emojis used"
        
        # Summarize topics
        topic_summaries = []
        if topics:
            for i, topic_words in enumerate(topics):
                topic_summaries.append(f"Topic {i+1}: {', '.join(topic_words[:5])}")
        
        # Overall sentiment
        avg_sentiment = df['sentiment_score'].mean()
        sentiment_description = "positive" if avg_sentiment > 0.05 else ("negative" if avg_sentiment < -0.05 else "neutral")
        
        friendly_summary = f"""
        📝 <b>Content Analysis:</b><br>
        • Most common words: <b>{top_words_str}</b>
        • Most used emojis: <b>{top_emojis_str}</b>
        • The overall tone of conversation is <b>{sentiment_description}</b>
        
        🏷️ <b>Conversation Topics:</b><br>
        """
        
        if topic_summaries:
            for topic in topic_summaries:
                friendly_summary += f"• {topic}<br>"
        else:
            friendly_summary += "• Not enough data to identify distinct topics<br>"
        
        content_analysis = {
            'word_freq': word_freq,
            'emoji_freq': emoji_freq,
            'topics': topics,
            'message_types': message_types,
            'sentiment_over_time': df_by_date
        }
        
        return content_analysis, friendly_summary
    
    def create_visualizations(self, df, user_insights, content_analysis):
        """
        Create visualization figures
        """
        visualizations = {}
        
        # 1. Message activity over time
        messages_by_date = df.groupby(df['date'].dt.date).size()
        
        fig_timeline = px.line(
            x=messages_by_date.index, 
            y=messages_by_date.values,
            labels={'x': 'Date', 'y': 'Number of Messages'},
            title='Daily Message Activity'
        )
        visualizations['daily_activity'] = fig_timeline
        
        # 2. Message distribution by user
        fig_user_dist = px.bar(
            x=user_insights['message_counts'].index,
            y=user_insights['message_counts'].values,
            labels={'x': 'User', 'y': 'Number of Messages'},
            title='Messages Sent by Each User',
            color=user_insights['message_counts'].values
        )
        visualizations['user_distribution'] = fig_user_dist
        
        # 3. Activity heatmap by hour and day
        activity_by_hour_day = pd.crosstab(df['day_name'], df['hour'])
        
        # Ensure days are in correct order
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        activity_by_hour_day = activity_by_hour_day.reindex(day_order)
        
        fig_heatmap = px.imshow(
            activity_by_hour_day, 
            labels=dict(x="Hour of Day", y="Day of Week", color="Message Count"),
            x=activity_by_hour_day.columns,
            y=activity_by_hour_day.index,
            title="Message Activity by Day and Hour",
            color_continuous_scale="Viridis"
        )
        visualizations['activity_heatmap'] = fig_heatmap
        
        # 4. Sentiment analysis by user
        sentiment_by_user = df.groupby('user')['sentiment_score'].mean().reset_index()
        sentiment_by_user = sentiment_by_user.sort_values('sentiment_score')
        
        fig_sentiment = px.bar(
            sentiment_by_user,
            x='user', y='sentiment_score',
            labels={'user': 'User', 'sentiment_score': 'Average Sentiment Score'},
            title='Average Message Sentiment by User',
            color='sentiment_score',
            color_continuous_scale=['red', 'yellow', 'green']
        )
        visualizations['user_sentiment'] = fig_sentiment
        
        # 5. Message types breakdown
        message_types = pd.Series(content_analysis['message_types'])
        
        fig_types = px.pie(
            values=message_types.values,
            names=message_types.index,
            title='Message Types Distribution',
            hole=0.4
        )
        visualizations['message_types'] = fig_types
        
        # 6. Word Cloud
        if content_analysis['word_freq']:
            word_freq_dict = dict(content_analysis['word_freq'])
            
            # Create Word Cloud
            wordcloud = WordCloud(
                width=800, height=400,
                background_color='white',
                max_words=100,
                contour_width=3,
                contour_color='steelblue'
            ).generate_from_frequencies(word_freq_dict)
            
            # Convert to figure
            fig_wordcloud = plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Most Common Words')
            visualizations['wordcloud'] = fig_wordcloud
        
        # 7. Emoji usage
        if content_analysis['emoji_freq']:
            emoji_df = pd.DataFrame(content_analysis['emoji_freq'], columns=['emoji', 'count'])
            
            fig_emoji = px.bar(
                emoji_df,
                x='emoji', y='count',
                labels={'emoji': 'Emoji', 'count': 'Frequency'},
                title='Top Emojis Used',
                color='count',
                color_continuous_scale='Viridis'
            )
            visualizations['emoji_usage'] = fig_emoji
        
        # 8. Sentiment over time
        if not content_analysis['sentiment_over_time'].empty:
            fig_sentiment_time = px.line(
                content_analysis['sentiment_over_time'],
                x='date', y='sentiment_score',
                labels={'date': 'Date', 'sentiment_score': 'Average Sentiment'},
                title='Sentiment Trend Over Time',
                markers=True
            )
            # Add horizontal line at neutral sentiment
            fig_sentiment_time.add_hline(y=0, line_dash="dash", line_color="gray")
            visualizations['sentiment_trend'] = fig_sentiment_time
        
        return visualizations

    def create_report(self, df):
        """
        Create a comprehensive analysis report with visualizations and user-friendly explanations
        """
        # Calculate and prepare all analyses
        stats, basic_summary = self.get_basic_stats(df)
        user_insights, user_summary = self.generate_user_activity_insights(df)
        content_analysis, content_summary = self.analyze_content(df)
        visualizations = self.create_visualizations(df, user_insights, content_analysis)
        
        # Combine all summaries
        full_summary = f"""
        # WhatsApp Chat Analysis Report
        
        {basic_summary}
        
        {user_summary}
        
        {content_summary}
        
        ## Detailed Statistics
        
        ### Basic Chat Statistics
        """
        
        # Add detailed stats
        for key, value in stats.items():
            full_summary += f"- **{key}:** {value}\n"
        
        # Return all components
        report = {
            'dataframe': df,
            'statistics': stats,
            'user_insights': user_insights,
            'content_analysis': content_analysis,
            'visualizations': visualizations,
            'summary': full_summary
        }
        
        return report

def create_streamlit_app():
    """
    Create a Streamlit app for the WhatsApp Chat Analyzer
    """
    st.set_page_config(page_title="WhatsApp Chat Analyzer", page_icon="💬", layout="wide")
    
    st.title("📱 WhatsApp Chat Analyzer")
    st.markdown("""
    Upload your WhatsApp chat export and get detailed insights about your conversations!
    
    ### How to export your WhatsApp chat:
    1. Open the chat in WhatsApp
    2. Tap the three dots (⋮) > More > Export chat
    3. Choose 'Without Media' for faster analysis
    4. Save the file and upload it here
    """)
    
    analyzer = WhatsAppChatAnalyzer()
    
    uploaded_file = st.file_uploader("Choose a WhatsApp chat export file (.txt)", type=["txt"])
    
    if uploaded_file is not None:
        # Save the file temporarily
        with open("temp_chat.txt", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.spinner("Analyzing your chat... This may take a moment."):
            try:
                # Process the chat
                df = analyzer.preprocess_chat("temp_chat.txt")
                
                # Check if we have valid data
                if len(df) == 0:
                    st.error("No messages found in the file. Please check if this is a valid WhatsApp chat export.")
                else:
                    # Create full report
                    report = analyzer.create_report(df)
                    
                    # Display summary
                    st.subheader("📊 Chat Overview")
                    st.markdown(report['summary'].replace('<b>', '**').replace('</b>', '**').replace('<br>', '\n'), unsafe_allow_html=False)
                    
                    # Create tabs for different sections
                    tab1, tab2, tab3 = st.tabs(["📈 Message Activity", "👥 User Analysis", "💬 Content Analysis"])
                    
                    with tab1:
                        st.subheader("Message Activity Over Time")
                        st.plotly_chart(report['visualizations']['daily_activity'], use_container_width=True)
                        
                        st.subheader("Activity by Day and Hour")
                        st.markdown("This heatmap shows when the chat is most active. Darker colors indicate more messages.")
                        st.plotly_chart(report['visualizations']['activity_heatmap'], use_container_width=True)
                        
                        # Get most active day and hour
                        day_totals = df['day_name'].value_counts()
                        hour_totals = df['hour'].value_counts()
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Most Active Day", day_totals.idxmax(), f"{day_totals.max()} messages")
                        with col2:
                            st.metric("Most Active Hour", f"{hour_totals.idxmax()}:00", f"{hour_totals.max()} messages")
                    
                    with tab2:
                        st.subheader("User Participation")
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.plotly_chart(report['visualizations']['user_distribution'], use_container_width=True)
                        
                        with col2:
                            # Get top and bottom users
                            message_counts = report['user_insights']['message_counts']
                            total_messages = message_counts.sum()
                            
                            st.markdown("### User Participation")
                            for user, count in message_counts.items():
                                percentage = (count / total_messages) * 100
                                st.markdown(f"**{user}**: {count} messages ({percentage:.1f}%)")
                        
                        st.subheader("Message Sentiment by User")
                        st.markdown("This shows whether users tend to send positive or negative messages. Higher values are more positive.")
                        st.plotly_chart(report['visualizations']['user_sentiment'], use_container_width=True)
                        
                        # Average message length by user
                        avg_lengths = report['user_insights']['avg_message_length'].reset_index()
                        avg_lengths.columns = ['User', 'Average Words']
                        
                        st.subheader("Communication Style")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### Average Message Length")
                            st.dataframe(avg_lengths, hide_index=True)
                        
                        with col2:
                            # Response times
                            if not report['user_insights']['response_times'].empty:
                                st.markdown("### Average Response Time")
                                response_df = report['user_insights']['response_times'].reset_index()
                                response_df.columns = ['User', 'Response Time']
                                st.dataframe(response_df, hide_index=True)
                    
                    with tab3:
                        st.subheader("Message Content Analysis")
                        
                        # Word frequency and word cloud
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            if 'wordcloud' in report['visualizations']:
                                st.pyplot(report['visualizations']['wordcloud'])
                            else:
                                st.info("Not enough text data to generate a word cloud.")
                        
                        with col2:
                            st.markdown("### Most Common Words")
                            if report['content_analysis']['word_freq']:
                                word_df = pd.DataFrame(report['content_analysis']['word_freq'], columns=['Word', 'Count'])
                                st.dataframe(word_df.head(10), hide_index=True)
                            else:
                                st.info("No significant words found in the analysis.")
                        
                        # Emoji analysis
                        st.subheader("Emoji Analysis")
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            if 'emoji_usage' in report['visualizations']:
                                st.plotly_chart(report['visualizations']['emoji_usage'], use_container_width=True)
                            else:
                                st.info("No emojis found in the chat.")
                        
                        with col2:
                            st.markdown("### Top Emojis")
                            if report['content_analysis']['emoji_freq']:
                                emoji_df = pd.DataFrame(report['content_analysis']['emoji_freq'], columns=['Emoji', 'Count'])
                                st.dataframe(emoji_df.head(10), hide_index=True)
                            else:
                                st.info("No emojis found in the chat.")
                        
                        # Message types
                        st.subheader("Message Types")
                        st.plotly_chart(report['visualizations']['message_types'], use_container_width=True)
                        
                        # Sentiment over time
                        st.subheader("Sentiment Trend")
                        st.markdown("This chart shows how the overall mood of the conversation changed over time.")
                        if 'sentiment_trend' in report['visualizations']:
                            st.plotly_chart(report['visualizations']['sentiment_trend'], use_container_width=True)
                            
                            # Calculate some meaningful insights about sentiment
                            sentiment_data = report['content_analysis']['sentiment_over_time']
                            if not sentiment_data.empty:
                                most_positive_day = sentiment_data.loc[sentiment_data['sentiment_score'].idxmax()]
                                most_negative_day = sentiment_data.loc[sentiment_data['sentiment_score'].idxmin()]
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Most Positive Day", 
                                             most_positive_day['date'].strftime('%Y-%m-%d'), 
                                             f"Score: {most_positive_day['sentiment_score']:.2f}")
                                with col2:
                                    st.metric("Most Negative Day", 
                                             most_negative_day['date'].strftime('%Y-%m-%d'), 
                                             f"Score: {most_negative_day['sentiment_score']:.2f}")
                        else:
                            st.info("Not enough data to analyze sentiment trends.")
                        
                        # Topic modeling
                        st.subheader("Conversation Topics")
                        st.markdown("These are the main topics detected in your conversation, based on word patterns.")
                        
                        if report['content_analysis']['topics']:
                            topics = report['content_analysis']['topics']
                            for i, topic_words in enumerate(topics):
                                st.markdown(f"**Topic {i+1}:** {', '.join(topic_words[:10])}")
                        else:
                            st.info("Not enough data to identify distinct conversation topics.")
                    
                    # Add a fourth tab for custom insights
                    tab4 = st.tabs(["🔍 Custom Insights"])[0]
                    
                    with tab4:
                        st.subheader("Custom Insights & Data Explorer")
                        st.markdown("""
                        Here you can explore specific aspects of your chat data. Select options below to generate custom insights.
                        """)
                        
                        insight_type = st.selectbox(
                            "What would you like to analyze?",
                            ["User Activity Over Time", "Message Length Distribution", "Conversation Flow", 
                             "Response Patterns", "Individual User Profile"]
                        )
                        
                        if insight_type == "User Activity Over Time":
                            st.subheader("User Activity Timeline")
                            
                            # Allow selecting specific users
                            users = list(df['user'].unique())
                            selected_users = st.multiselect("Select users to compare", users, default=users[:min(5, len(users))])
                            
                            if selected_users:
                                # Filter data for selected users
                                filtered_df = df[df['user'].isin(selected_users)]
                                
                                # Group by user and date
                                user_activity = filtered_df.groupby(['user', pd.Grouper(key='date', freq='D')])['message'].count().reset_index()
                                
                                # Create interactive chart
                                fig = px.line(user_activity, x='date', y='message', color='user',
                                             labels={'message': 'Number of Messages', 'date': 'Date'},
                                             title='User Activity Over Time')
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Show activity patterns
                                st.subheader("Activity Patterns")
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # Activity by day of week
                                    day_activity = pd.crosstab(filtered_df['user'], filtered_df['day_name'])
                                    # Ensure correct day order
                                    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                                    day_activity = day_activity.reindex(columns=day_order)
                                    
                                    fig = px.bar(day_activity.reset_index().melt(id_vars='user', var_name='day', value_name='count'),
                                                x='day', y='count', color='user', barmode='group',
                                                labels={'count': 'Number of Messages', 'day': 'Day of Week'},
                                                title='Activity by Day of Week')
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                with col2:
                                    # Activity by hour
                                    hour_activity = pd.crosstab(filtered_df['user'], filtered_df['hour'])
                                    
                                    fig = px.line(hour_activity.reset_index().melt(id_vars='user', var_name='hour', value_name='count'),
                                                 x='hour', y='count', color='user',
                                                 labels={'count': 'Number of Messages', 'hour': 'Hour of Day'},
                                                 title='Activity by Hour of Day')
                                    st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("Please select at least one user to analyze.")
                        
                        elif insight_type == "Message Length Distribution":
                            st.subheader("Message Length Analysis")
                            
                            # Message length distribution
                            fig = px.histogram(df, x='word_count', nbins=20, color='user',
                                              marginal='box',
                                              labels={'word_count': 'Words per Message', 'count': 'Frequency'},
                                              title='Distribution of Message Lengths')
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Message length categories
                            length_categories = df['message_length_category'].value_counts().reset_index()
                            length_categories.columns = ['Category', 'Count']
                            
                            fig = px.pie(length_categories, values='Count', names='Category',
                                        title='Message Length Categories',
                                        hole=0.4)
                            
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                st.markdown("### Message Length Categories")
                                st.markdown("""
                                - **Very Short**: 0-10 characters
                                - **Short**: 11-50 characters
                                - **Medium**: 51-200 characters
                                - **Long**: 201-1000 characters
                                - **Very Long**: Over 1000 characters
                                """)
                                
                                # Average length stats
                                avg_by_user = df.groupby('user')['word_count'].agg(['mean', 'max']).reset_index()
                                avg_by_user.columns = ['User', 'Average Words', 'Longest Message']
                                st.dataframe(avg_by_user.sort_values('Average Words', ascending=False), hide_index=True)
                        
                        elif insight_type == "Conversation Flow":
                            st.subheader("Conversation Flow Analysis")
                            
                            # Message count over time (hourly)
                            hourly_messages = df.groupby(pd.Grouper(key='date', freq='H')).size().reset_index()
                            hourly_messages.columns = ['Date', 'Message Count']
                            
                            fig = px.line(hourly_messages, x='Date', y='Message Count',
                                         labels={'Message Count': 'Number of Messages', 'Date': 'Date & Time'},
                                         title='Conversation Intensity (Hourly)')
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Identify conversation bursts
                            hourly_messages['is_burst'] = hourly_messages['Message Count'] > hourly_messages['Message Count'].mean() * 2
                            bursts = hourly_messages[hourly_messages['is_burst']]
                            
                            if not bursts.empty:
                                st.subheader("Conversation Bursts")
                                st.markdown("These are periods when the chat was particularly active (more than twice the average message rate):")
                                
                                burst_data = []
                                for _, burst in bursts.iterrows():
                                    # Get messages during this burst
                                    burst_messages = df[(df['date'] >= burst['Date']) & 
                                                       (df['date'] < burst['Date'] + pd.Timedelta(hours=1))]
                                    
                                    # Get top users during burst
                                    top_users = burst_messages['user'].value_counts().head(3)
                                    top_users_str = ", ".join([f"{user} ({count})" for user, count in top_users.items()])
                                    
                                    burst_data.append({
                                        'Date': burst['Date'].strftime('%Y-%m-%d %H:%M'),
                                        'Messages': burst['Message Count'],
                                        'Duration': '1 hour',
                                        'Top Participants': top_users_str
                                    })
                                
                                st.dataframe(pd.DataFrame(burst_data), hide_index=True)
                            else:
                                st.info("No significant conversation bursts detected.")
                            
                            # Conversation gaps
                            df_sorted = df.sort_values('date')
                            df_sorted['next_message_time'] = df_sorted['date'].shift(-1)
                            df_sorted['time_gap'] = (df_sorted['next_message_time'] - df_sorted['date']).dt.total_seconds() / 3600  # hours
                            
                            # Filter for gaps > 24 hours
                            big_gaps = df_sorted[df_sorted['time_gap'] > 24].copy()
                            
                            if not big_gaps.empty:
                                st.subheader("Conversation Gaps")
                                st.markdown("These are periods when the chat was inactive for more than 24 hours:")
                                
                                big_gaps['gap_days'] = big_gaps['time_gap'] / 24
                                big_gaps['last_message'] = big_gaps['message'].str[:50] + '...'
                                big_gaps['gap_start'] = big_gaps['date'].dt.strftime('%Y-%m-%d %H:%M')
                                big_gaps['gap_end'] = big_gaps['next_message_time'].dt.strftime('%Y-%m-%d %H:%M')
                                
                                gap_data = big_gaps[['gap_start', 'gap_end', 'gap_days', 'user', 'last_message']].sort_values('gap_days', ascending=False)
                                gap_data.columns = ['Gap Start', 'Gap End', 'Days of Silence', 'Last Active User', 'Last Message']
                                
                                st.dataframe(gap_data.head(10), hide_index=True)
                            else:
                                st.info("No significant conversation gaps detected (> 24 hours).")
                        
                        elif insight_type == "Response Patterns":
                            st.subheader("Conversation Response Patterns")
                            
                            # Prepare data
                            df_sorted = df.sort_values('date').copy()
                            df_sorted['next_user'] = df_sorted['user'].shift(-1)
                            df_sorted['next_time'] = df_sorted['date'].shift(-1)
                            df_sorted['is_response'] = df_sorted['user'] != df_sorted['next_user']
                            df_sorted['response_time'] = (df_sorted['next_time'] - df_sorted['date']).dt.total_seconds() / 60  # minutes
                            
                            # Filter valid responses (under 60 minutes to exclude conversation breaks)
                            valid_responses = df_sorted[(df_sorted['is_response']) & (df_sorted['response_time'] < 60)]
                            
                            if not valid_responses.empty:
                                # Who responds to whom
                                response_pairs = valid_responses.groupby(['user', 'next_user']).size().reset_index()
                                response_pairs.columns = ['User', 'Responds To', 'Frequency']
                                response_pairs = response_pairs.sort_values('Frequency', ascending=False)
                                
                                col1, col2 = st.columns([2, 1])
                                
                                with col1:
                                    # Create a heatmap of who responds to whom
                                    response_matrix = pd.crosstab(valid_responses['user'], valid_responses['next_user'])
                                    
                                    fig = px.imshow(response_matrix,
                                                   labels=dict(x="Responded To", y="User", color="Frequency"),
                                                   title="Who Responds to Whom",
                                                   color_continuous_scale="Viridis")
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                with col2:
                                    st.markdown("### Top Response Patterns")
                                    st.dataframe(response_pairs.head(10), hide_index=True)
                                
                                # Response time analysis
                                st.subheader("Response Time Analysis")
                                
                                # Average response time by user
                                avg_response_time = valid_responses.groupby('next_user')['response_time'].mean().reset_index()
                                avg_response_time.columns = ['User', 'Avg Response Time (min)']
                                avg_response_time = avg_response_time.sort_values('Avg Response Time (min)')
                                
                                fig = px.bar(avg_response_time,
                                            x='User', y='Avg Response Time (min)',
                                            color='Avg Response Time (min)',
                                            labels={'User': 'User', 'Avg Response Time (min)': 'Average Response Time (minutes)'},
                                            title='Average Response Time by User')
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Distribution of response times
                                fig = px.histogram(valid_responses, x='response_time',
                                                 color='next_user',
                                                 nbins=30,
                                                 range_x=[0, 15],
                                                 labels={'response_time': 'Response Time (minutes)', 'count': 'Frequency'},
                                                 title='Distribution of Response Times (up to 15 minutes)')
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("Not enough data to analyze response patterns.")
                        
                        elif insight_type == "Individual User Profile":
                            st.subheader("Individual User Profile")
                            
                            # User selection
                            selected_user = st.selectbox("Select a user to analyze", df['user'].unique())
                            
                            if selected_user:
                                # Filter data for selected user
                                user_data = df[df['user'] == selected_user]
                                
                                # User statistics
                                total_messages = len(user_data)
                                avg_length = user_data['word_count'].mean()
                                media_percentage = (user_data['is_media'].sum() / total_messages) * 100
                                emoji_per_msg = user_data['emoji_count'].sum() / total_messages
                                sentiment_score = user_data['sentiment_score'].mean()
                                sentiment_label = "Positive" if sentiment_score > 0.05 else ("Negative" if sentiment_score < -0.05 else "Neutral")
                                
                                # Display stats
                                st.markdown(f"### Profile for: {selected_user}")
                                
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("Total Messages", total_messages)
                                with col2:
                                    st.metric("Avg Words/Message", f"{avg_length:.1f}")
                                with col3:
                                    st.metric("Media Messages", f"{media_percentage:.1f}%")
                                with col4:
                                    st.metric("Sentiment", sentiment_label, f"{sentiment_score:.2f}")
                                
                                # User activity patterns
                                st.subheader("Activity Patterns")
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # Activity by hour
                                    hour_counts = user_data['hour'].value_counts().sort_index()
                                    
                                    fig = px.bar(x=hour_counts.index, y=hour_counts.values,
                                               labels={'x': 'Hour of Day', 'y': 'Number of Messages'},
                                               title=f"Activity by Hour - {selected_user}")
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                with col2:
                                    # Activity by day
                                    day_counts = user_data['day_name'].value_counts()
                                    # Ensure correct day order
                                    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                                    day_counts = day_counts.reindex(day_order)
                                    
                                    fig = px.bar(x=day_counts.index, y=day_counts.values,
                                               labels={'x': 'Day of Week', 'y': 'Number of Messages'},
                                               title=f"Activity by Day - {selected_user}")
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # Word usage
                                st.subheader("Common Words and Phrases")
                                
                                # Get word frequency for this user
                                user_clean_text = ' '.join(user_data['clean_message'].dropna())
                                if user_clean_text.strip():
                                    user_words = user_clean_text.split()
                                    user_word_freq = Counter(user_words).most_common(10)
                                    
                                    word_df = pd.DataFrame(user_word_freq, columns=['Word', 'Count'])
                                    
                                    fig = px.bar(word_df, x='Word', y='Count',
                                               title=f"Most Used Words - {selected_user}")
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.info("Not enough text data to analyze word usage.")
                                
                                # Emoji usage
                                st.subheader("Emoji Usage")
                                
                                user_emojis = []
                                for message in user_data['message']:
                                    if isinstance(message, str):
                                        user_emojis.extend([c for c in message if c in emoji.EMOJI_DATA])
                                
                                if user_emojis:
                                    emoji_freq = Counter(user_emojis).most_common(10)
                                    emoji_df = pd.DataFrame(emoji_freq, columns=['Emoji', 'Count'])
                                    
                                    fig = px.bar(emoji_df, x='Emoji', y='Count',
                                               title=f"Most Used Emojis - {selected_user}")
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.info(f"{selected_user} doesn't use emojis in their messages.")
                            else:
                                st.warning("Please select a user to analyze.")
                    
                    # Add download options
                    st.subheader("Download Data")
                    
                    # Prepare CSV data
                    csv = df.to_csv(index=False).encode('utf-8')
                    
                    st.download_button(
                        label="Download Raw Data as CSV",
                        data=csv,
                        file_name="whatsapp_chat_data.csv",
                        mime="text/csv",
                    )
                    
                    # Cleanup temporary file
                    if os.path.exists("temp_chat.txt"):
                        os.remove("temp_chat.txt")
                    
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
                if os.path.exists("temp_chat.txt"):
                    os.remove("temp_chat.txt")

if __name__ == "__main__":
    create_streamlit_app()