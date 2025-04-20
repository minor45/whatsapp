import streamlit as st
import preprocessor, helper
import matplotlib.pyplot as plt
import seaborn as sns
import io

st.sidebar.title("Whatsapp Chat Analyzer")

# Upload file
uploaded_file = st.sidebar.file_uploader("Choose a file")

if uploaded_file is not None:
    # Add a debug section
    with st.expander("Debug Information"):
        st.write("### File Information")
        st.write(f"File name: {uploaded_file.name}")
        st.write(f"File size: {uploaded_file.size} bytes")
        
        # Read raw file
        bytes_data = uploaded_file.getvalue()
        
        # Try multiple encodings
        encodings = ["utf-8", "latin-1", "utf-16", "cp1252"]
        data = None
        
        for encoding in encodings:
            try:
                data = bytes_data.decode(encoding)
                st.success(f"Successfully decoded with {encoding}")
                
                # Show sample of raw data
                st.write("### Raw Data Sample (first 500 chars)")
                st.text_area("Raw data", data[:500], height=200)
                
                # Check for common format indicators
                if "WhatsApp" in data[:200]:
                    st.write("✅ WhatsApp text detected")
                
                lines = data.split('\n')
                st.write(f"Total lines in file: {len(lines)}")
                
                # Show first 5 lines
                st.write("### First 5 lines:")
                for i, line in enumerate(lines[:5]):
                    st.code(line, language=None)
                
                break
            except UnicodeDecodeError:
                continue
        
        if data is None:
            st.error("Could not decode the file with any standard encoding.")
            st.stop()
    
    # Process data
    if st.button("Process Chat Data"):
        # Reset the file position
        uploaded_file.seek(0)
        
        # Read the data again
        bytes_data = uploaded_file.getvalue()
        try:
            data = bytes_data.decode("utf-8")
        except UnicodeDecodeError:
            data = bytes_data.decode("latin-1")
        
        # Process the data
        with st.spinner("Processing..."):
            df = preprocessor.preprocess(data)
            
            if df.empty:
                st.error("No messages found in the chat data. Please check the file format.")
                st.stop()
            
            st.success(f"Successfully processed {len(df)} messages!")
            
            # Show sample data
            st.write("### Sample Processed Data")
            st.dataframe(df.head())
            
            # Extract unique users
            user_list = df['user'].unique().tolist()
            user_list = [user for user in user_list if user != "System"]
            user_list.sort()
            user_list.insert(0, "Overall")
            
            # Sidebar selection
            selected_user = st.selectbox("Select User for Analysis", user_list)
            
            # Store the dataframe in session state so it persists
            st.session_state['df'] = df
            st.session_state['selected_user'] = selected_user

    # Add this section to check if data has been processed
    if 'df' in st.session_state:
        df = st.session_state['df']
        
        # Update selected user if user changes selection
        selected_user = st.selectbox("Select User for Analysis", 
                                   ["Overall"] + sorted([user for user in df['user'].unique() if user != "System"]),
                                   key="user_select")
        
        if st.button("Show Analysis"):
            # Calculate stats
            num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
            
            # Display stats
            st.title("Top Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.header("Total Messages")
                st.title(num_messages)
            with col2:
                st.header("Total Words")
                st.title(words)
            with col3:
                st.header("Media Shared")
                st.title(num_media_messages)
            with col4:
                st.header("Links Shared")
                st.title(num_links)
            
            # Monthly timeline
            st.title("Monthly Timeline")
            timeline = helper.monthly_timeline(selected_user, df)
            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
            
            # Daily timeline
            st.title("Daily Timeline")
            daily_timeline = helper.daily_timeline(selected_user, df)
            fig, ax = plt.subplots()
            ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
            
            # Activity map
            st.title('Activity Map')
            col1, col2 = st.columns(2)
            
            with col1:
                st.header("Most Busy Day")
                busy_day = helper.week_activity_map(selected_user, df)
                fig, ax = plt.subplots()
                ax.bar(busy_day.index, busy_day.values, color='purple')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            
            with col2:
                st.header("Most Busy Month")
                busy_month = helper.month_activity_map(selected_user, df)
                fig, ax = plt.subplots()
                ax.bar(busy_month.index, busy_month.values, color='orange')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            
            # Weekly activity heatmap
            st.title("Weekly Activity Heatmap")
            user_heatmap = helper.activity_heatmap(selected_user, df)
            fig, ax = plt.subplots()
            ax = sns.heatmap(user_heatmap)
            st.pyplot(fig)
            
            # Find busiest users in the group (only for overall selection)
            if selected_user == 'Overall':
                st.title('Most Active Users')
                busy_users, busy_percent = helper.most_busy_users(df)
                fig, ax = plt.subplots()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    ax.bar(busy_users.index, busy_users.values, color='red')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)
                with col2:
                    st.dataframe(busy_percent)
            
            # WordCloud
            st.title("Word Cloud")
            df_wc = helper.create_wordcloud(selected_user, df)
            fig, ax = plt.subplots()
            ax.imshow(df_wc)
            st.pyplot(fig)
            
            # Most common words
            st.title('Most Common Words')
            most_common_df = helper.most_common_words(selected_user, df)
            
            fig, ax = plt.subplots()
            ax.barh(most_common_df[0], most_common_df[1])
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
            
            # Emoji analysis
            st.title("Emoji Analysis")
            emoji_df = helper.emoji_helper(selected_user, df)
            
            if not emoji_df.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.dataframe(emoji_df)
                with col2:
                    fig, ax = plt.subplots()
                    ax.pie(emoji_df[1].head(), labels=emoji_df[0].head(), autopct="%0.2f")
                    st.pyplot(fig)
            else:
                st.write("No emojis used in the selected messages.")
else:
    st.info("Please upload a WhatsApp chat file to begin analysis.")