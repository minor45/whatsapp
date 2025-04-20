import pandas as pd
from datetime import datetime
import re

def preprocess(data):
    """
    Alternative preprocessor that uses a more direct approach to parsing WhatsApp chat data.
    """
    print("=================== DEBUG INFO ===================")
    
    # Print first few lines of data for debugging
    print("First 5 lines of input data:")
    for i, line in enumerate(data.split('\n')[:5]):
        print(f"Line {i}: {line}")
    
    messages = []
    users = []
    dates = []
    is_system_msg = []
    
    # Split data into lines
    lines = data.split('\n')
    line_index = 0
    
    while line_index < len(lines):
        line = lines[line_index].strip()
        line_index += 1
        
        if not line:
            continue
        
        # Check if the line starts with a date pattern
        # This basic pattern just looks for digits/digits/digits at the start
        date_match = re.match(r'^(\d{1,2}/\d{1,2}/\d{2,4})', line)
        
        if date_match:
            # Try to separate the date/time part and the message part
            try:
                # Look for the " - " separator
                parts = line.split(" - ", 1)
                if len(parts) == 2:
                    date_time_part = parts[0].strip()
                    message_part = parts[1].strip()
                    
                    # Now try to parse the date/time part
                    # First, normalize the AM/PM format
                    date_time_part = date_time_part.replace("am", "AM").replace("pm", "PM")
                    
                    # Try to parse the message part to extract user and message
                    colon_index = message_part.find(": ")
                    
                    if colon_index > 0:
                        # Regular message with user and content
                        user = message_part[:colon_index].strip()
                        message_content = message_part[colon_index + 2:].strip()
                        system_msg = False
                    else:
                        # System message without colon
                        user = "System"
                        message_content = message_part
                        system_msg = True
                    
                    # Try different date formats
                    date_time_obj = None
                    formats = [
                        "%m/%d/%y, %I:%M %p",    # 8/24/24, 10:43 PM
                        "%d/%m/%Y, %I:%M %p",    # 26/01/2020, 4:19 PM
                        "%d/%m/%y, %I:%M %p",    # 26/01/20, 4:19 PM
                        "%m/%d/%Y, %I:%M %p",    # 8/24/2024, 10:43 PM
                        "%m/%d/%y, %H:%M",       # 24-hour format
                        "%d/%m/%Y, %H:%M",       # 24-hour format
                        "%d/%m/%y, %H:%M",       # 24-hour format
                        "%m/%d/%Y, %H:%M"        # 24-hour format
                    ]
                    
                    for fmt in formats:
                        try:
                            date_time_obj = datetime.strptime(date_time_part, fmt)
                            print(f"Success parsing date '{date_time_part}' with format '{fmt}'")
                            break
                        except ValueError:
                            # Try lowercase for AM/PM if applicable
                            if "%p" in fmt:
                                try:
                                    date_time_obj = datetime.strptime(date_time_part.lower(), fmt)
                                    print(f"Success parsing date '{date_time_part.lower()}' with format '{fmt}'")
                                    break
                                except ValueError:
                                    pass
                    
                    if date_time_obj:
                        # Successfully parsed a message
                        users.append(user)
                        dates.append(date_time_obj)
                        messages.append(message_content)
                        is_system_msg.append(system_msg)
                        continue
                    else:
                        print(f"Failed to parse date-time: {date_time_part}")
            except Exception as e:
                print(f"Error processing line: {line}")
                print(f"Error: {e}")
        
        # If we get here and we have messages, treat this line as a continuation of the previous message
        if messages:
            messages[-1] += "\n" + line
    
    print(f"\nTotal messages parsed: {len(messages)}")
    
    if not messages:
        print("⚠️ No messages were parsed successfully!")
        return pd.DataFrame()  # Empty DataFrame

    # Create DataFrame
    df = pd.DataFrame({
        'user': users, 
        'message': messages, 
        'date': dates,
        'is_system_msg': is_system_msg
    })
    
    # Create additional time-based columns
    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month_name()
    df['month_num'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df['period'] = df['hour'].apply(lambda x: f"{x}-{x+1}")

    print(f"DataFrame created with {len(df)} rows and columns: {list(df.columns)}")
    
    return df