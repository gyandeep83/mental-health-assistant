from transformers import pipeline
from django.core.mail import send_mail
from django.conf import settings
from django.contrib.auth.models import User
from .models import UserNotificationPreference, JournalEntry
from datetime import datetime, timedelta, time
import pytz
import logging


sentiment_model = pipeline("sentiment-analysis")  # Loaded once at import

def get_sentiment(entry_text):
    result = sentiment_model(entry_text[:512])[0]  # avoid token limit
    label = result['label']
    score = float(result['score'])
    return score, label



logger = logging.getLogger(__name__)

def send_journal_reminder(user_email, user_name):
    """Send a simple journal reminder email"""
    
    # Determine greeting based on current time
    current_hour = datetime.now().hour
    if current_hour < 12:
        greeting = "Good morning"
        emoji = "🌅"
    elif current_hour < 17:
        greeting = "Good afternoon" 
        emoji = "☀️"
    else:
        greeting = "Good evening"
        emoji = "🌙"

    subject = f"{emoji} Time for Your Daily Journal Reflection"
    
    message = f"""Hi {user_name},

{greeting}! This is your gentle reminder to take a few minutes for journaling.

Take a moment to reflect on:
• How are you feeling right now?
• What's on your mind today?
• What are you grateful for?
• Any thoughts or experiences you'd like to capture?

Remember, even a few sentences can make a difference in your mental wellness journey.

Your mental health matters - take care of yourself! 💚

Best regards,
Mental Health Journal

---
To change your reminder time or turn off these emails, visit your journal settings.
"""

    try:
        send_mail(
            subject=subject,
            message=message,
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[user_email],
            fail_silently=False,
        )
        logger.info(f"Reminder email sent to {user_email}")
        return True
    except Exception as e:
        logger.error(f"Failed to send email to {user_email}: {str(e)}")
        return False

def send_scheduled_reminders():
    """Send reminders to users based on their individual scheduled times"""
    # Get current time in UTC
    now_utc = datetime.now(pytz.UTC)
    current_time = now_utc.time()
    
    # Get all users with reminders enabled
    preferences = UserNotificationPreference.objects.filter(
        email_reminders_enabled=True,
        reminder_time__isnull=False
    )
    
    for pref in preferences:
        user = pref.user
        if not user.email:
            continue
            
        # Convert user's local time to UTC for comparison
        try:
            user_tz = pytz.timezone(pref.timezone)
            
            # Create a datetime object for today with user's reminder time
            today = now_utc.date()
            user_reminder_datetime = user_tz.localize(
                datetime.combine(today, pref.reminder_time)
            )
            
            # Convert to UTC
            reminder_utc = user_reminder_datetime.astimezone(pytz.UTC)
            reminder_time_utc = reminder_utc.time()
            
            # Check if current time matches reminder time (within 1 minute window)
            time_diff = abs(
                (current_time.hour * 60 + current_time.minute) - 
                (reminder_time_utc.hour * 60 + reminder_time_utc.minute)
            )
            
            if time_diff <= 1:  # Within 1 minute
                send_journal_reminder(user.email, user.first_name or user.username)
                
        except Exception as e:
            logger.error(f"Error processing reminder for user {user.username}: {str(e)}")