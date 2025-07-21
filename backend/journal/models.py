from django.db import models
from django.contrib.auth.models import User

class JournalEntry(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    entry_date = models.DateTimeField(auto_now_add=True)
    entry_text = models.TextField()
    sentiment_score = models.FloatField(null=True, blank=True)
    sentiment_label = models.CharField(max_length=20, null=True, blank=True)

    def __str__(self):
        return f"{self.user.username} - {self.entry_date.strftime('%Y-%m-%d')}"



class Conversation(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    title = models.CharField(max_length=255, default="New Conversation")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.title} - {self.user.username}"
    
    class Meta:
        ordering = ['-updated_at']

class Message(models.Model):
    ROLE_CHOICES = (
        ('user', 'User'),
        ('assistant', 'Assistant'),
    )
    
    conversation = models.ForeignKey(Conversation, related_name='messages', on_delete=models.CASCADE)
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.role}: {self.content[:30]}..."
    
    class Meta:
        ordering = ['created_at']
        

# Add this new model for user preferences
class UserNotificationPreference(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    email_reminders_enabled = models.BooleanField(default=False)
    reminder_time = models.TimeField(null=True, blank=True, help_text="Time to send daily reminder (24-hour format)")
    timezone = models.CharField(max_length=50, default='UTC', help_text="User's timezone")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        if self.email_reminders_enabled and self.reminder_time:
            return f"{self.user.username} - {self.reminder_time.strftime('%H:%M')}"
        return f"{self.user.username} - Disabled"