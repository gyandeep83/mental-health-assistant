from rest_framework import generics, permissions
from .models import JournalEntry
from .serializers import JournalEntrySerializer
from .utils import get_sentiment
from rest_framework.permissions import AllowAny  # ✅ Add this  
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from .models import UserNotificationPreference
from datetime import time
import pytz
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from django.contrib.auth.models import User
from rest_framework import status

@api_view(['POST'])
@permission_classes([AllowAny])
def register(request):
    username = request.data.get('username')
    password = request.data.get('password')

    if not username or not password:
        return Response({'error': 'Username and password are required'}, status=400)

    if User.objects.filter(username=username).exists():
        return Response({'error': 'Username already exists'}, status=400)

    user = User.objects.create_user(username=username, password=password)
    return Response({'message': 'User registered successfully'}, status=201)


class JournalEntryListCreate(generics.ListCreateAPIView):
    serializer_class = JournalEntrySerializer
    permission_classes = [AllowAny]  # 🔁 Change this temporarily

    def get_queryset(self):
        return JournalEntry.objects.all().order_by('-entry_date')  # Optional: test all entries
        # Or stick with: return JournalEntry.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        entry_text = self.request.data.get("entry_text", "")
        sentiment_score, sentiment_label = get_sentiment(entry_text)
        serializer.save(user=self.request.user if self.request.user.is_authenticated else None,
                        sentiment_score=sentiment_score,
                        sentiment_label=sentiment_label)

class JournalEntryDetail(generics.RetrieveUpdateDestroyAPIView):
    serializer_class = JournalEntrySerializer
    permission_classes = [AllowAny]  # 🔁 Change this temporarily

    def get_queryset(self):
        return JournalEntry.objects.all()




@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def notification_preferences(request):
    """Get or update user's notification preferences"""
    
    # Get or create preferences
    pref, created = UserNotificationPreference.objects.get_or_create(
        user=request.user,
        defaults={
            'email_reminders_enabled': False,
            'reminder_time': None,
            'timezone': 'UTC'
        }
    )
    
    if request.method == 'GET':
        return Response({
            'email_reminders_enabled': pref.email_reminders_enabled,
            'reminder_time': pref.reminder_time.strftime('%H:%M') if pref.reminder_time else None,
            'timezone': pref.timezone,
            'available_timezones': [
                'UTC', 'US/Eastern', 'US/Central', 'US/Mountain', 'US/Pacific',
                'Europe/London', 'Europe/Paris', 'Asia/Tokyo', 'Asia/Kolkata',
                'Australia/Sydney', 'Canada/Eastern', 'Canada/Pacific'
            ]
        })
    
    elif request.method == 'POST':
        email_enabled = request.data.get('email_reminders_enabled', False)
        reminder_time_str = request.data.get('reminder_time')  # Format: "HH:MM"
        timezone_str = request.data.get('timezone', 'UTC')
        
        # Validate timezone
        try:
            pytz.timezone(timezone_str)
        except:
            return Response({
                'error': 'Invalid timezone'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Parse time if provided
        reminder_time_obj = None
        if email_enabled and reminder_time_str:
            try:
                hour, minute = map(int, reminder_time_str.split(':'))
                reminder_time_obj = time(hour, minute)
            except:
                return Response({
                    'error': 'Invalid time format. Use HH:MM (24-hour format)'
                }, status=status.HTTP_400_BAD_REQUEST)
        
        # Update preferences
        pref.email_reminders_enabled = email_enabled
        pref.reminder_time = reminder_time_obj
        pref.timezone = timezone_str
        pref.save()
        
        return Response({
            'message': 'Notification preferences updated successfully',
            'email_reminders_enabled': pref.email_reminders_enabled,
            'reminder_time': pref.reminder_time.strftime('%H:%M') if pref.reminder_time else None,
            'timezone': pref.timezone
        })