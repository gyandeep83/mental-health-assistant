from django.urls import path
from .views import (
    JournalEntryListCreate,
    JournalEntryDetail,
    notification_preferences,
)

urlpatterns = [
    path('entries/', JournalEntryListCreate.as_view(), name='journal-entry-list-create'),
    path('entries/<int:pk>/', JournalEntryDetail.as_view(), name='journal-entry-detail'),
    path('notification-preferences/', notification_preferences, name='notification-preferences'),
]
