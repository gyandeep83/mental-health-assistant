from django.urls import path
from .views import ConversationListView, ConversationDetailView, MessageView

urlpatterns = [
    path('conversations/', ConversationListView.as_view(), name='conversation-list'),
    path('conversations/<int:conversation_id>/', ConversationDetailView.as_view(), name='conversation-detail'),
    path('conversations/<int:conversation_id>/messages/', MessageView.as_view(), name='message-create'),
]