from django.shortcuts import render

# Create your views here.

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from django.shortcuts import get_object_or_404
from journal.models import Conversation, Message
from chatbot.utils import process_blended_response
from langchain.chains import LLMChain

class ConversationListView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        """Get all conversations for the current user"""
        conversations = Conversation.objects.filter(user=request.user)
        data = [
            {
                'id': conv.id,
                'title': conv.title,
                'created_at': conv.created_at,
                'updated_at': conv.updated_at
            }
            for conv in conversations
        ]
        return Response(data)
    
    def post(self, request):
        """Create a new conversation"""
        conversation = Conversation.objects.create(
            user=request.user,
            title=request.data.get('title', 'New Conversation')
        )
        return Response({
            'id': conversation.id,
            'title': conversation.title,
            'created_at': conversation.created_at,
            'updated_at': conversation.updated_at
        }, status=status.HTTP_201_CREATED)

class ConversationDetailView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]
    
    def get(self, request, conversation_id):
        """Get a specific conversation with all messages"""
        conversation = get_object_or_404(Conversation, id=conversation_id, user=request.user)
        messages = conversation.messages.all()
        
        data = {
            'id': conversation.id,
            'title': conversation.title,
            'created_at': conversation.created_at,
            'updated_at': conversation.updated_at,
            'messages': [
                {
                    'id': msg.id,
                    'role': msg.role,
                    'content': msg.content,
                    'created_at': msg.created_at
                }
                for msg in messages
            ]
        }
        return Response(data)
    
    def patch(self, request, conversation_id):
        """Update conversation title"""
        conversation = get_object_or_404(Conversation, id=conversation_id, user=request.user)
        if 'title' in request.data:
            conversation.title = request.data['title']
            conversation.save()
        
        return Response({
            'id': conversation.id,
            'title': conversation.title,
            'created_at': conversation.created_at,
            'updated_at': conversation.updated_at
        })
    
    def delete(self, request, conversation_id):
        """Delete a conversation"""
        conversation = get_object_or_404(Conversation, id=conversation_id, user=request.user)
        conversation.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

class MessageView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]
    
    def post(self, request, conversation_id):
        """Add a new message to a conversation and get AI response"""
        conversation = get_object_or_404(Conversation, id=conversation_id, user=request.user)
        user_input = request.data.get('content')
        
        if not user_input:
            return Response({'error': 'Message content is required'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Save user message
        user_message = Message.objects.create(
            conversation=conversation,
            role='user',
            content=user_input
        )
        
        # Get conversation history in the format expected by process_blended_response
        # Fix: Replace negative indexing with proper Django ORM query
        user_messages = conversation.messages.filter(role='user').order_by('created_at')
        assistant_messages = conversation.messages.filter(role='assistant').order_by('created_at')
        
        # Get all user messages except the one we just created
        user_message_count = user_messages.count()
        if user_message_count > 1:  # Only if we have previous messages (excluding the one just created)
            previous_user_messages = user_messages[:user_message_count-1]
            chat_history = [
                {"user": user_msg.content, "ai": ai_msg.content}
                for user_msg, ai_msg in zip(previous_user_messages, assistant_messages)
            ]
        else:
            chat_history = []
        
        # Process the response using the imported medibot function
        try:
            response, source_docs = process_blended_response(user_input, chat_history)
            
            # Format source documents if available
            if source_docs:
                formatted_sources = "\n\n".join(
                    f"**Source {i+1} (Page: {doc.metadata.get('page', 'N/A')}):**\n{doc.page_content.strip()}"
                    for i, doc in enumerate(source_docs)
                )
                full_response = f"{response.strip()}\n\n---\n\n**Sources:**\n{formatted_sources}"
            else:
                full_response = response.strip()
            
            # Save assistant message
            assistant_message = Message.objects.create(
                conversation=conversation,
                role='assistant',
                content=full_response
            )
            
            # Update conversation timestamp
            conversation.save()  # This will update the updated_at field
            
            # Generate a title for new conversations based on first message
            if conversation.title == "New Conversation" and conversation.messages.count() <= 3:
                # Use the LLM to generate a short title
                from langchain.prompts import PromptTemplate
                from .utils import load_llm
                
                llm = load_llm()
                title_prompt = PromptTemplate(
                    input_variables=["message"],
                    template="Generate a very short title (3-5 words) for a conversation that starts with: {message}"
                )
                title_chain = LLMChain(llm=llm, prompt=title_prompt)
                new_title = title_chain.run(user_input).strip()
                
                conversation.title = new_title
                conversation.save()
            
            return Response({
                'user_message': {
                    'id': user_message.id,
                    'role': user_message.role,
                    'content': user_message.content,
                    'created_at': user_message.created_at
                },
                'assistant_message': {
                    'id': assistant_message.id,
                    'role': assistant_message.role,
                    'content': assistant_message.content,
                    'created_at': assistant_message.created_at
                },
                'conversation_title': conversation.title
            })
            
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)