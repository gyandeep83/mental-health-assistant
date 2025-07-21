# sentiment_analyzer.py
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
import json
import os
from datetime import datetime

class EnhancedSentimentAnalyzer:
    def __init__(self):
        """Initialize the enhanced sentiment analyzer with multiple models"""
        try:
            # Primary: Mental health focused model (fallback to general if not available)
            self.primary_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
        except:
            # Fallback to default model
            self.primary_analyzer = pipeline("sentiment-analysis", return_all_scores=True)
        
        # Secondary: VADER for informal text
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Load user corrections for learning
        self.corrections_file = "user_corrections.json"
        self.user_corrections = self._load_corrections()
        
        # Mental health keywords for context
        self.mental_health_keywords = {
            'anxiety': ['anxious', 'worried', 'panic', 'nervous', 'stress', 'overwhelmed'],
            'depression': ['sad', 'empty', 'hopeless', 'worthless', 'tired', 'exhausted'],
            'anger': ['angry', 'frustrated', 'irritated', 'annoyed', 'mad', 'furious'],
            'joy': ['happy', 'excited', 'joyful', 'grateful', 'accomplished', 'proud'],
            'calm': ['peaceful', 'relaxed', 'content', 'serene', 'balanced', 'centered']
        }
    
    def _load_corrections(self):
        """Load user corrections from file"""
        if os.path.exists(self.corrections_file):
            try:
                with open(self.corrections_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def _save_correction(self, text, predicted, actual):
        """Save user correction for future learning"""
        correction = {
            'text': text,
            'predicted': predicted,
            'actual': actual,
            'timestamp': datetime.now().isoformat()
        }
        self.user_corrections.append(correction)
        
        # Keep only last 1000 corrections
        if len(self.user_corrections) > 1000:
            self.user_corrections = self.user_corrections[-1000:]
        
        try:
            with open(self.corrections_file, 'w') as f:
                json.dump(self.user_corrections, f)
        except:
            pass  # Fail silently if can't save
    
    def _detect_mental_health_context(self, text):
        """Detect mental health context from keywords"""
        text_lower = text.lower()
        context_scores = {}
        
        for emotion, keywords in self.mental_health_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                context_scores[emotion] = score
        
        return context_scores
    
    def _apply_corrections_bias(self, text, predicted_sentiment):
        """Apply learned bias from user corrections"""
        if not self.user_corrections:
            return predicted_sentiment, 1.0
        
        # Simple similarity check with past corrections
        similar_corrections = []
        text_words = set(text.lower().split())
        
        for correction in self.user_corrections[-100:]:  # Check last 100
            correction_words = set(correction['text'].lower().split())
            overlap = len(text_words.intersection(correction_words))
            if overlap >= 2:  # At least 2 common words
                similar_corrections.append(correction)
        
        if similar_corrections:
            # If most similar corrections disagree with prediction, lower confidence
            disagreements = sum(1 for c in similar_corrections 
                             if c['predicted'] != c['actual'])
            if disagreements > len(similar_corrections) / 2:
                return predicted_sentiment, 0.3  # Lower confidence
        
        return predicted_sentiment, 1.0
    
    def analyze_sentiment(self, text):
        """Main sentiment analysis with multiple approaches"""
        # Get transformer analysis
        try:
            transformer_results = self.primary_analyzer(text)
            transformer_scores = {
                result['label'].upper(): result['score'] 
                for result in transformer_results
            }
        except:
            transformer_scores = {'POSITIVE': 0.33, 'NEGATIVE': 0.33, 'NEUTRAL': 0.34}
        
        # Get VADER analysis
        vader_scores = self.vader_analyzer.polarity_scores(text)
        
        # Detect mental health context
        context = self._detect_mental_health_context(text)
        
        # Combine scores with weights
        weight_transformer = 0.6
        weight_vader = 0.4
        
        # Normalize transformer scores (handle different label formats)
        if 'POSITIVE' in transformer_scores:
            pos_score = transformer_scores['POSITIVE']
            neg_score = transformer_scores.get('NEGATIVE', 0)
            neu_score = transformer_scores.get('NEUTRAL', 0)
        elif 'LABEL_2' in transformer_scores:  # Some models use LABEL_0, LABEL_1, LABEL_2
            pos_score = transformer_scores.get('LABEL_2', 0)
            neg_score = transformer_scores.get('LABEL_0', 0) 
            neu_score = transformer_scores.get('LABEL_1', 0)
        else:
            # Fallback
            pos_score = neg_score = neu_score = 0.33
        
        # Calculate combined scores
        final_positive = (weight_transformer * pos_score + 
                         weight_vader * vader_scores['pos'])
        final_negative = (weight_transformer * neg_score + 
                         weight_vader * vader_scores['neg'])
        final_neutral = (weight_transformer * neu_score + 
                        weight_vader * vader_scores['neu'])
        
        # Apply context adjustments
        if 'depression' in context or 'anxiety' in context:
            final_negative *= 1.2  # Boost negative for mental health keywords
        elif 'joy' in context or 'calm' in context:
            final_positive *= 1.2  # Boost positive for positive keywords
        
        # Determine final sentiment
        scores = {
            'positive': final_positive,
            'negative': final_negative,
            'neutral': final_neutral
        }
        
        max_sentiment = max(scores.items(), key=lambda x: x[1])
        predicted_sentiment = max_sentiment[0].upper()
        base_confidence = max_sentiment[1]
        
        # Apply user corrections bias
        adjusted_sentiment, correction_factor = self._apply_corrections_bias(
            text, predicted_sentiment
        )
        final_confidence = base_confidence * correction_factor
        
        # Set minimum confidence threshold
        if final_confidence < 0.4:
            predicted_sentiment = "NEUTRAL"
            final_confidence = 0.4
        
        return {
            'sentiment': predicted_sentiment,
            'confidence': min(final_confidence, 1.0),
            'scores': scores,
            'context': context,
            'method': 'hybrid_enhanced'
        }
    
    def add_user_correction(self, text, predicted, actual):
        """Add user correction to improve future predictions"""
        self._save_correction(text, predicted, actual)
        return f"Thank you! Learning from your feedback to improve future predictions."