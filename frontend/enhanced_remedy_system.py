# enhanced_remedy_system.py
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import json
import os
from datetime import datetime
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class EnhancedRemedyRecommender:
    def __init__(self, remedy_bank_path="/Users/gyandeep/remedies_bank/remedy_bank.pkl"):
        """Initialize enhanced remedy recommender"""
        # Load remedy bank
        with open(remedy_bank_path, "rb") as f:
            remedy_bank = pickle.load(f)
        
        self.remedy_texts = remedy_bank["remedy_texts"]
        self.remedy_embeddings = remedy_bank["remedy_embeddings"]
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load user preferences
        self.preferences_file = "user_preferences.json"
        self.user_preferences = self._load_preferences()
        
        # Define remedy categories and their keywords
        self.remedy_categories = {
            'breathing': ['breath', 'inhale', 'exhale', 'breathing', 'air'],
            'grounding': ['grounding', '5-4-3-2-1', 'present', 'surroundings', 'sensation', 'anchor'],
            'movement': ['walk', 'stretch', 'exercise', 'dancing', 'physical', 'move'],
            'calming': ['meditation', 'calm', 'peaceful', 'relax', 'quiet', 'serene'],
            'social': ['friend', 'family', 'talk', 'share', 'compliment', 'connect'],
            'creative': ['creative', 'drawing', 'music', 'color', 'write', 'art'],
            'self_care': ['tea', 'blanket', 'treat', 'sleep', 'care', 'gentle'],
            'cognitive': ['mantra', 'remind', 'list', 'gratitude', 'letter', 'think'],
            'sensory': ['aromatherapy', 'candle', 'water', 'ice', 'touch', 'smell']
        }
        
        # Enhanced sentiment-specific remedy preferences
        self.sentiment_preferences = {
            'NEGATIVE': {
                'priority_categories': ['breathing', 'grounding', 'calming', 'self_care'],
                'boost_keywords': [
                    'breathing', 'grounding', 'temporary', 'cry', 'breaks',
                    'kindness', 'fresh air', 'water', 'present', 'meditation',
                    'safe', 'peaceful', 'mantra', 'tea', 'gentle', 'comfort'
                ],
                'avoid_keywords': ['challenge', 'push', 'intense', 'difficult'],
                'weight_multiplier': 1.5
            },
            'POSITIVE': {
                'priority_categories': ['social', 'creative', 'movement'],
                'boost_keywords': [
                    'celebrate', 'share', 'compliment', 'gratitude', 'photo',
                    'playlist', 'treat', 'anchor', 'future', 'succeed',
                    'joy', 'energy', 'accomplish', 'create', 'express'
                ],
                'avoid_keywords': ['slow', 'quiet', 'rest'],
                'weight_multiplier': 1.3
            },
            'NEUTRAL': {
                'priority_categories': ['movement', 'cognitive', 'self_care'],
                'boost_keywords': [
                    'sleep', 'walk', 'organize', 'list', 'routine',
                    'music', 'stretch', 'tea', 'creative', 'mindful', 'balance'
                ],
                'avoid_keywords': [],
                'weight_multiplier': 1.2
            }
        }
        
        # Context-specific boosting
        self.context_boosters = {
            'stress': {
                'keywords': ['stress', 'overwhelmed', 'pressure', 'deadline', 'busy'],
                'boost_remedies': ['breath', 'calm', 'relax', 'ground', 'present'],
                'multiplier': 1.4
            },
            'sadness': {
                'keywords': ['sad', 'cry', 'lonely', 'empty', 'hurt', 'loss'],
                'boost_remedies': ['comfort', 'gentle', 'warm', 'tea', 'blanket', 'kind'],
                'multiplier': 1.3
            },
            'anxiety': {
                'keywords': ['anxious', 'worry', 'panic', 'nervous', 'fear'],
                'boost_remedies': ['grounding', '5-4-3-2-1', 'breath', 'present', 'safe'],
                'multiplier': 1.5
            },
            'anger': {
                'keywords': ['angry', 'frustrated', 'irritated', 'mad', 'annoyed'],
                'boost_remedies': ['walk', 'physical', 'air', 'space', 'cool'],
                'multiplier': 1.3
            },
            'fatigue': {
                'keywords': ['tired', 'exhausted', 'drained', 'weary', 'sleepy'],
                'boost_remedies': ['gentle', 'rest', 'sleep', 'tea', 'comfort', 'easy'],
                'multiplier': 1.2
            }
        }
    
    def _load_preferences(self):
        """Load user preferences from file"""
        if os.path.exists(self.preferences_file):
            try:
                with open(self.preferences_file, 'r') as f:
                    return json.load(f)
            except:
                return {'remedy_ratings': {}, 'category_preferences': {}, 'used_remedies': {}}
        return {'remedy_ratings': {}, 'category_preferences': {}, 'used_remedies': {}}
    
    def _save_preferences(self):
        """Save user preferences to file"""
        try:
            with open(self.preferences_file, 'w') as f:
                json.dump(self.user_preferences, f)
        except:
            pass
    
    def _detect_context(self, text):
        """Enhanced context detection from journal entry"""
        text_lower = text.lower()
        detected_contexts = []
        confidence_scores = {}
        
        for context, config in self.context_boosters.items():
            matches = sum(1 for keyword in config['keywords'] if keyword in text_lower)
            if matches > 0:
                detected_contexts.append(context)
                confidence_scores[context] = matches / len(config['keywords'])
        
        return detected_contexts, confidence_scores
    
    def _apply_context_boosting(self, similarities, entry_text):
        """Apply context-based boosting to similarity scores"""
        detected_contexts, context_confidence = self._detect_context(entry_text)
        boosted_scores = similarities.copy()
        
        for context in detected_contexts:
            if context in self.context_boosters:
                config = self.context_boosters[context]
                multiplier = config['multiplier'] * context_confidence[context]
                
                for i, remedy_text in enumerate(self.remedy_texts['text']):
                    if any(keyword in remedy_text.lower() for keyword in config['boost_remedies']):
                        boosted_scores[i] *= multiplier
        
        return boosted_scores
    
    def _apply_sentiment_preferences(self, scores, sentiment):
        """Apply sentiment-specific preferences"""
        if sentiment not in self.sentiment_preferences:
            return scores
        
        prefs = self.sentiment_preferences[sentiment]
        adjusted_scores = scores.copy()
        
        for i, remedy_text in enumerate(self.remedy_texts['text']):
            remedy_lower = remedy_text.lower()
            
            # Boost preferred keywords
            boost_count = sum(1 for keyword in prefs['boost_keywords'] 
                            if keyword in remedy_lower)
            if boost_count > 0:
                adjusted_scores[i] *= (prefs['weight_multiplier'] * (1 + boost_count * 0.1))
            
            # Penalize avoided keywords for certain sentiments
            avoid_count = sum(1 for keyword in prefs.get('avoid_keywords', []) 
                            if keyword in remedy_lower)
            if avoid_count > 0:
                adjusted_scores[i] *= 0.7
        
        return adjusted_scores
    
    def _diversify_recommendations(self, remedy_indices, scores, top_n):
        """Ensure diversity in recommendation categories"""
        recommendations = []
        used_categories = set()
        remaining_indices = list(zip(remedy_indices, scores))
        remaining_indices.sort(key=lambda x: x[1], reverse=True)
        
        # First pass: Select top remedies ensuring category diversity
        for idx, score in remaining_indices:
            if len(recommendations) >= top_n:
                break
                
            remedy_text = self.remedy_texts.iloc[idx]['text']
            categories = self._categorize_remedy(remedy_text)
            
            # Check if this adds a new category or if we haven't filled quota
            if not used_categories.intersection(categories) or len(recommendations) < top_n // 2:
                recommendations.append((idx, score))
                used_categories.update(categories)
        
        # Second pass: Fill remaining slots with highest scoring
        if len(recommendations) < top_n:
            for idx, score in remaining_indices:
                if len(recommendations) >= top_n:
                    break
                if (idx, score) not in recommendations:
                    recommendations.append((idx, score))
        
        return recommendations[:top_n]
    
    def _categorize_remedy(self, remedy_text):
        """Categorize a remedy based on its text content"""
        categories = []
        text_lower = remedy_text.lower()
        
        for category, keywords in self.remedy_categories.items():
            if any(keyword in text_lower for keyword in keywords):
                categories.append(category)
        
        return categories if categories else ['general']
    
    def _track_remedy_usage(self, remedy_text):
        """Track remedy usage for personalization"""
        if 'used_remedies' not in self.user_preferences:
            self.user_preferences['used_remedies'] = {}
        
        if remedy_text not in self.user_preferences['used_remedies']:
            self.user_preferences['used_remedies'][remedy_text] = {
                'count': 0,
                'last_used': None
            }
        
        self.user_preferences['used_remedies'][remedy_text]['count'] += 1
        self.user_preferences['used_remedies'][remedy_text]['last_used'] = datetime.now().isoformat()
        self._save_preferences()
    
    def recommend_remedies(self, entry_text, sentiment='NEUTRAL', top_n=3, ensure_diversity=True):
        """Enhanced remedy recommendation with multiple improvement factors"""
        # Basic similarity scoring
        entry_embedding = self.embedder.encode([entry_text])
        base_similarities = cosine_similarity(entry_embedding, self.remedy_embeddings)[0]
        
        # Apply context boosting
        context_boosted = self._apply_context_boosting(base_similarities, entry_text)
        
        # Apply sentiment preferences
        sentiment_adjusted = self._apply_sentiment_preferences(context_boosted, sentiment)
        
        # Apply usage-based penalties to avoid repetition
        final_scores = sentiment_adjusted.copy()
        if 'used_remedies' in self.user_preferences:
            for i, remedy_text in enumerate(self.remedy_texts['text']):
                if remedy_text in self.user_preferences['used_remedies']:
                    usage_count = self.user_preferences['used_remedies'][remedy_text]['count']
                    # Gradually reduce score for frequently used remedies
                    penalty = min(0.3, usage_count * 0.05)
                    final_scores[i] *= (1 - penalty)
        
        # Get top candidates
        top_candidates = min(top_n * 4, len(final_scores))  # Get more candidates for diversity
        top_indices = final_scores.argsort()[-top_candidates:][::-1]
        top_scores = final_scores[top_indices]
        
        # Apply diversity if requested
        if ensure_diversity:
            final_recommendations = self._diversify_recommendations(top_indices, top_scores, top_n)
        else:
            final_recommendations = list(zip(top_indices[:top_n], top_scores[:top_n]))
        
        # Format recommendations with metadata
        formatted_recommendations = []
        for idx, score in final_recommendations:
            remedy_text = self.remedy_texts.iloc[idx]['text']
            categories = self._categorize_remedy(remedy_text)
            
            # Track usage
            self._track_remedy_usage(remedy_text)
            
            formatted_recommendations.append({
                'text': remedy_text,
                'score': float(score),
                'categories': categories,
                'base_similarity': float(base_similarities[idx]),
                'context_boost': float(context_boosted[idx] / base_similarities[idx]) if base_similarities[idx] > 0 else 1.0,
                'sentiment_match': sentiment in str(categories).upper()
            })
        
        return formatted_recommendations
    
    def rate_remedy(self, remedy_text, rating):
        """Rate a remedy for future personalization"""
        if 1 <= rating <= 5:
            self.user_preferences['remedy_ratings'][remedy_text] = rating
            self._save_preferences()
            return True
        return False

# Enhanced Sentiment Analyzer for better sentiment detection
class EnhancedSentimentAnalyzer:
    def __init__(self):
        """Initialize enhanced sentiment analyzer"""
        try:
            self.transformer_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
        except:
            self.transformer_analyzer = pipeline("sentiment-analysis", return_all_scores=True)
        
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Mental health specific keywords for better classification
        self.emotion_keywords = {
            'strong_negative': ['hopeless', 'devastated', 'crushed', 'worthless', 'suicidal'],
            'moderate_negative': ['sad', 'upset', 'disappointed', 'frustrated', 'worried'],
            'mild_negative': ['tired', 'bored', 'confused', 'uncertain'],
            'mild_positive': ['okay', 'fine', 'decent', 'satisfied'],
            'moderate_positive': ['good', 'happy', 'pleased', 'content', 'grateful'],
            'strong_positive': ['amazing', 'fantastic', 'thrilled', 'ecstatic', 'accomplished']
        }
    
    def analyze_sentiment(self, text, return_detailed=False):
        """Analyze sentiment with enhanced accuracy for mental health context"""
        # Get transformer results
        try:
            transformer_results = self.transformer_analyzer(text)
            if isinstance(transformer_results[0], dict):
                transformer_scores = {r['label']: r['score'] for r in transformer_results}
            else:
                transformer_scores = {'POSITIVE': 0.33, 'NEGATIVE': 0.33, 'NEUTRAL': 0.34}
        except:
            transformer_scores = {'POSITIVE': 0.33, 'NEGATIVE': 0.33, 'NEUTRAL': 0.34}
        
        # Get VADER results
        vader_result = self.vader_analyzer.polarity_scores(text)
        
        # Apply keyword-based adjustments
        text_lower = text.lower()
        keyword_adjustment = 0
        
        for emotion_level, keywords in self.emotion_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            if matches > 0:
                if 'strong_negative' in emotion_level:
                    keyword_adjustment = -0.3
                elif 'moderate_negative' in emotion_level:
                    keyword_adjustment = -0.2
                elif 'mild_negative' in emotion_level:
                    keyword_adjustment = -0.1
                elif 'mild_positive' in emotion_level:
                    keyword_adjustment = 0.1
                elif 'moderate_positive' in emotion_level:
                    keyword_adjustment = 0.2
                elif 'strong_positive' in emotion_level:
                    keyword_adjustment = 0.3
                break
        
        # Combine scores
        final_compound = vader_result['compound'] + keyword_adjustment
        
        # Determine final sentiment
        if final_compound >= 0.1:
            sentiment = 'POSITIVE'
            confidence = min(abs(final_compound), 0.95)
        elif final_compound <= -0.1:
            sentiment = 'NEGATIVE'
            confidence = min(abs(final_compound), 0.95)
        else:
            sentiment = 'NEUTRAL'
            confidence = 0.6
        
        if return_detailed:
            return {
                'sentiment': sentiment,
                'confidence': confidence,
                'compound_score': final_compound,
                'vader_scores': vader_result,
                'transformer_scores': transformer_scores,
                'keyword_adjustment': keyword_adjustment
            }
        
        return sentiment, confidence

# Convenience function for easy integration
def get_enhanced_recommendations(entry_text, sentiment='NEUTRAL', top_n=3):
    """Convenience function to get enhanced recommendations"""
    recommender = EnhancedRemedyRecommender()
    analyzer = EnhancedSentimentAnalyzer()
    
    # If sentiment not provided, analyze it
    if sentiment == 'NEUTRAL' and entry_text.strip():
        detected_sentiment, confidence = analyzer.analyze_sentiment(entry_text)
        if confidence > 0.7:  # Use detected sentiment only if confident
            sentiment = detected_sentiment
    
    recommendations = recommender.recommend_remedies(
        entry_text=entry_text,
        sentiment=sentiment,
        top_n=top_n,
        ensure_diversity=True
    )
    
    return recommendations, sentiment