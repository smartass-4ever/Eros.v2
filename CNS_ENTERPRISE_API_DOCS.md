# CNS Enterprise API Documentation

## Overview

The CNS (Cognitive Neural System) Enterprise API provides advanced emotional intelligence capabilities for your digital products. Our API enables:

- **Emotional Understanding** - Detect emotions with 90% confidence
- **Adaptive Personality** - Responses adjust warmth based on emotional context
- **Context-Aware Engagement** - Story-specific questions and empathetic dialogue
- **Multi-Tenant Architecture** - Dedicated instances for Pro/Enterprise clients

## Quick Start

### 1. Get Your API Key

Contact our sales team to get your API key. Keys are available in three tiers:

| Tier | Rate Limit | Features | Instance Type |
|------|------------|----------|---------------|
| **Free** | 1,000/hour, 10,000/day | Basic emotion, simple responses | Shared |
| **Pro** | 10,000/hour, 100,000/day | Memory, personality adaptation | Dedicated |
| **Enterprise** | Unlimited | All features, custom config, priority support | Dedicated |

### 2. Make Your First Request

```bash
curl -X POST https://api.cns-emotional.ai/api/v1/message \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I'\''m feeling anxious about my presentation tomorrow",
    "user_id": "user_123",
    "user_name": "Sarah"
  }'
```

**Response:**
```json
{
  "response": "what's making you nervous about it?",
  "processing_time": 1.2,
  "emotional_analysis": {
    "emotion": {
      "emotion": "anxiety",
      "valence": -0.7,
      "arousal": 0.8,
      "confidence": 0.9
    },
    "valence": -0.7,
    "confidence": 0.9
  },
  "metadata": {
    "user_id": "user_123",
    "tier": "pro",
    "instance_type": "dedicated"
  }
}
```

## Authentication

All API requests (except `/health`) require authentication using an API key.

**Include your API key in the `Authorization` header:**

```
Authorization: Bearer YOUR_API_KEY
```

**Invalid or missing API keys return 401 Unauthorized:**

```json
{
  "error": "Invalid API key",
  "message": "The provided API key is invalid or has been revoked",
  "docs": "https://docs.cns-api.com/authentication"
}
```

## Rate Limiting

Rate limits are enforced per API key based on your tier:

### Rate Limit Headers

Every response includes rate limit information:

```
X-RateLimit-Limit-Hour: 10000
X-RateLimit-Remaining-Hour: 9847
X-RateLimit-Limit-Day: 100000
X-RateLimit-Remaining-Day: 95234
X-API-Tier: pro
```

### Exceeding Rate Limits

When you exceed your rate limit, you'll receive a `429 Too Many Requests` response:

```json
{
  "error": "Rate limit exceeded",
  "message": "You have exceeded your hourly rate limit of 10000 requests",
  "tier": "pro",
  "retry_after": 3600,
  "upgrade_info": "Upgrade to Enterprise for unlimited requests"
}
```

## API Endpoints

### POST /api/v1/message

Process a message with CNS emotional intelligence.

**Request Body:**

```json
{
  "message": "User's message text",
  "user_id": "unique_user_identifier",
  "user_name": "User Display Name",
  "context": {
    "conversation_history": [],
    "warmth_override": 0.8
  }
}
```

**Parameters:**

- `message` (required): The message to process
- `user_id` (optional): Unique identifier for the user (default: "anonymous")
- `user_name` (optional): Display name for the user (default: "User")
- `context` (optional): Additional context for processing

**Response:**

```json
{
  "response": "AI-generated empathetic response",
  "processing_time": 1.2,
  "emotional_analysis": {
    "emotion": {
      "emotion": "anxiety",
      "valence": -0.7,
      "arousal": 0.8,
      "confidence": 0.9
    },
    "valence": -0.7,
    "confidence": 0.9
  },
  "metadata": {
    "user_id": "user_123",
    "tier": "pro",
    "instance_type": "dedicated"
  }
}
```

### GET /api/v1/usage

Get usage statistics for your API key.

**Query Parameters:**

- `hours` (optional): Number of hours to look back (default: 24)

**Response:**

```json
{
  "key_id": "key_abc123",
  "client_name": "Your Company",
  "tier": "pro",
  "period_hours": 24,
  "total_requests": 1523,
  "successful_requests": 1501,
  "failed_requests": 22,
  "success_rate": 98.6,
  "total_tokens": 45690,
  "avg_processing_time": 1.15,
  "endpoints": {
    "/api/v1/message": 1523
  },
  "rate_limit_status": {
    "hour": {
      "used": 47,
      "limit": 10000,
      "remaining": 9953,
      "exceeded": false
    },
    "day": {
      "used": 1523,
      "limit": 100000,
      "remaining": 98477,
      "exceeded": false
    }
  }
}
```

### GET /api/v1/limits

Check current rate limit status.

**Response:**

```json
{
  "key_id": "key_abc123",
  "tier": "pro",
  "hour": {
    "used": 47,
    "limit": 10000,
    "remaining": 9953,
    "exceeded": false
  },
  "day": {
    "used": 1523,
    "limit": 100000,
    "remaining": 98477,
    "exceeded": false
  },
  "is_blocked": false
}
```

### GET /api/v1/info

Get information about your API key and capabilities.

**Response:**

```json
{
  "key_id": "key_abc123",
  "client_name": "Your Company",
  "tier": "pro",
  "features": ["basic_emotion", "simple_responses", "memory", "personality_adaptation"],
  "rate_limits": {
    "hour": 10000,
    "day": 100000
  },
  "instance_type": "dedicated",
  "created_at": 1698765432.0,
  "last_used": 1698865432.0
}
```

## Code Examples

### Python

```python
import requests

API_KEY = "your_api_key_here"
API_URL = "https://api.cns-emotional.ai/api/v1/message"

def process_message(message, user_id="user_123", user_name="User"):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "message": message,
        "user_id": user_id,
        "user_name": user_name
    }
    
    response = requests.post(API_URL, json=payload, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 429:
        print("Rate limit exceeded. Retry after:", response.headers.get('Retry-After'))
    else:
        print("Error:", response.json())
    
    return None

# Example usage
result = process_message("I'm feeling anxious about my job interview")
print(result['response'])
print(f"Detected emotion: {result['emotional_analysis']['emotion']['emotion']}")
```

### JavaScript (Node.js)

```javascript
const axios = require('axios');

const API_KEY = 'your_api_key_here';
const API_URL = 'https://api.cns-emotional.ai/api/v1/message';

async function processMessage(message, userId = 'user_123', userName = 'User') {
  try {
    const response = await axios.post(API_URL, {
      message: message,
      user_id: userId,
      user_name: userName
    }, {
      headers: {
        'Authorization': `Bearer ${API_KEY}`,
        'Content-Type': 'application/json'
      }
    });
    
    return response.data;
  } catch (error) {
    if (error.response?.status === 429) {
      console.log('Rate limit exceeded. Retry after:', error.response.headers['retry-after']);
    } else {
      console.error('Error:', error.response?.data);
    }
    return null;
  }
}

// Example usage
(async () => {
  const result = await processMessage("I'm feeling anxious about my job interview");
  console.log(result.response);
  console.log('Detected emotion:', result.emotional_analysis.emotion.emotion);
})();
```

### JavaScript (Browser/React)

```javascript
const API_KEY = 'your_api_key_here';
const API_URL = 'https://api.cns-emotional.ai/api/v1/message';

async function processMessage(message, userId, userName) {
  const response = await fetch(API_URL, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${API_KEY}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      message: message,
      user_id: userId,
      user_name: userName
    })
  });
  
  const data = await response.json();
  
  if (response.ok) {
    return data;
  } else if (response.status === 429) {
    throw new Error('Rate limit exceeded');
  } else {
    throw new Error(data.error);
  }
}

// React component example
function ChatBot() {
  const [message, setMessage] = useState('');
  const [response, setResponse] = useState('');
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    const result = await processMessage(message, 'user_123', 'User');
    setResponse(result.response);
  };
  
  return (
    <div>
      <input value={message} onChange={(e) => setMessage(e.target.value)} />
      <button onClick={handleSubmit}>Send</button>
      <p>AI: {response}</p>
    </div>
  );
}
```

### cURL

```bash
# Basic request
curl -X POST https://api.cns-emotional.ai/api/v1/message \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I'\''m feeling overwhelmed with work",
    "user_id": "user_123",
    "user_name": "Sarah"
  }'

# Check usage stats
curl -X GET 'https://api.cns-emotional.ai/api/v1/usage?hours=24' \
  -H "Authorization: Bearer YOUR_API_KEY"

# Check rate limits
curl -X GET https://api.cns-emotional.ai/api/v1/limits \
  -H "Authorization: Bearer YOUR_API_KEY"
```

## Integration Patterns

### Customer Service Chatbot

```python
class EmotionalChatbot:
    def __init__(self, api_key):
        self.api_key = api_key
        self.api_url = "https://api.cns-emotional.ai/api/v1/message"
    
    def handle_customer_message(self, message, customer_id):
        response = requests.post(
            self.api_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "message": message,
                "user_id": customer_id,
                "user_name": "Customer"
            }
        )
        
        data = response.json()
        
        # Log emotional state for analytics
        self.log_emotion(customer_id, data['emotional_analysis'])
        
        # Escalate if high negative emotion
        if data['emotional_analysis']['valence'] < -0.8:
            self.escalate_to_human(customer_id)
        
        return data['response']
```

### Mental Health App

```python
class TherapyCompanion:
    def __init__(self, api_key):
        self.api_key = api_key
        self.api_url = "https://api.cns-emotional.ai/api/v1/message"
    
    def process_journal_entry(self, entry, user_id):
        response = requests.post(
            self.api_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"message": entry, "user_id": user_id}
        )
        
        data = response.json()
        
        # Track emotional trends
        emotion_data = data['emotional_analysis']['emotion']
        self.save_mood_entry(user_id, emotion_data)
        
        # Provide empathetic response
        return {
            'response': data['response'],
            'mood': emotion_data['emotion'],
            'valence': emotion_data['valence']
        }
```

### Social Robot Integration

```python
class EmotionalRobot:
    def __init__(self, api_key):
        self.api_key = api_key
        self.api_url = "https://api.cns-emotional.ai/api/v1/message"
    
    def interact_with_user(self, user_speech, user_id):
        # Process with CNS
        response = requests.post(
            self.api_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"message": user_speech, "user_id": user_id}
        )
        
        data = response.json()
        
        # Adjust robot behavior based on emotion
        emotion = data['emotional_analysis']['emotion']
        
        if emotion['valence'] < -0.5:
            self.set_facial_expression('concerned')
            self.adjust_voice_tone('soft', 'slow')
        elif emotion['valence'] > 0.5:
            self.set_facial_expression('happy')
            self.adjust_voice_tone('cheerful', 'normal')
        
        # Speak response
        self.speak(data['response'])
```

## Error Handling

### Common Errors

| Status Code | Error | Description |
|-------------|-------|-------------|
| 400 | Bad Request | Invalid request body or missing required fields |
| 401 | Unauthorized | Invalid or missing API key |
| 429 | Rate Limit Exceeded | Too many requests |
| 500 | Internal Server Error | Server-side error |

### Best Practices

1. **Always check status codes** before processing responses
2. **Implement retry logic** for rate limit errors
3. **Cache responses** when appropriate to reduce API calls
4. **Handle network errors** gracefully
5. **Log API errors** for debugging

```python
import time

def call_cns_api_with_retry(message, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, 
                headers={"Authorization": f"Bearer {API_KEY}"},
                json={"message": message}
            )
            
            if response.status_code == 200:
                return response.json()
            
            elif response.status_code == 429:
                # Rate limit - wait and retry
                retry_after = int(response.headers.get('Retry-After', 60))
                time.sleep(retry_after)
                continue
            
            else:
                # Other error
                print(f"API error {response.status_code}: {response.json()}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Network error: {e}")
            time.sleep(2 ** attempt)  # Exponential backoff
    
    return None
```

## Support

- **Documentation**: https://docs.cns-api.com
- **Email**: support@cns-emotional.ai
- **Slack**: Join our [developer community](https://slack.cns-api.com)

## Pricing

Upgrade to higher tiers for more requests and features:

- **Free**: $0/month - 10K requests/day
- **Pro**: $99/month - 100K requests/day + dedicated instance
- **Enterprise**: Custom pricing - Unlimited requests + custom config + priority support

Contact sales@cns-emotional.ai for Enterprise pricing.
