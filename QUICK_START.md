# CNS Enterprise API - Quick Start Guide

Get started with the CNS (Cognitive Neural System) Enterprise API in less than 5 minutes!

## ⚡ Instant Demo Access

### Demo API Key (Free Tier)

A public demo key is available for immediate testing:

**Note:** The actual demo API key is generated and stored in `DEMO_API_KEY.txt`. Run `python setup_demo_environment.py` to create or retrieve it.

**Demo Tier Limits:**
- ✅ 1,000 requests per hour
- ✅ 10,000 requests per day
- ⚠️ Shared CNS instance (not for production)
- ⚠️ Basic emotional intelligence features only

## 🚀 Your First Request

### Using cURL

```bash
curl -X POST http://localhost:8000/api/v1/message \
  -H 'Authorization: Bearer YOUR_DEMO_API_KEY' \
  -H 'Content-Type: application/json' \
  -d '{
    "message": "I am feeling anxious about my presentation tomorrow",
    "user_id": "user_123",
    "user_name": "Alex"
  }'
```

### Using Python

```python
import requests

API_KEY = "YOUR_DEMO_API_KEY"
API_URL = "http://localhost:8000/api/v1/message"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

data = {
    "message": "I am feeling anxious about my presentation tomorrow",
    "user_id": "user_123",
    "user_name": "Alex"
}

response = requests.post(API_URL, headers=headers, json=data)
result = response.json()

print(f"CNS Response: {result['response']}")
print(f"Emotion Detected: {result['emotion']}")
print(f"Warmth Level: {result['warmth']}")
```

### Using JavaScript (Node.js)

```javascript
const fetch = require('node-fetch');

const API_KEY = 'YOUR_DEMO_API_KEY';
const API_URL = 'http://localhost:8000/api/v1/message';

async function askCNS(message, userId, userName) {
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
  
  const result = await response.json();
  
  console.log('CNS Response:', result.response);
  console.log('Emotion:', result.emotion);
  console.log('Warmth:', result.warmth);
}

askCNS('I am feeling anxious about my presentation tomorrow', 'user_123', 'Alex');
```

## 📋 API Response Format

```json
{
  "response": "I can sense the anxiety about your presentation... that's really understandable. Want to talk about what's making you most nervous?",
  "emotion": {
    "primary": "anxiety",
    "confidence": 0.89,
    "valence": -0.4,
    "arousal": 0.6
  },
  "warmth": 0.85,
  "personality_state": {
    "openness": 0.8,
    "conscientiousness": 0.7,
    "extraversion": 0.6,
    "agreeableness": 0.9,
    "neuroticism": 0.3
  },
  "processing_time_ms": 145,
  "tier": "free",
  "rate_limit_remaining": {
    "hour": 999,
    "day": 9999
  }
}
```

## 🎯 Common Use Cases

### 1. Mental Health Chatbot

```python
def emotional_support_bot(user_message, user_id):
    response = requests.post(
        "http://localhost:8000/api/v1/message",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "message": user_message,
            "user_id": user_id,
            "user_name": "User",
            "context": {
                "warmth_override": 1.0  # Maximum empathy
            }
        }
    )
    return response.json()
```

### 2. Customer Service Agent

```python
def customer_service(customer_query, customer_id, company_name):
    response = requests.post(
        "http://localhost:8000/api/v1/message",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "message": customer_query,
            "user_id": customer_id,
            "user_name": company_name,
            "context": {
                "warmth_override": 0.7,  # Professional but warm
                "conversation_history": []
            }
        }
    )
    return response.json()
```

### 3. Educational Tutor

```python
def educational_tutor(student_question, student_id):
    response = requests.post(
        "http://localhost:8000/api/v1/message",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "message": student_question,
            "user_id": student_id,
            "user_name": "Student",
            "context": {
                "warmth_override": 0.8,  # Encouraging and supportive
            }
        }
    )
    return response.json()
```

## 📊 Monitoring Your Usage

### Check Rate Limits

```bash
curl -X GET http://localhost:8000/api/v1/limits \
  -H 'Authorization: Bearer YOUR_DEMO_API_KEY'
```

### View Usage Statistics

```bash
curl -X GET http://localhost:8000/api/v1/usage?hours=24 \
  -H 'Authorization: Bearer YOUR_DEMO_API_KEY'
```

### View System Metrics

```bash
curl -X GET http://localhost:8000/api/v1/metrics?hours=1 \
  -H 'Authorization: Bearer YOUR_DEMO_API_KEY'
```

## 🔐 Upgrade to Production

The demo key is perfect for testing, but for production use, you'll need a dedicated API key:

### Free Tier (Forever Free)
- 1,000 requests/hour
- 10,000 requests/day
- Shared instance
- Basic features
- **Cost: $0/month**

### Pro Tier
- 10,000 requests/hour  
- 100,000 requests/day
- Dedicated CNS instance
- Full emotional intelligence
- Memory & personality adaptation
- **Cost: $99/month + $0.05/1K requests**

### Enterprise Tier
- Unlimited requests
- Dedicated CNS instance
- Custom configuration
- Priority support
- 99.9% SLA
- **Cost: $999/month + $0.02/1K requests**

### Create Your Own API Key

Contact the admin to create a production API key:

```bash
curl -X POST http://localhost:8000/admin/keys/create \
  -H 'Authorization: Bearer ADMIN_API_KEY' \
  -H 'Content-Type: application/json' \
  -d '{
    "client_name": "Your Company",
    "client_email": "billing@yourcompany.com",
    "tier": "pro",
    "notes": "Production key for customer support chatbot"
  }'
```

## 📚 Additional Resources

- **Full API Documentation:** [CNS_ENTERPRISE_API_DOCS.md](CNS_ENTERPRISE_API_DOCS.md)
- **Terms of Service:** [TERMS_OF_SERVICE.md](TERMS_OF_SERVICE.md)
- **Privacy Policy:** [PRIVACY_POLICY.md](PRIVACY_POLICY.md)

## 🆘 Troubleshooting

### Common Issues

**401 Unauthorized**
- Check that your API key is correct
- Ensure you're using `Authorization: Bearer YOUR_KEY` format
- Verify the key hasn't been revoked

**429 Rate Limit Exceeded**
- You've hit the hourly or daily limit
- Upgrade to Pro or Enterprise for higher limits
- Wait for the rate limit window to reset

**500 Internal Server Error**
- Check the CNS Enterprise API Server is running
- View server logs for detailed error information
- Contact support if issue persists

## 💬 Support

- **Documentation:** Check the full docs for detailed information
- **Issues:** Report bugs or request features
- **Email:** support@cns-api.com

---

**Ready to build emotionally intelligent applications? Start coding!** 🚀
