"""
Cloud Tools for Eros Agentic System
Tier 1: No-auth tools (web search, weather, news, stocks, wikipedia)
Tier 2: OAuth tools (Spotify, Google Calendar, Gmail) - uses Replit connectors
"""

import os
import json
import asyncio
import aiohttp
import base64
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
from email.mime.text import MIMEText
from concurrent.futures import ThreadPoolExecutor

from experience_bus import emit_action_experience

try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        DDGS_AVAILABLE = True
    except ImportError:
        DDGS_AVAILABLE = False
        print("[CLOUD-TOOLS] ⚠️ ddgs not available, web search will be limited")

_thread_pool = ThreadPoolExecutor(max_workers=3)


class ToolResult:
    """Result from executing a cloud tool"""
    def __init__(self, success: bool, data: Any = None, error: str = None, 
                 display_text: str = None, raw_response: Dict = None):
        self.success = success
        self.data = data
        self.error = error
        self.display_text = display_text or str(data)
        self.raw_response = raw_response
    
    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "display_text": self.display_text
        }


class CloudToolExecutor:
    """Executes Tier 1 cloud tools (no OAuth required)"""
    
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._api_keys = {
            "weather": os.environ.get("OPENWEATHER_API_KEY"),
            "news": os.environ.get("NEWS_API_KEY"),
            "stocks": os.environ.get("ALPHA_VANTAGE_API_KEY"),
        }
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def execute(self, action_type: str, params: Dict, user_id: str) -> ToolResult:
        """Execute a cloud tool and emit experience event"""
        executor_map = {
            "web_search": self.web_search,
            "check_weather": self.check_weather,
            "check_news": self.check_news,
            "check_stocks": self.check_stocks,
            "wikipedia": self.wikipedia_search,
            "calculate": self.calculate,
            "timezone": self.timezone_convert,
            "send_email": self.send_email,
        }
        
        executor = executor_map.get(action_type)
        if not executor:
            result = ToolResult(success=False, error=f"Unknown tool: {action_type}")
        else:
            try:
                result = await executor(params)
            except Exception as e:
                result = ToolResult(success=False, error=str(e))
        
        emit_action_experience(
            user_id=user_id,
            action_type=action_type,
            action_params=params,
            success=result.success,
            result=result.display_text if result.success else None,
            error=result.error
        )
        
        return result
    
    async def web_search(self, params: Dict) -> ToolResult:
        """Search the web using DuckDuckGo (real search results, free, no key)"""
        query = params.get("query", "")
        if not query:
            return ToolResult(success=False, error="No search query provided")
        
        if not DDGS_AVAILABLE:
            return ToolResult(success=False, error="Web search library not available")
        
        def do_search():
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=5))
                    return results
            except Exception as e:
                return str(e)
        
        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(_thread_pool, do_search)
            
            if isinstance(results, str):
                return ToolResult(success=False, error=f"Search error: {results}")
            
            if not results:
                return ToolResult(
                    success=False,
                    error=f"No search results found for '{query}'"
                )
            
            formatted_results = []
            for r in results[:5]:
                title = r.get('title', 'No title')
                body = r.get('body', '')[:200]
                href = r.get('href', '')
                formatted_results.append(f"• **{title}**: {body}")
            
            display_text = f"Search results for '{query}':\n\n" + "\n\n".join(formatted_results)
            
            return ToolResult(
                success=True,
                data={"query": query, "results": results, "count": len(results)},
                display_text=display_text,
                raw_response={"results": results}
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Search error: {str(e)}")
    
    async def check_weather(self, params: Dict) -> ToolResult:
        """Get weather info using wttr.in (free, no API key needed)"""
        location = params.get("location", "New York")
        
        session = await self._get_session()
        url = f"https://wttr.in/{location}?format=j1"
        
        try:
            async with session.get(url, headers={"User-Agent": "Eros-Bot/1.0"}) as response:
                if response.status == 200:
                    data = await response.json()
                    current = data.get("current_condition", [{}])[0]
                    temp_f = current.get("temp_F", "?")
                    feels_like = current.get("FeelsLikeF", "?")
                    description = current.get("weatherDesc", [{}])[0].get("value", "Unknown")
                    humidity = current.get("humidity", "?")
                    
                    return ToolResult(
                        success=True,
                        data={
                            "temp": temp_f,
                            "feels_like": feels_like,
                            "description": description,
                            "humidity": humidity,
                            "location": location
                        },
                        display_text=f"Weather in {location}: {temp_f}°F (feels like {feels_like}°F), {description}. Humidity: {humidity}%",
                        raw_response=data
                    )
                elif response.status == 404:
                    return ToolResult(success=False, error=f"Location '{location}' not found")
                else:
                    return ToolResult(success=False, error=f"Weather API error: HTTP {response.status}")
        except Exception as e:
            return ToolResult(success=False, error=f"Weather error: {str(e)}")
    
    async def check_news(self, params: Dict) -> ToolResult:
        """Get news headlines using DuckDuckGo news search (real news, free, no key)"""
        topic = params.get("topic", "technology")
        
        if not DDGS_AVAILABLE:
            return ToolResult(success=False, error="News search library not available")
        
        def do_news_search():
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.news(topic, max_results=5))
                    return results
            except Exception as e:
                return str(e)
        
        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(_thread_pool, do_news_search)
            
            if isinstance(results, str):
                return ToolResult(success=False, error=f"News error: {results}")
            
            if not results:
                return ToolResult(
                    success=False,
                    error=f"No news found for '{topic}'"
                )
            
            headlines = []
            for r in results[:5]:
                title = r.get('title', 'No title')
                source = r.get('source', 'Unknown')
                date = r.get('date', '')
                body = r.get('body', '')[:150]
                headlines.append(f"• **{title}** ({source})\n  {body}")
            
            display_text = f"Latest news about '{topic}':\n\n" + "\n\n".join(headlines)
            
            return ToolResult(
                success=True,
                data={"topic": topic, "articles": results, "count": len(results)},
                display_text=display_text,
                raw_response={"results": results}
            )
        except Exception as e:
            return ToolResult(success=False, error=f"News error: {str(e)}")
    
    async def check_stocks(self, params: Dict) -> ToolResult:
        """Get crypto prices using CoinGecko (free, no API key needed)"""
        symbol = params.get("symbol", "").upper()
        if not symbol:
            return ToolResult(success=False, error="No symbol provided")
        
        crypto_map = {
            "BTC": "bitcoin", "ETH": "ethereum", "DOGE": "dogecoin",
            "SOL": "solana", "XRP": "ripple", "ADA": "cardano",
            "DOT": "polkadot", "MATIC": "matic-network", "LINK": "chainlink",
            "AVAX": "avalanche-2", "LTC": "litecoin", "SHIB": "shiba-inu"
        }
        
        session = await self._get_session()
        
        try:
            if symbol in crypto_map:
                coin_id = crypto_map[symbol]
                url = f"https://api.coingecko.com/api/v3/simple/price"
                async with session.get(url, params={
                    "ids": coin_id,
                    "vs_currencies": "usd",
                    "include_24hr_change": "true"
                }) as response:
                    if response.status == 200:
                        data = await response.json()
                        coin_data = data.get(coin_id, {})
                        if coin_data:
                            price = coin_data.get("usd", 0)
                            change = coin_data.get("usd_24h_change", 0)
                            return ToolResult(
                                success=True,
                                data={"symbol": symbol, "price": price, "change_24h": change, "type": "crypto"},
                                display_text=f"{symbol}: ${price:,.2f} USD ({change:+.2f}% 24h)",
                                raw_response=data
                            )
                    return ToolResult(success=False, error=f"Could not fetch price for {symbol}")
            else:
                return ToolResult(
                    success=False,
                    error=f"Stock quotes require an API key. Supported crypto: {', '.join(crypto_map.keys())}"
                )
        except Exception as e:
            return ToolResult(success=False, error=f"Stock error: {str(e)}")
    
    async def wikipedia_search(self, params: Dict) -> ToolResult:
        """Search Wikipedia for information"""
        query = params.get("query", "")
        if not query:
            return ToolResult(success=False, error="No search query provided")
        
        session = await self._get_session()
        url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + query.replace(" ", "_")
        
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    extract = data.get("extract", "")
                    title = data.get("title", query)
                    
                    return ToolResult(
                        success=True,
                        data={"title": title, "extract": extract},
                        display_text=f"**{title}**\n{extract[:500]}{'...' if len(extract) > 500 else ''}",
                        raw_response=data
                    )
                elif response.status == 404:
                    return ToolResult(success=False, error=f"No Wikipedia article found for '{query}'")
                else:
                    return ToolResult(success=False, error=f"Wikipedia error: HTTP {response.status}")
        except Exception as e:
            return ToolResult(success=False, error=f"Wikipedia error: {str(e)}")
    
    async def calculate(self, params: Dict) -> ToolResult:
        """Perform mathematical calculations safely"""
        expression = params.get("expression", "")
        if not expression:
            return ToolResult(success=False, error="No expression provided")
        
        allowed_chars = set("0123456789+-*/().^ ")
        if not all(c in allowed_chars for c in expression):
            return ToolResult(success=False, error="Invalid characters in expression")
        
        try:
            expression = expression.replace("^", "**")
            result = eval(expression, {"__builtins__": {}}, {})
            return ToolResult(
                success=True,
                data={"expression": params.get("expression"), "result": result},
                display_text=f"{params.get('expression')} = {result}"
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Calculation error: {str(e)}")
    
    async def timezone_convert(self, params: Dict) -> ToolResult:
        """Get current time in different timezones"""
        tz_name = params.get("timezone", "UTC")
        
        try:
            from zoneinfo import ZoneInfo
            tz = ZoneInfo(tz_name)
            now = datetime.now(tz)
            
            return ToolResult(
                success=True,
                data={"timezone": tz_name, "time": now.isoformat()},
                display_text=f"Current time in {tz_name}: {now.strftime('%I:%M %p, %A %B %d, %Y')}"
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Timezone error: {str(e)}")
    
    async def _get_gmail_access_token(self) -> Optional[str]:
        """Get Gmail access token from Replit connector"""
        hostname = os.environ.get("REPLIT_CONNECTORS_HOSTNAME")
        repl_identity = os.environ.get("REPL_IDENTITY")
        web_repl_renewal = os.environ.get("WEB_REPL_RENEWAL")
        
        if repl_identity:
            x_replit_token = f"repl {repl_identity}"
        elif web_repl_renewal:
            x_replit_token = f"depl {web_repl_renewal}"
        else:
            return None
        
        if not hostname:
            return None
        
        session = await self._get_session()
        try:
            async with session.get(
                f"https://{hostname}/api/v2/connection?include_secrets=true&connector_names=google-mail",
                headers={
                    "Accept": "application/json",
                    "X_REPLIT_TOKEN": x_replit_token
                }
            ) as resp:
                data = await resp.json()
                connection = data.get("items", [{}])[0]
                settings = connection.get("settings", {})
                access_token = settings.get("access_token") or settings.get("oauth", {}).get("credentials", {}).get("access_token")
                return access_token
        except Exception as e:
            print(f"[GMAIL] Failed to get access token: {e}")
            return None
    
    async def send_email(self, params: Dict) -> ToolResult:
        """Send an email using Gmail via Replit connector"""
        recipient = params.get("recipient") or params.get("to")
        subject = params.get("subject", "Message from Eros")
        body = params.get("body") or params.get("message") or params.get("content", "")
        
        if not recipient:
            return ToolResult(success=False, error="No recipient email address provided")
        
        if not body:
            return ToolResult(success=False, error="No email content provided")
        
        access_token = await self._get_gmail_access_token()
        if not access_token:
            return ToolResult(success=False, error="Gmail not connected. Please connect Gmail first.")
        
        message = MIMEText(body)
        message['to'] = recipient
        message['subject'] = subject
        
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
        
        session = await self._get_session()
        try:
            async with session.post(
                "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json"
                },
                json={"raw": raw_message}
            ) as resp:
                if resp.status == 200:
                    result_data = await resp.json()
                    return ToolResult(
                        success=True,
                        data={"message_id": result_data.get("id"), "recipient": recipient},
                        display_text=f"Email sent to {recipient}",
                        raw_response=result_data
                    )
                else:
                    error_text = await resp.text()
                    return ToolResult(success=False, error=f"Gmail API error ({resp.status}): {error_text}")
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to send email: {str(e)}")


_cloud_executor: Optional[CloudToolExecutor] = None


def get_cloud_executor() -> CloudToolExecutor:
    """Get the global cloud tool executor"""
    global _cloud_executor
    if _cloud_executor is None:
        _cloud_executor = CloudToolExecutor()
    return _cloud_executor


async def execute_cloud_tool(action_type: str, params: Dict, user_id: str) -> ToolResult:
    """Convenience function to execute a cloud tool"""
    executor = get_cloud_executor()
    return await executor.execute(action_type, params, user_id)
