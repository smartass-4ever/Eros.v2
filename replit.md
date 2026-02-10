# CNS Discord Bot - Cognitive Neural System

## Overview
This project aims to develop a sophisticated Discord bot powered by the Cognitive Neural System (CNS) architecture, simulating human-like consciousness, personality, and memory. It features dynamic personality, episodic and semantic memory, emotional processing, and self-awareness, evolving through interactions and forming relationships. Key capabilities include a dopamine-driven curiosity system, a humanness reward model, and proactive assistance. The project also includes a B2B API platform for enterprise features like logging, monitoring, and billing. The ultimate goal is a production-ready AI for natural, emotionally intelligent conversations and an API platform for broader integration, with a business vision of establishing a leading AI platform for empathetic and engaging digital interactions.

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture

### Core Cognitive Architecture
The `IntegratedCNSBrain` acts as the central engine, integrating multi-layered memory systems (episodic, semantic, emotional, working), a dynamic Big Five personality model, a multi-dimensional consciousness simulation, and a PAD emotional model. It also tracks relationships to adapt personality expression.

### Core Cognitive Systems
-   **Context Judge System**: Interprets casual language before perception, including slang translation, "I am X" detection, intent detection (greeting, asking_how_you_are, sharing_state), and tone detection.
-   **Cognitive Orchestrator**: Coordinates 50+ cognitive system outputs into a coherent expression package. It features a `SynthesizedContext` with various response modes, blends cognitive outputs, and includes belief conflict detection, user needs detection, social context detection, and a moment-triggered warmth system. **CRITICAL: The orchestrator does NOT modify personality traits - only Personality Pill controls WHO Eros is.**
-   **Game Theory Decision Engine** (`game_theory_decision.py`): TONE-ONLY control system for response intensity:
    - **CRITICAL SEPARATION**: 
      - **Personality Pill** (PersonalityState) = WHO Eros is (James Bond: witty, charming, confident) - IMMUTABLE
      - **Game Theory** (InteractionMode) = HOW he says it (tone intensity only) - MODULATES, never overrides
    - **5 Players**: MEMORY, CURIOSITY, WARMTH, WIT, BELIEFS - scored by LLM context analysis
    - **Tone-Only Control**: warmth, playfulness, assertiveness, directness, tempo
    - **Extreme Situation Rule**: When vulnerability > 0.9 or crisis = true, Game Theory can ONLY make tone STRICTER (more careful), never soften the Bond personality
    - **James Bond Floor**: Playfulness NEVER drops below MEDIUM - wit is always there, only timing changes
    - **Veto System**: vulnerability > 0.9 vetoes WIT timing (not personality), crisis=true vetoes all except WARMTH
    - **ExecutableDirective**: Machine-governable output:
      - Game Theory outputs: focus, interaction_mode (TONE ONLY), prohibitions
      - Personality Pill: ALWAYS defines character traits (never overridden)
      - Expression LLM: phrases naturally within tone constraints
    - **Compact Prompts**: ~200 tokens using structured [FOCUS], [MODE], [PROHIBIT] tags
-   **Unified Self Systems Hub**: Connects all self and growth systems for coordinated pre-response (self-reflection, metacognition) and post-response (metacognitive reflection, learning event recording, neuroplastic optimization) processing.
-   **Manipulation Learning System**: Uses data-driven learning for dynamic opinion generation and psychological directives.
-   **Autonomous Psychological Agency System**: Unifies psychological state aggregation to enable proactive and autonomous response generation.
-   **Conversation Companion System**: Manages natural casual conversations through semantic detection, conversation drives, and proactive engagement.
-   **Contribution-First Response System**: Prioritizes "contribution drives" (knowledge, opinions, memories) for balanced interactions.

### Natural Personality & Expression System
A multi-layered system that translates psychological directives into natural personality expressions. This includes Psychology → Persona Translation, the "James Bond" persona, a Natural Internal Monologue System using `ThoughtSynthesizer` and `StrategyComposer`, and Natural Prompt Assembly for LLMs.

**ARCHITECTURAL RULE: Personality Pill is the SOLE source of WHO Eros is.**
- Only `PersonalityState` in `unified_cns_personality.py` defines character traits (wit, charm, confidence)
- No other system (Game Theory, Orchestrator, Consequence System) may modify personality traits
- Other systems may only control TONE (intensity levels) - not character
- Temporary emotional adaptations adjust expression intensity, never core identity

### User Interaction & Support Systems
-   **Ultra-Helpful Proactive Assistant System**: Proactively assists users by tracking problems, researching solutions, remembering dates, and offering help.
-   **Voice Assistant Interface**: A standalone interface for natural conversation via push-to-talk, integrated with the CNS brain.

### Enterprise API Platform (B2B)
A production-ready REST API offering CNS as a service with Free, Pro, and Enterprise tiers, including multi-user memory isolation, conversation history API, webhooks, secure API key authentication, rate limiting, usage tracking, and an admin dashboard.

### Self-Identity System
A persistent self-identity named "Eros" (after the god of love and desire), founded on kindness and service, with core beliefs, likes, and speech patterns stored in a `self_identity` database table and injected into all LLM calls.

### Belief Registry System (`eros_beliefs.py`)
A structured belief system defining Eros's worldview with 15 core beliefs, each having a statement, conviction level, reasoning, trigger patterns, and counter-positions. It enables conflict detection and proactive sharing of beliefs.

### Data Management & Persistence
-   **PostgreSQL Database Layer**: Production-grade persistence via SQLAlchemy ORM for `UserMemory`, `UserRelationship`, `ConversationHistory`, `BrainState`, `APIKey`, and `UsageLog`.
-   **Secure API Key Management**: Database-backed secure key management with SHA-256 hashing, rate limiting, usage tracking, and key revocation.
-   **Long-Term Memory System**: Automatically stores significant memories in the `user_memories` table, recalls them, and injects them into conversation context.
-   **Integrated Learning System**: Unifies database persistence with neuroplastic and world model systems, persisting `WorldModelMemory` to PostgreSQL and managing learned opinions (`OpinionLearner`) and knowledge (`KnowledgeLearner`).

### Safety Systems
A unified safety manager (`cns_safety_systems.py`) with a **Content Safety System** for deflecting harmful requests and a **Crisis Detection System** for detecting distress signals and providing support.

### ExperienceBus - Unified Learning Architecture (`experience_bus.py`)
A central event bus (`ExperienceBus` singleton) that broadcasts `ExperiencePayload` objects to 12 subscribed learning systems. This facilitates collaborative learning and a feedback loop where systems contribute learning back, which the `CognitiveOrchestrator` uses for real-time adjustment.

### Emotional Reinforcement System (`emotional_reinforcement_system.py`)
Provides Eros with internal feelings (dopamine for joy/satisfaction, sadness for introspection) that drive learning and behavior based on user engagement tracking.

### Consequence System (`consequence_system.py`)
Makes emotional states have REAL lasting effects - mistakes close doors, trust requires time to rebuild:
-   **Compartmentalized State**: `confidence_global` (Eros's internal boldness), `trust[user_id]` (per-user relationship strength), `cooldowns[user_id][move_type]` (specific moves that burned)
-   **Move Taxonomy**: 12 enumerated moves across tiers (Base, Trust-Gated, Confidence-Gated, Specialty) with specific requirements
-   **Consequence Events**: Guilt → confidence_global drop (less bold everywhere), Sadness → trust[user_id] drop + specific move cooldown
-   **Recovery Friction**: Requires varied signals (reply_count + engagement_depth + time_spacing + emotional_positive), not just time
-   **Belief-Aligned Repair Moves**: `grounded_observation`, `reflective_mirroring`, `quiet_presence` always available, never locked
-   **Safeguards**: Floors at 0.2 (never paralyzed), auto-expiring cooldowns, database persistence via SQLAlchemy ORM

### Inner Life System (`inner_life_system.py`)
Gives Eros an active mental life, including a Personal Interest Registry, a Reflection Queue for important conversation moments, and an Active Thinking Loop during idle time to form new insights and trigger proactive outreach.

### Other Advanced AI Systems
Includes Creative and Subconscious Systems (Imagination Engine, REM Subconscious Engine, Stream of Consciousness) and Advanced AI Systems (LLM Fine-Tuning System, Humanness Reward Model, Enhanced Expression System, multimodal capabilities).

### Agentic Action System
Enables Eros to execute real-world actions, not just chat. Like a true Jarvis-style AI assistant.

**Core Components:**
-   **Action Orchestrator** (`action_orchestrator.py`): Coordinates intent detection, safety checks, and tool execution.
-   **LLM Intent Detector** (`agentic_actions.py`): Uses Mistral to understand natural language action requests with JSON extraction. Falls back to regex for reliability.
-   **Cloud Tools** (`cloud_tools.py`): Tier 1 tools (no OAuth): web search (DuckDuckGo), weather, news, stocks, Wikipedia, calculator, timezone.
-   **Local Node** (`eros_local_node.py`): Standalone daemon for user's computer enabling local actions: file operations, app launch, clipboard, screenshot, system info.
-   **Node Bridge Server** (`node_bridge_server.py`): WebSocket server running on port 8765 for Local Node connections. Features secure pairing with 6-character codes, heartbeat, and command routing.

**Discord Commands:**
-   `!pair` - Generate a pairing code to connect your computer
-   `!nodes` - Show your connected computers
-   `!unpair` - Disconnect your computer

**Database Tables:**
-   `action_audit_log`: Tracks all physical actions for safety and learning
-   `user_tool_permissions`: Per-user tool enable/disable settings
-   `connected_services`: OAuth token storage for Spotify, Google, etc.
-   `local_node_connections`: Tracks user's paired computers

**CNS Integration (Actions Learn and Remember):**
-   **Experience Bus**: All actions emit PHYSICAL_ACTION, ACTION_SUCCESS, ACTION_FAILURE events
-   **Memory System**: Stores significant actions as episodic memories via `store_action_memory()`
-   **Knowledge System**: Extracts facts from web search, Wikipedia, and news results
-   **Consequence System**: Tracks action trust per user - success builds trust, failures reduce confidence
-   **Inner Life System**: Queues interesting action discoveries for reflection

**MDC Physical Actions**: Extended Markov Decision Controller with 17 physical action types integrated into Q-learning

## Recent Changes (February 2026)

-   **Web Search Upgraded**: Switched from deprecated `duckduckgo-search` to new `ddgs` library for real web search results. Searches now return actual current results instead of limited instant answers.
-   **News Search Working**: Uses `ddgs` news feature for real headlines with sources and dates.
-   **Gmail Email Sending**: OAuth connected and functional. Emails send without confirmation requirement for smooth flow.
-   **Email Reading Disabled**: Deliberately disabled due to prompt injection risk from malicious email content.
-   **Fast-track Email Detection**: Added regex-based fast-track for email intents to bypass LLM processing when email address is clearly present.
-   **Action Execution Flow Fixed**: Actions now properly flow: Detection → Execution → Results → Natural Response integration.

## External Dependencies

-   **Discord API**: Used for the Discord bot interface via `discord.py`.
-   **Mistral AI**: External Large Language Model service for complex query processing.
-   **Stripe**: Integrated for billing, subscription management, and payment processing for the B2B API platform.
-   **ddgs**: DuckDuckGo Search library for web and news search capabilities.
-   **Gmail API**: For sending emails via OAuth-connected Gmail accounts.