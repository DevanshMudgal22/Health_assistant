from app.state import AgentState
from app.tools import get_search_tool
from app.rag import get_retriever
from app.supabase_client import supabase, save_message, save_summary, get_message_count
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from app.config import GROQ_API_KEY

llm = ChatGroq(
    temperature=0.2,
    model_name="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
)

SUMMARIZE_EVERY = 4


# =====================================================
# ROUTER NODE
# =====================================================

def router_node(state: AgentState):
    messages   = state.get("messages", [])
    session_id = state.get("session_id", "default")

    if not messages:
        return {"intent": "general_info"}

    text = messages[-1].content.lower().strip()

    save_message(session_id, "user", messages[-1].content)
    count = get_message_count(session_id)

    # ── GREETING: exact word match to avoid false triggers (e.g. "history", "thigh") ──
    greeting_words = ["hi", "hello", "hey", "hii", "helo", "hlo"]
    if any(text == x or text.startswith(x) for x in greeting_words) or "how are you" in text:
        reply = "👋 Hello! Main HealthPilot AI hoon — aapka personal health assistant.\n\nAaj main aapki kya madad kar sakta hoon? Apni problem batayein. 🩺"
        incognito = state.get("incognito", False)
        if not incognito:
            save_message(session_id, "assistant", reply)
        return {
            "intent": "greeting_done",
            "message_count": count,
            "messages": messages + [AIMessage(content=reply)],
        }

    # ── ACKNOWLEDGEMENT: thanks, ok, accha — short reply, no LLM needed ──
    # FIX: "Thanks" pe bekar generic response aata tha, ab short warm reply dega
    if any(x in text for x in [
        "thanks", "thank you", "ok", "okay", "got it",
        "sure", "alright", "thik hai", "shukriya",
        "theek", "accha", "haan", "👍", "hmm", "fine",
        "bahut acha", "perfect", "great", "awesome",
    ]):
        reply = "😊 Koi baat nahi! Kuch aur poochna ho toh batao."
        save_message(session_id, "assistant", reply)
        return {
            "intent": "greeting_done",
            "message_count": count,
            "messages": messages + [AIMessage(content=reply)],
        }

    # ── EMERGENCY: critical symptoms ──
    if any(x in text for x in [
        # breathing
        "can't breathe", "cannot breathe", "breathing difficulty", "shortness of breath", "choking",
        # heart
        "chest pain", "heart attack", "heart failure", "palpitations", "irregular heartbeat",
        # consciousness
        "faint", "fainting", "unconscious", "passed out", "unresponsive", "collapsed",
        # bleeding
        "bleeding heavily", "heavy bleeding", "blood loss", "hemorrhage", "bleeding won't stop",
        # brain
        "stroke", "seizure", "convulsion", "paralysis", "sudden numbness", "can't speak",
        # overdose / poisoning
        "overdose", "poisoning", "swallowed poison", "took too many pills",
        # trauma
        "accident", "head injury", "broken bone", "deep cut", "severe burn",
        # other critical
        "suicide", "kill myself", "self harm", "allergic reaction", "anaphylaxis", "swelling throat",
    ]):
        return {"intent": "emergency", "message_count": count}

    # ── DATA QUERY: user explicitly asking for doctors/medicines from DB ──
    # FIX: "medicine", "medicines" hata diye — ye words medical advice mein bhi aate hain
    # sirf explicit DB intent wale words rakhe
    if any(x in text for x in [
        "doctor dhundo", "find doctor", "find medicine",
        "price batao", "price of", "kitne ka",
        "₹", "stock hai", "available hai",
        "book appointment", "doctor chahiye",
        "show medicines", "list medicines",
        "fee kitna", "consultation fee",
        "specialist chahiye", "specialization",
    ]):
        return {"intent": "data_query", "message_count": count}

    # ── MEDICAL ADVICE: symptoms and health concerns ──
    if any(x in text for x in [
        "fever", "allergy", "pain", "cold", "cough", "headache",
        "diabetes", "blood pressure", "stomach", "vomiting",
        "diarrhea", "rash", "fatigue", "tired", "infection",
        "migraine", "anxiety", "stress", "sleep", "back pain",
        "symptom", "symptoms", "treatment", "cure", "remedy",
    ]):
        return {"intent": "medical_advice", "message_count": count}

    return {"intent": "general_info", "message_count": count}


# =====================================================
# CLARIFICATION NODE
# =====================================================

def clarification_node(state: AgentState):
    messages              = state.get("messages", [])
    session_id            = state.get("session_id", "default")
    clarification_step    = state.get("clarification_step", 0)
    clarification_answers = state.get("clarification_answers", {}) or {}

    user_input = messages[-1].content if messages else ""

    # ── Step 0: Generate MAX 1 question, only if query is too vague ──
    # FIX: pehle 2 questions force karta tha, ab LLM khud decide karta hai
    # clear query hai toh seedha answer, vague hai toh 1 question
    if clarification_step == 0:
        gen_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are HealthPilot AI.
User described a health concern. Generate exactly 1 short, focused clarifying question.
Only ask if the query is too vague to answer safely (e.g. just "pain" or "problem").
If query is clear enough, return: {{"questions": []}}
Return ONLY valid JSON: {{"questions": ["Q1"]}}
No explanation outside JSON."""),
            ("user", "{complaint}")
        ])

        chain = gen_prompt | llm | JsonOutputParser()

        try:
            result    = chain.invoke({"complaint": user_input})
            questions = result.get("questions", [])[:1]
        except Exception:
            questions = ["How long have you been experiencing this, and where exactly?"]

        # Query clear hai — skip clarification, direct answer dedo
        if not questions:
            return {
                "clarification_needed":    False,
                "clarification_step":      0,
                "clarification_questions": [],
            }

        reply = questions[0]
        save_message(session_id, "assistant", reply)

        return {
            "messages":                messages + [AIMessage(content=reply)],
            "clarification_needed":    True,
            "clarification_questions": questions,
            "clarification_step":      1,
            "clarification_answers":   {},
        }

    # ── Step 1: User ne jawab diya — bas, answer karo ──
    # FIX: pehle Step 1 → Step 2 tha (2 questions), ab Step 1 ke baad seedha done
    elif clarification_step == 1:
        clarification_answers["q1"] = user_input
        return {
            "clarification_needed":  False,
            "clarification_step":    0,
            "clarification_answers": clarification_answers,
        }

    return {"clarification_needed": False, "clarification_step": 0}


# =====================================================
# SUMMARIZATION NODE
# =====================================================

def summarization_node(state: AgentState):
    session_id = state.get("session_id", "default")

    message_count    = get_message_count(session_id)
    existing_summary = state.get("conversation_summary", "") or ""
    messages         = state.get("messages", [])

    if message_count < 2:
        print(f"[Summarizer] Skipping — not enough messages: {message_count}")
        return {}

    should_summarize = (
        message_count % SUMMARIZE_EVERY == 0
        or (not existing_summary and message_count >= 2)
    )

    if not should_summarize:
        print(f"[Summarizer] Skipping — message count: {message_count}")
        return {}

    print(f"[Summarizer] Generating summary at message count: {message_count}")

    recent_msgs = messages[-SUMMARIZE_EVERY:] if messages else []
    recent_text = "\n".join([
        f"{m.type.upper()}: {m.content}"
        for m in recent_msgs
    ])

    if not recent_text.strip():
        print("[Summarizer] Skipping — no valid recent text")
        return {}

    full_prompt = f"""You are a medical chat summarizer.
Summarize the conversation below into 3-5 concise bullet points.
Focus on: user symptoms, advice given, medicines or doctors mentioned.
If a previous summary exists, merge with new info — do not repeat.

Previous Summary:
{existing_summary if existing_summary else "None"}

Recent Conversation:
{recent_text}

Return ONLY the updated bullet-point summary. No preamble. No extra text."""

    try:
        new_summary = llm.invoke(full_prompt).content.strip()
        print(f"\n[SUMMARY GENERATED]\n{new_summary}\n")
    except Exception as e:
        print(f"[Summarizer] Error: {e}")
        return {}

    if not new_summary:
        print("[Summarizer] Empty summary — skipping save")
        return {}

    save_summary(session_id, new_summary, message_count)
    return {"conversation_summary": new_summary}


# =====================================================
# SAFETY NODE
# =====================================================

def safety_node(state: AgentState):
    session_id = state.get("session_id", "default")
    intent     = state.get("intent")

    if intent == "emergency":
        reply = (
            "🚨 Emergency Detected!\n\n"
            "Please seek immediate medical help or contact:\n"
            "- Nearest emergency room\n"
            "- Emergency services: 112 (India) / 911 (US)\n"
            "- Ambulance: 108 (India)\n\n"
            "Do not wait — go now or call for help immediately."
        )
        save_message(session_id, "assistant", reply)
        return {
            "messages": state.get("messages", []) + [AIMessage(content=reply)],
            "error": "emergency"
        }

    return {}


# =====================================================
# SQL AGENT NODE
# =====================================================

def sql_agent_node(state: AgentState):
    messages              = state.get("messages", [])
    clarification_answers = state.get("clarification_answers", {}) or {}

    if not messages:
        return {"sql_result": ""}

    user_input = messages[-1].content
    if clarification_answers:
        extra      = " | ".join([f"{k}: {v}" for k, v in clarification_answers.items()])
        user_input = f"{user_input} [Context: {extra}]"

    print(f"\n[SQL] User input: {user_input}")

    planner_prompt = f"""You are a database query planner for a medical app.

Tables available:
1. medicines  → columns: name, description, dosage_form, price, requires_prescription, stock_quantity
2. doctors    → columns: name, specialization, consultation_fee, availability, rating

User query: {user_input}

Return ONLY valid JSON (no explanation, no markdown, no code block):
{{
  "table": "medicines" or "doctors",
  "filters": {{
    "price_lte": number or null,
    "consultation_fee_lte": number or null,
    "specialization": string or null,
    "name": string or null,
    "requires_prescription": true or false or null
  }},
  "limit": 10
}}"""

    try:
        import re, json
        raw     = llm.invoke(planner_prompt).content.strip()
        print(f"[SQL] LLM planner raw output: {raw}")
        cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
        plan    = json.loads(cleaned)
        print(f"[SQL] Parsed plan: {plan}")
    except Exception as e:
        print(f"[SQL] Planner error: {e}")
        return {"sql_result": f"Query planner error: {str(e)}"}

    table   = plan.get("table", "").strip()
    filters = plan.get("filters", {}) or {}
    limit   = int(plan.get("limit", 10))

    if table not in ("medicines", "doctors"):
        print(f"[SQL] Invalid table: {table}")
        return {"sql_result": "Could not determine which table to query."}

    print(f"[SQL] Querying table: {table} | filters: {filters} | limit: {limit}")

    try:
        query = supabase.table(table).select("*")

        if filters.get("price_lte") is not None:
            query = query.lte("price", filters["price_lte"])
        if filters.get("consultation_fee_lte") is not None:
            query = query.lte("consultation_fee", filters["consultation_fee_lte"])
        if filters.get("specialization"):
            query = query.ilike("specialization", f"%{filters['specialization']}%")
        if filters.get("name"):
            query = query.ilike("name", f"%{filters['name']}%")
        if filters.get("requires_prescription") is not None:
            query = query.eq("requires_prescription", filters["requires_prescription"])

        res  = query.limit(limit).execute()
        data = res.data

        if not data:
            print("[SQL] No results found.")
            return {"sql_result": "No results found in the database."}

        if table == "medicines":
            lines = []
            for m in data:
                rx = "Prescription required" if m.get("requires_prescription") else "OTC"
                lines.append(
                    f"• {m['name']} | {m.get('dosage_form','')} | "
                    f"₹{m.get('price','')} | {rx} | Stock: {m.get('stock_quantity','')}"
                )
            formatted = f"Medicines Found ({len(data)}):\n" + "\n".join(lines)
        else:
            lines = []
            for d in data:
                lines.append(
                    f"• {d['name']} | {d.get('specialization','')} | "
                    f"Fee: ₹{d.get('consultation_fee','')} | "
                    f"Available: {d.get('availability','')} | "
                    f"Rating: {d.get('rating','')}/5"
                )
            formatted = f"Doctors Found ({len(data)}):\n" + "\n".join(lines)

        print(f"\n{'='*50}")
        print(f"[SQL RESULT] Table: {table} | Rows: {len(data)}")
        print(formatted)
        print(f"{'='*50}\n")

        return {"sql_result": formatted}

    except Exception as e:
        print(f"[SQL] Execution error: {e}")
        return {"sql_result": f"Database error: {str(e)}"}


# =====================================================
# RAG NODE
# =====================================================

def rag_node(state: AgentState):
    messages              = state.get("messages", [])
    clarification_answers = state.get("clarification_answers", {}) or {}

    if not messages:
        return {}

    original_complaint = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            original_complaint = m.content
            break

    answers_text = " ".join(clarification_answers.values())
    full_query   = f"{original_complaint} {answers_text}".strip()

    print(f"\n[RAG QUERY] {full_query}")

    retriever = get_retriever()

    try:
        docs    = retriever.invoke(full_query)
        context = "\n\n".join([d.page_content for d in docs]) if docs else ""
    except Exception:
        docs    = []
        context = ""

    print(f"\n{'='*50}")
    print(f"[RAG RESULT] Chunks retrieved: {len(docs)}")
    print(context if context else "No RAG context found.")
    print(f"{'='*50}\n")

    return {"retrieved_docs": context}


# =====================================================
# WEB SEARCH NODE
# =====================================================

def web_search_node(state: AgentState):
    messages = state.get("messages", [])
    if not messages:
        return {}

    query = messages[-1].content
    tool  = get_search_tool()

    try:
        results   = tool.invoke(query)[:5]
        formatted = "\n".join([
            f"- {r.get('content', '')} ({r.get('url', '')})"
            for r in results
        ])
    except Exception as e:
        formatted = f"Web search error: {str(e)}"

    return {"web_search_result": formatted}


# =====================================================
# RESPONSE NODE
# =====================================================

def response_node(state: AgentState):
    messages   = state.get("messages", [])
    session_id = state.get("session_id", "default")
    intent     = state.get("intent", "") or ""

    if not messages:
        return {}

    user_input = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            user_input = m.content
            break

    print(f"[RESPONSE] Intent: {intent} | User input: {user_input[:80]}")

    rag_context   = state.get("retrieved_docs", "") or ""
    web_context   = state.get("web_search_result", "") or ""
    sql_result    = state.get("sql_result", "") or ""
    summary       = state.get("conversation_summary", "") or ""
    clarification = state.get("clarification_answers", {}) or {}

    context_parts = []

    if summary:
        context_parts.append(f"Conversation Summary:\n{summary}")
    if clarification:
        answers_text = "\n".join([f"- {v}" for v in clarification.values()])
        context_parts.append(f"User Info:\n{answers_text}")
    if sql_result:
        context_parts.append(f"DATABASE RESULTS (use these exactly):\n{sql_result}")
    if rag_context:
        context_parts.append(f"Medical Knowledge:\n{rag_context}")
    if web_context:
        context_parts.append(f"Web Results:\n{web_context}")

    context_str = "\n\n".join(context_parts)

    if intent == "data_query":
        system_prompt = """You are HealthPilot AI — a concise medical assistant.

ROLE: Present database results as a smart, brief recommendation — not a dump.

RULES:
- Only answer from the provided database results. If results are empty or irrelevant, say so briefly.
- Never show raw data, IDs, or field names.
- Skip items that don't match the query.
- Max 3–5 items unless user asks for more.
- Prioritize relevance: best match first.

FORMAT:
- Lead with one short line (what you found).
- List only key info: name, price, availability, Rx-required — inline, not stacked.
- Group only if 3+ items share a category.
- End with one practical tip or next step (optional, only if useful).
- No markdown. No headers. Light emoji ok.

TONE: Confident, brief, helpful — like a pharmacist giving a quick answer.

LANGUAGE: Match user's language/style exactly (Hindi, English, Hinglish — mirror it).
"""
    else:
        # FIX: "Thanks/ok" pe bhi agar general_info se aa jaye toh short reply dega
        # "If user says thanks or acknowledges" rule add kiya
        system_prompt = """You are HealthPilot AI — a sharp, experienced doctor giving quick clinic-style advice.

RULES:
- Answer directly. No greetings, no clarifying questions, no rephrasing the user's words back.
- If user says thanks, ok, or acknowledges — reply with one short warm line only. Nothing else.
- Health topics only. Off-topic? One line decline, nothing more.
- Never say "consult a doctor", "I'm an AI", or add disclaimers.
- Never invent facts. If unsure, say so in one line.

FORMAT (use only what's relevant):
🔍 [Likely cause — 1 line]
💊 [Treatment / medicine + dosage if applicable]
🥗 [Diet or lifestyle tip — only if useful]
⚠️ [Red flag — only if genuinely urgent]

LENGTH: 4–6 lines max. Cut anything that doesn't add value.

TONE: Confident, warm, direct — like a doctor in a 5-minute consult.

LANGUAGE: Mirror the user exactly — Hinglish, Hindi, English, whatever they use.
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("system", "Context:\n{context}"),
        ("user", "{input}")
    ])

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({
        "context": context_str,
        "input": user_input
    })

    print(f"\n{'='*50}")
    print(f"[FINAL RESPONSE]")
    print(response)
    print(f"{'='*50}\n")

    save_message(session_id, "assistant", response)
    return {"messages": messages + [AIMessage(content=response)]}