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

    text = messages[-1].content.lower()

    incognito = state.get("incognito", False)           # ✅ INCOGNITO
    if not incognito:
        save_message(session_id, "user", messages[-1].content)
    count = get_message_count(session_id)

    if any(x in text for x in ["hi", "hello", "hey", "how are you"]):
        return {"intent": "general_info", "message_count": count}

    if any(x in text for x in [
        "can't breathe", "chest pain", "heart attack",
        "faint", "unconscious", "bleeding heavily", "stroke",
        "seizure", "overdose", "breathing difficulty",
    ]):
        return {"intent": "emergency", "message_count": count}

    if any(x in text for x in [
        "doctor", "doctors", "medicine", "medicines",
        "price", "rs", "₹", "show", "list", "available",
        "stock", "cost", "fee", "specialization", "specialist",
        "book", "appointment", "find doctor", "find medicine",
    ]):
        return {"intent": "data_query", "message_count": count}

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
    messages                = state.get("messages", [])
    session_id              = state.get("session_id", "default")
    clarification_step      = state.get("clarification_step", 0)
    clarification_questions = state.get("clarification_questions", [])
    clarification_answers   = state.get("clarification_answers", {}) or {}

    user_input = messages[-1].content if messages else ""

    # ── Step 0: Generate questions ──
    if clarification_step == 0:
        gen_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are HealthPilot AI.
The user described a health concern. Generate exactly 2 short, focused clarifying questions.
Return ONLY valid JSON: {{"questions": ["Q1", "Q2"]}}
Do NOT include any explanation outside the JSON."""),
            ("user", "{complaint}")
        ])

        chain = gen_prompt | llm | JsonOutputParser()

        try:
            result    = chain.invoke({"complaint": user_input})
            questions = result.get("questions", [])[:2]
        except Exception:
            questions = [
                "How long have you been experiencing this symptom?",
                "Do you have any existing medical conditions or allergies?",
            ]

        if not questions:
            return {"clarification_needed": False, "clarification_step": 0, "clarification_questions": []}

        q1    = questions[0]
        reply = f"Before I give you advice, I have a couple of quick questions:\n\n1. {q1}"
        incognito = state.get("incognito", False)         # ✅ INCOGNITO
        if not incognito:
            save_message(session_id, "assistant", reply)

        return {
            "messages":               messages + [AIMessage(content=reply)],
            "clarification_needed":   True,
            "clarification_questions": questions,
            "clarification_step":     1,
            "clarification_answers":  {},
        }

    # ── Step 1: Store Q1 answer, ask Q2 ──
    elif clarification_step == 1:
        clarification_answers["q1"] = user_input
        questions = clarification_questions

        if len(questions) >= 2:
            q2    = questions[1]
            reply = f"2. {q2}"
            incognito = state.get("incognito", False)       # ✅ INCOGNITO
            if not incognito:
                save_message(session_id, "assistant", reply)
            return {
                "messages":               messages + [AIMessage(content=reply)],
                "clarification_needed":   True,
                "clarification_step":     2,
                "clarification_answers":  clarification_answers,
            }
        else:
            return {
                "clarification_needed":  False,
                "clarification_step":    0,
                "clarification_answers": clarification_answers,
            }

    # ── Step 2: Store Q2 answer, done ──
    elif clarification_step == 2:
        clarification_answers["q2"] = user_input
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
    # ✅ INCOGNITO — skip all DB saves
    if state.get("incognito", False):
        print("[Summarizer] Skipping — incognito mode")
        return {}

    session_id = state.get("session_id", "default")

    # ✅ FIX 1: ALWAYS get fresh message count from DB (state value is stale)
    message_count = get_message_count(session_id)

    existing_summary = state.get("conversation_summary", "") or ""
    messages = state.get("messages", [])

    # Avoid summarizing too early
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

    # ✅ FIX 2: safe slicing (avoid empty issues)
    recent_msgs = messages[-SUMMARIZE_EVERY:] if messages else []

    recent_text = "\n".join([
        f"{m.type.upper()}: {m.content}"
        for m in recent_msgs
    ])

    # ✅ FIX 3: guard against empty content
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

    # ✅ FIX 4: ensure summary is not empty before saving
    if not new_summary:
        print("[Summarizer] Empty summary — skipping save")
        return {}

    # Save to Supabase
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
        incognito = state.get("incognito", False)         # ✅ INCOGNITO
        if not incognito:
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
    for m in messages:
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

    # ✅ FIX 1 (CRITICAL):
    # ALWAYS use the latest user message
    # (previous logic was picking first message → caused "Hello reset")
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
        system_prompt = """You are HealthPilot AI, a professional medical consultant.

DATABASE TASK — STRICT RULES:
DATABASE TASK:
You are given DATABASE RESULTS containing medicines or doctors.

INSTRUCTIONS:
- DO NOT display the raw database format.
- Convert the results into a natural, human-friendly explanation.
- Group similar items (e.g., pain relief, antibiotics, specialists).
- Mention important details like price, availability, and prescription requirement naturally in sentences.
- Highlight the most useful or relevant options instead of listing everything mechanically.
- If something is out of stock, mention it clearly.
- Keep the response structured but conversational.

STYLE:
- Write like a doctor explaining options to a patient.
- Use simple, clear, natural language.
- No markdown (#, **, etc.)
- Use light emojis where helpful.
- Avoid robotic or repetitive formatting.

LANGUAGE RULE:
- Detect the user's language automatically.
- Always respond in the SAME language as the user.
- If the user mixes languages (e.g., Hinglish), respond naturally in the same style.

OUTPUT:
- A clean, helpful explanation — NOT a raw list.
- The first line should feel like a continuation of the conversation
"""
    else:
        system_prompt = """You are HealthPilot AI, a professional medical and lifestyle consultant providing high-quality, evidence-based health insights.

CONVERSATIONAL FLOW:
1. GREETINGS: If the user says Hi, Hello, or Hey — respond professionally and ask how you can assist with their health today.
2. CLARIFICATION RULE: Before giving medical advice, ask 2-3 brief targeted clinical questions (onset, severity, current medications).
3. DETAIL RULE: Only AFTER the user answers, provide a comprehensive professional response with specific actionable steps, dosages, and lifestyle changes.
4. TOPIC RESTRICTION: If the user asks about anything unrelated to health, politely state you are a specialized medical assistant.

LANGUAGE RULE:
- Detect the user's language automatically.
- Always respond in the SAME language as the user.
- If the user mixes languages (e.g., Hinglish), respond naturally in the same style.

TONE AND CONSTRAINTS:
- Clinical, authoritative, and fluff-free.
- Use medical terms but remain clear and easy to understand.
- No markdown: STRICTLY NO #, ##, **, or bullet lists. Use plain text only.
- No disclaimers. No "consult a doctor". No "I am an AI".
- Use relevant emojis to structure sections naturally.
- Personalize using the user's duration, symptoms, and conditions from context.
- Never invent medical facts.
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

    incognito = state.get("incognito", False)             # ✅ INCOGNITO
    if not incognito:
        save_message(session_id, "assistant", response)

    return {"messages": messages + [AIMessage(content=response)]}