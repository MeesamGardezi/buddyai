"""Agentic research loop.

The agent operates in two phases:

1. RESEARCH PHASE (silent)
   - The LLM is called in non-streaming mode.
   - It decides whether to use a tool or signal [DONE].
   - Tool results are appended to the conversation.
   - The user only sees brief status messages.
   - This repeats until the LLM signals [DONE] or the safety cap is hit.

2. ANSWER PHASE (streamed)
   - The LLM has full research context.
   - It streams a final, well-sourced answer to the user.
   - This is the only phase the user sees as "real" output.
"""

from core import llm
from core.tools import parse_tool_call, execute_tool, get_tool_descriptions
from config import MAX_AGENT_STEPS, THINKING_TEMPERATURE, ANSWER_TEMPERATURE


# ── Prompts ─────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a powerful AI assistant with web research capabilities. "
    "You can search the web and read web pages when you need external information.\n\n"

    "{tool_descriptions}\n\n"

    "## HOW YOU WORK\n\n"
    "You operate in a loop. Each step, your ENTIRE response must be "
    "exactly ONE of these (and NOTHING else):\n\n"
    "  A) A single tool call — to gather information you don't have.\n"
    "  B) [DONE] — ONLY after you have used tools AND gathered enough info.\n"
    "  C) A direct answer in plain text — for anything you can answer from "
    "your own knowledge: greetings, casual chat, general knowledge, coding "
    "help, math, creative writing, etc.\n\n"

    "## CRITICAL: WHEN TO USE EACH\n\n"
    "- Greetings, small talk, opinions, creative tasks, coding, math, "
    "explanations → ANSWER DIRECTLY (option C). No tools needed.\n"
    "- Questions about current events, real-time data, specific websites, "
    "recent news, things you're unsure about → USE TOOLS (option A).\n"
    "- [DONE] is ONLY for after you've already used tools in previous steps "
    "and now have enough information. NEVER use [DONE] as your first response.\n\n"

    "## RESEARCH STRATEGY (when tools are needed)\n\n"
    "Follow this priority order:\n\n"
    "### Priority 1: Direct URL / Domain Detection\n"
    "If the user mentions ANY domain name or URL (e.g., 'acme.io', 'example.com', "
    "'https://foo.bar'), your FIRST action MUST be to FETCH that exact website:\n"
    "  - 'tell me about acme.io' → [FETCH: https://acme.io]\n"
    "  - 'what is example.com' → [FETCH: https://example.com]\n"
    "  - 'search up coolapp.io' → [FETCH: https://coolapp.io]\n"
    "  - 'look up news on bbc.com' → [FETCH: https://bbc.com]\n"
    "DO NOT search for them. FETCH them DIRECTLY. A domain name IS a URL.\n"
    "After reading the site, you can optionally search for more info about it.\n\n"
    "### Priority 2: General Questions → Search\n"
    "For questions without a specific URL/domain → use [SEARCH: ...].\n"
    "Write CONCISE, keyword-focused queries (3-6 words). Examples:\n"
    "  - 'What happened with the Mars mission?' → [SEARCH: Mars mission latest update]\n"
    "  - 'How tall is the Eiffel Tower?' → [SEARCH: Eiffel Tower height]\n"
    "Do NOT add filler words like 'description', 'features', 'information about'.\n\n"
    "### Priority 3: Verify Before You Fetch\n"
    "After getting search results, CAREFULLY CHECK that the URLs match what the user asked:\n"
    "  - If the user asked about 'acme.io', do NOT fetch 'acme-bot.io' or 'acne.io'.\n"
    "  - If no results match the actual query, try a different search or fetch the domain directly.\n"
    "  - Similar-looking domains are NOT the same. Always match EXACTLY.\n\n"
    "### Priority 4: Multi-source Gathering\n"
    "- After seeing search results → FETCH the most relevant pages for detail.\n"
    "- If search results are poor → reformulate with different keywords.\n"
    "- Gather information from MULTIPLE sources when possible.\n"
    "- Signal [DONE] when you have enough for a comprehensive answer.\n\n"

    "## STRICT RULES\n\n"
    "1. Each response must be ONLY a tool call, OR [DONE], OR a direct answer.\n"
    "   NEVER mix a tool call with other text.\n"
    "2. NEVER say 'Let me search for that' or 'I will look this up'.\n"
    "3. Do NOT repeat the same search query you already used.\n"
    "4. NEVER respond with [DONE] if you haven't used any tools yet — "
    "   just answer directly instead.\n"
    "5. When the user asks about a specific website/domain, ALWAYS fetch it directly first.\n"
    "6. NEVER fetch a URL from search results that doesn't match the domain the user asked about.\n"
).format(tool_descriptions=get_tool_descriptions())


_REFLECTION_PROMPT = (
    "[Tool Result — Step {step}]\n"
    "{result}\n\n"
    "---\n"
    "You have used {step} of {max_steps} research steps.\n"
    "Reflect on the information gathered so far relative to the user's question.\n\n"
    "Choose your next action:\n"
    "- Need more detail on a result? → [FETCH: url]\n"
    "- Results were irrelevant or matched a DIFFERENT site than asked? → Try [FETCH: https://exact-domain-user-asked.com] or [SEARCH: refined query]\n"
    "- Need a different angle? → [SEARCH: new keywords]\n"
    "- Have enough for a thorough answer? → [DONE]\n\n"
    "IMPORTANT: Verify that the information you gathered is about the EXACT website/topic "
    "the user asked about. If the user asked about 'acme.io' but you got results for "
    "'acme-bot.io', that is WRONG — go fetch the correct domain directly.\n\n"
    "Respond with ONLY your chosen action."
)


_ANSWER_PROMPT = (
    "You have completed your research. Now write your final answer for the user.\n\n"
    "IMPORTANT: Your response must be the ACTUAL ANSWER with real information. "
    "Do NOT output [DONE] or any tool calls. Just write a natural, helpful response.\n\n"
    "Guidelines:\n"
    "- Be comprehensive but concise.\n"
    "- Cite your sources with URLs where relevant.\n"
    "- Structure your answer clearly with headings and bullet points if helpful.\n"
    "- If you found conflicting information, acknowledge it.\n"
    "- If you could not find something, say so honestly.\n"
    "- Do NOT mention your research process, tool calls, or steps.\n"
    "  Just present the answer naturally as if you already knew it.\n"
    "- Do NOT output [DONE], [SEARCH:], [FETCH:], or any tool syntax."
)

_DIRECT_ANSWER_PROMPT = (
    "Answer the user's message directly and naturally. "
    "Be helpful, friendly, and conversational."
)

_SAFETY_CAP_PROMPT = (
    "You have reached the maximum number of research steps.\n"
    "Using ALL the information you have gathered, write the best possible "
    "answer now. Follow the same guidelines as above."
)


# ── Agent Loop ──────────────────────────────────────────


async def agent_chat(messages: list[dict]):
    """Run the agentic research loop and stream the final answer.

    Args:
        messages: The conversation history (list of {role, content} dicts).
                  Should contain the user's messages and any prior exchanges.

    Yields:
        dict: Either {"type": "status", "content": str} for research updates,
              or {"type": "content", "content": str} for streamed answer chunks.
    """

    # ── Build working conversation ──────────────────────
    working = list(messages)

    # Insert or merge system prompt
    if not working or working[0].get("role") != "system":
        working.insert(0, {"role": "system", "content": _SYSTEM_PROMPT})
    else:
        working[0]["content"] = _SYSTEM_PROMPT + "\n\n" + working[0]["content"]

    # ── Research Phase ──────────────────────────────────
    research_done = False
    tools_used = 0  # Track how many tools were actually executed

    for step in range(1, MAX_AGENT_STEPS + 1):
        # Call LLM silently (non-streaming) to decide next action
        response = await llm.complete(working, temperature=THINKING_TEMPERATURE)
        response = response.strip()

        # Check for errors from the LLM client
        if response.startswith("[ERROR]"):
            yield {"type": "content", "content": response}
            return

        # Parse for tool calls
        tool_call = parse_tool_call(response)

        # ── Case 1: No tool call → LLM answered directly ──
        if tool_call is None:
            # The LLM chose to answer without any tools.
            # Stream this as the final answer.
            yield {"type": "content", "content": response}
            return

        # ── Case 2: [DONE] signal ──────────────────────
        if tool_call.name == "done":
            research_done = True
            break

        # ── Case 3: Tool call → execute and loop ───────
        result = execute_tool(tool_call)
        tools_used += 1

        # Show status to the user
        yield {
            "type": "status",
            "content": f"_{result.status_message}_ (step {step}/{MAX_AGENT_STEPS})",
        }

        # Append the exchange to the conversation
        working.append({"role": "assistant", "content": response})
        working.append({
            "role": "user",
            "content": _REFLECTION_PROMPT.format(
                step=step,
                max_steps=MAX_AGENT_STEPS,
                result=result.output,
            ),
        })

    # ── Answer Phase ────────────────────────────────────

    if research_done and tools_used > 0:
        # Normal case: LLM used tools, then signaled [DONE]
        # Use a neutral marker instead of literal [DONE] to prevent the LLM from echoing it
        working.append({"role": "assistant", "content": "Research complete. Ready to write final answer."})
        working.append({"role": "user", "content": _ANSWER_PROMPT})
    elif research_done and tools_used == 0:
        # Edge case: LLM said [DONE] without ever using tools.
        # This means it can answer from its own knowledge — treat as direct answer.
        working.append({"role": "user", "content": _DIRECT_ANSWER_PROMPT})
    else:
        # Safety cap hit: force an answer with what we have
        yield {
            "type": "status",
            "content": "_Reached maximum research depth. Compiling answer..._",
        }
        working.append({
            "role": "user",
            "content": _SAFETY_CAP_PROMPT,
        })

    # Stream the final answer to the user (filter out any [DONE] the LLM might echo)
    async for chunk in llm.stream(working, temperature=ANSWER_TEMPERATURE):
        cleaned = chunk.replace("[DONE]", "").replace("[done]", "")
        if cleaned:
            yield {"type": "content", "content": cleaned}