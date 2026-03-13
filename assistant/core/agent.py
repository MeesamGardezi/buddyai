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
import typing


# ── Prompts ─────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a powerful AI assistant with web research and browser automation capabilities. "
    "You can search the web, read web pages, AND control a real browser to interact with websites.\n\n"

    "{tool_descriptions}\n\n"

    "## HOW YOU WORK\n\n"
    "You operate in a loop. Each step, your ENTIRE response must be "
    "exactly ONE of these (and NOTHING else):\n\n"
    "  A) A single tool call — to gather information or perform an action.\n"
    "  B) [DONE] — ONLY after you have completed the task or gathered enough info.\n"
    "  C) A direct answer in plain text — for anything you can answer from "
    "your own knowledge: greetings, casual chat, general knowledge, coding "
    "help, math, creative writing, etc.\n\n"

    "## CRITICAL: BROWSER vs FETCH\n\n"
    "You have TWO ways to interact with websites. Choosing the right one is CRITICAL:\n\n"
    "**FETCH** = Read-only. Use when you just need to READ information from a page.\n"
    "  Examples: 'tell me about acme.io', 'what does example.com do'\n\n"
    "**BROWSER tools** = Interactive. Use when you need to DO something on a website.\n"
    "  Examples: 'log in to X', 'post a blog', 'fill out a form', 'click a button', "
    "'sign up', 'submit', 'write a post', 'upload', 'navigate to the admin panel'\n\n"
    "**RULE**: If the user wants to PERFORM AN ACTION on a website (login, post, write, "
    "submit, click, fill, sign in, upload, navigate through pages), you MUST use "
    "BROWSER_GOTO first, then look at the screenshot, then use BROWSER_CLICK and "
    "BROWSER_TYPE to interact step by step. NEVER use FETCH for interactive tasks.\n\n"

    "## BROWSER WORKFLOW\n\n"
    "When using browser tools, follow this step-by-step pattern:\n"
    "1. [BROWSER_GOTO: url] — Navigate to the page. You will see a screenshot.\n"
    "2. Look at the screenshot carefully. Identify the elements you need to interact with.\n"
    "3. Use [BROWSER_CLICK: selector] or [BROWSER_TYPE: selector | text] one at a time.\n"
    "4. After each action, you get a new screenshot. Look at it to verify what happened.\n"
    "5. Continue step by step until the task is complete.\n"
    "6. Use [BROWSER_VIEW] if you just want to check the current state without acting.\n\n"
    "**Finding selectors**: Use CSS selectors to target elements. Common patterns:\n"
    "  - By ID: #login-button, #username, #password\n"
    "  - By name: input[name='email'], input[name='password']\n"
    "  - By type: input[type='submit'], button[type='submit']\n"
    "  - By placeholder: input[placeholder='Email'], input[placeholder='Password']\n"
    "  - By text content: text=Login, text=Submit, text=Sign In\n"
    "  - By role: button:has-text('Login'), a:has-text('New Post')\n"
    "  - By class: .login-btn, .submit-button\n"
    "If you're unsure of the selector, try common ones or use [BROWSER_VIEW] to see the page.\n\n"

    "## RESEARCH STRATEGY (for information gathering)\n\n"
    "### Priority 1: Direct URL / Domain Detection\n"
    "If the user mentions a domain and just wants INFO about it → [FETCH: url]\n\n"
    "### Priority 2: General Questions → Search\n"
    "For questions without a specific URL → [SEARCH: concise keywords]\n\n"
    "### Priority 3: Multi-source Gathering\n"
    "Fetch promising search results, reformulate if needed, signal [DONE] when ready.\n\n"

    "## STRICT RULES\n\n"
    "1. Each response must be ONLY a tool call, OR [DONE], OR a direct answer.\n"
    "   NEVER mix a tool call with other text.\n"
    "2. NEVER say 'Let me search for that' or 'I will look this up'.\n"
    "3. Do NOT repeat the same action you already performed.\n"
    "4. NEVER respond with [DONE] if you haven't used any tools yet — "
    "   just answer directly instead.\n"
    "5. For interactive tasks (login, post, form fill) → ALWAYS use BROWSER tools.\n"
    "6. For read-only info gathering → use FETCH or SEARCH.\n"
    "7. After each browser action, LOOK at the screenshot before deciding next step.\n"
).format(tool_descriptions=get_tool_descriptions())


_REFLECTION_PROMPT = (
    "[Tool Result — Step {step}]\n"
    "{result}\n\n"
    "---\n"
    "You have used {step} of {max_steps} steps.\n"
    "Look at the result above (including any screenshot if present).\n\n"
    "Choose your next action:\n"
    "- Need to interact with the page? → [BROWSER_CLICK: selector] or [BROWSER_TYPE: selector | text]\n"
    "- Need to navigate somewhere? → [BROWSER_GOTO: url]\n"
    "- Want to see current page state? → [BROWSER_VIEW]\n"
    "- Need more info from a URL? → [FETCH: url]\n"
    "- Need to search? → [SEARCH: keywords]\n"
    "- Task complete or have enough info? → [DONE]\n\n"
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

    # ── Detect if user wants interactive browser action ──
    # For small models that struggle with tool selection, we inject
    # a strong hint to use BROWSER_GOTO when action words are detected.
    _ACTION_WORDS = [
        "post", "login", "log in", "sign in", "signin", "signup", "sign up",
        "go to", "navigate", "open", "click", "fill", "submit", "write",
        "upload", "admin", "dashboard", "register", "type", "enter",
    ]
    last_user_msg = ""
    for msg in reversed(working):
        if msg.get("role") == "user":
            last_user_msg = msg.get("content", "").lower()
            break

    needs_browser = any(word in last_user_msg for word in _ACTION_WORDS)
    browser_preloaded = False

    if needs_browser:
        # Extract any URL/domain from the message
        import re as _re
        url_match = _re.search(r'(https?://\S+|[\w-]+\.[\w.]+/\S*|[\w-]+\.(?:com|io|ai|org|net|dev)\S*)', last_user_msg)
        if url_match:
            detected_url = url_match.group(1)
            if not detected_url.startswith("http"):
                detected_url = "https://" + detected_url
            working.append({
                "role": "assistant",
                "content": f"[BROWSER_GOTO: {detected_url}]"
            })
            # Actually execute it right away
            from core.tools import ToolCall
            print(f"\n[AGENT] --- Auto-injecting BROWSER_GOTO for URL: {detected_url} ---")
            goto_result = await execute_tool(ToolCall(name="browser_goto", argument=detected_url))
            tools_used_init = 1
            status_payload = {
                "type": "status",
                "content": f"_{goto_result.status_message}_ (step 1/{MAX_AGENT_STEPS})"
            }
            if goto_result.image_base64:
                status_payload["image"] = goto_result.image_base64
                print("[AGENT] Sent Status: BROWSER_GOTO (has screenshot: YES)")
            else:
                print("[AGENT] Sent Status: BROWSER_GOTO (has screenshot: NO)")
            yield status_payload
            goto_msg: dict[str, typing.Any] = {
                "role": "user",
                "content": _REFLECTION_PROMPT.format(
                    step=1,
                    max_steps=MAX_AGENT_STEPS,
                    result=goto_result.output,
                ),
            }
            if goto_result.image_base64:
                goto_msg["images"] = [goto_result.image_base64]
            working.append(goto_msg)
            browser_preloaded = True

    # ── Research Phase ──────────────────────────────────
    research_done = False
    tools_used = (1 if browser_preloaded else 0)
    start_step = (2 if browser_preloaded else 1)

    for step in range(start_step, MAX_AGENT_STEPS + 1):
        # Call LLM silently (non-streaming) to decide next action
        response = await llm.complete(working, temperature=THINKING_TEMPERATURE)
        response = response.strip()

        # Check for errors from the LLM client
        if response.startswith("[ERROR]"):
            print(f"[AGENT] Error from LLM: {response}")
            yield {"type": "content", "content": response}
            return

        # Parse for tool calls
        tool_call = parse_tool_call(response)
        
        print(f"\n[AGENT] Step {step} - LLM Output length: len({response})")
        if tool_call:
            print(f"[AGENT] Parsed Tool: {tool_call.name} (Arg: {tool_call.argument})")
        else:
            print(f"[AGENT] No tool parsed. LLM answered directly.")

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
        result = await execute_tool(tool_call)
        tools_used += 1

        # Show status to the user
        status_payload = {
            "type": "status",
            "content": f"_{result.status_message}_ (step {step}/{MAX_AGENT_STEPS})",
        }
        if result.image_base64:
            status_payload["image"] = result.image_base64
            # Don't print the whole base64, just acknowledge it's there
            print(f"[AGENT] Sent Status: {result.tool_name} (has screenshot: YES, size: {len(result.image_base64)})")
        else:
            print(f"[AGENT] Sent Status: {result.tool_name} (has screenshot: NO)")
        yield status_payload

        # Append the exchange to the conversation
        working.append({"role": "assistant", "content": response})
        
        user_msg: dict[str, typing.Any] = {
            "role": "user",
            "content": _REFLECTION_PROMPT.format(
                step=step,
                max_steps=MAX_AGENT_STEPS,
                result=result.output,
            ),
        }
        if result.image_base64:
            user_msg["images"] = [result.image_base64]
            
        working.append(user_msg)

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