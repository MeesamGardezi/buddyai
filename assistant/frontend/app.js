document.addEventListener('DOMContentLoaded', () => {
    const messagesContainer = document.getElementById('messages');
    const chatForm = document.getElementById('chat-form');
    const messageInput = document.getElementById('message-input');
    const sendBtn = document.getElementById('send-btn');
    const chatContainer = document.getElementById('chat-container');
    const clearBtn = document.getElementById('clear-btn');
    const welcomeMessage = document.querySelector('.welcome-message');

    let chatHistory = [];
    let isGenerating = false;

    // Configure marked to use highlight.js
    if (typeof marked !== 'undefined' && typeof hljs !== 'undefined') {
        marked.setOptions({
            highlight: function (code, lang) {
                const language = hljs.getLanguage(lang) ? lang : 'plaintext';
                return hljs.highlight(code, { language }).value;
            },
            langPrefix: 'hljs language-'
        });
    }

    // ── Input State & Auto-resize ──────────────────────

    function updateSendButton() {
        if (messageInput.value.trim() !== '' && !isGenerating) {
            sendBtn.disabled = false;
            sendBtn.classList.add('active');
        } else {
            sendBtn.disabled = true;
            sendBtn.classList.remove('active');
        }
    }

    messageInput.addEventListener('input', function () {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 200) + 'px';
        if (this.value.trim() === '') {
            this.style.height = 'auto';
        }
        updateSendButton();
    });

    // ── Enter to send (Shift+Enter for newline) ───────

    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!isGenerating && messageInput.value.trim()) {
                chatForm.dispatchEvent(new Event('submit'));
            }
        }
    });

    // ── Clear chat ─────────────────────────────────────

    clearBtn.addEventListener('click', () => {
        if (!isGenerating) {
            chatHistory = [];
            messagesContainer.innerHTML = '';
            if (welcomeMessage) welcomeMessage.style.display = 'block';
            updateSendButton();
        }
    });

    // ── Form submission ────────────────────────────────

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const text = messageInput.value.trim();
        if (!text || isGenerating) return;

        if (welcomeMessage) {
            welcomeMessage.style.display = 'none';
        }

        // Show user message
        appendUserMessage(text);
        chatHistory.push({ role: 'user', content: text });

        // Reset input
        messageInput.value = '';
        messageInput.style.height = 'auto';
        messageInput.focus();

        isGenerating = true;
        updateSendButton();

        let contentEl = null;
        let assistantContent = '';

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ messages: chatHistory }),
            });

            if (!response.ok) throw new Error('Network response was not ok');

            const reader = response.body.getReader();
            const decoder = new TextDecoder('utf-8');

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });

                for (const line of chunk.split('\n')) {
                    if (!line.startsWith('data: ')) continue;

                    const dataStr = line.substring(6);
                    if (dataStr === '[DONE]') continue;

                    let data;
                    try {
                        data = JSON.parse(dataStr);
                    } catch {
                        continue;
                    }

                    if (data.type === 'status') {
                        appendStatus(data.content);
                    } else if (data.type === 'content') {
                        if (!contentEl) {
                            contentEl = appendAssistantBubble();
                        }
                        assistantContent += data.content;
                        contentEl.innerHTML = marked.parse(assistantContent);
                    } else if (data.type === 'error') {
                        if (!contentEl) {
                            contentEl = appendAssistantBubble();
                        }
                        assistantContent += `\n\n**Error:** ${data.content}`;
                        contentEl.innerHTML = marked.parse(assistantContent);
                    }

                    scrollToBottom();
                }
            }

            if (assistantContent) {
                chatHistory.push({ role: 'assistant', content: assistantContent });
            }
        } catch (error) {
            console.error('Fetch error:', error);
            if (!contentEl) {
                contentEl = appendAssistantBubble();
            }
            contentEl.innerHTML = `<p style="color: #ef4444;">Connection error: ${error.message}</p>`;
        } finally {
            isGenerating = false;
            updateSendButton();
        }
    });

    // ── DOM helpers ────────────────────────────────────

    function appendUserMessage(text) {
        const msgWrapper = document.createElement('div');
        msgWrapper.className = 'message-wrapper user';

        const msgDiv = document.createElement('div');
        msgDiv.className = 'message user';

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.innerText = text;
        
        msgDiv.appendChild(contentDiv);
        msgWrapper.appendChild(msgDiv);
        messagesContainer.appendChild(msgWrapper);
        scrollToBottom();
    }

    function appendAssistantBubble() {
        const msgWrapper = document.createElement('div');
        msgWrapper.className = 'message-wrapper assistant';

        const msgDiv = document.createElement('div');
        msgDiv.className = 'message assistant';

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.innerHTML = '<span class="typing-indicator"><span></span><span></span><span></span></span>';

        msgDiv.appendChild(contentDiv);
        msgWrapper.appendChild(msgDiv);
        messagesContainer.appendChild(msgWrapper);
        scrollToBottom();
        return contentDiv;
    }

    function appendStatus(text) {
        const statusDiv = document.createElement('div');
        statusDiv.className = 'status-msg';
        statusDiv.innerHTML = marked.parseInline(text);
        messagesContainer.appendChild(statusDiv);
        scrollToBottom();
    }

    function scrollToBottom() {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
    
    updateSendButton();
});