// Chatbox functionality
let conversationHistory = [
    {
        role: 'system',
        content: `You are an AI assistant for a Bangalore house price prediction website. Dataset includes: Area Types (Super built-up/Built-up/Plot), Availability (Ready To Move/Immediate Possession/19-Dec/New Launch), Locations (Chikka Tirupathi, Electronic City Phase II, Hebbal, JP Nagar, KR Puram, Kothanur, Lingadheeranahalli, Marathahalli, Uttarahalli, Whitefield), Sizes (1-4 BHK), Sqft (507-3990), Bathrooms (1-5), Balconies (0-3), Prices (20.05-199.19 Lakhs). Help users understand how features affect pricing, provide Bangalore market insights, and guide form usage. Be concise and reference only dataset features.`
    }
];

// DOM Elements
const chatboxContainer = document.getElementById('chatboxContainer');
const chatboxBody = document.getElementById('chatboxBody');
const chatboxMessages = document.getElementById('chatboxMessages');
const chatboxInput = document.getElementById('chatboxInput');
const chatboxSendBtn = document.getElementById('chatboxSendBtn');
const chatboxToggle = document.getElementById('chatboxToggle');
const chatboxFloatBtn = document.getElementById('chatboxFloatBtn');
const chatboxHeader = document.getElementById('chatboxHeader');

// Configuration - Replace with your OpenRouter API key
const OPENROUTER_API_KEY = ''; // Add your API key here
const OPENROUTER_API_URL = 'https://openrouter.ai/api/v1/chat/completions';
const MODEL="google/gemini-2.0-flash-001";

// Initialize chatbox
let isMinimized = false;

// Toggle chatbox minimize/maximize
function toggleChatbox() {
    isMinimized = !isMinimized;
    if (isMinimized) {
        chatboxContainer.classList.add('minimized');
        chatboxFloatBtn.classList.add('show');
        chatboxToggle.textContent = '+';
    } else {
        chatboxContainer.classList.remove('minimized');
        chatboxFloatBtn.classList.remove('show');
        chatboxToggle.textContent = '−';
    }
}

// Event listeners for toggle
chatboxToggle.addEventListener('click', (e) => {
    e.stopPropagation();
    toggleChatbox();
});

chatboxHeader.addEventListener('click', () => {
    if (isMinimized) {
        toggleChatbox();
    }
});

chatboxFloatBtn.addEventListener('click', () => {
    toggleChatbox();
});

// Add message to chat
function addMessage(content, isUser = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = content;
    
    messageDiv.appendChild(contentDiv);
    chatboxMessages.appendChild(messageDiv);
    
    // Scroll to bottom
    chatboxMessages.scrollTop = chatboxMessages.scrollHeight;
    
    return messageDiv;
}

// Add loading message
function addLoadingMessage() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot-message loading';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = '';
    
    messageDiv.appendChild(contentDiv);
    chatboxMessages.appendChild(messageDiv);
    chatboxMessages.scrollTop = chatboxMessages.scrollHeight;
    
    return messageDiv;
}

// Remove loading message
function removeLoadingMessage(loadingElement) {
    if (loadingElement && loadingElement.parentNode) {
        loadingElement.parentNode.removeChild(loadingElement);
    }
}

// Send message to OpenRouter API
async function sendMessage(userMessage) { 
    if (!OPENROUTER_API_KEY) {
        addMessage('Error: Please configure your OpenRouter API key in app.js', false);
        return;
    }

    // Disable input while processing
    setSendButtonState(true);

    // Add user message to conversation
    conversationHistory.push({
        role: 'user',
        content: userMessage
    });

    // Show loading indicator
    const loadingElement = addLoadingMessage();

    try {
        const response = await fetch(OPENROUTER_API_URL, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${OPENROUTER_API_KEY}`,
                'Content-Type': 'application/json',
                'HTTP-Referer': window.location.origin,
                'X-Title': 'House Price Predictor'
            },
            body: JSON.stringify({
                model: MODEL,
                messages: conversationHistory
            })
        });

        // Parse response
        let data;
        try {
            data = await response.json();
        } catch (e) {
            // If response is not JSON, create error from status
            if (!response.ok) {
                let errorMessage = `API error: ${response.status} ${response.statusText}`;
                if (response.status === 402) {
                    errorMessage = 'Payment Required: Your OpenRouter account has insufficient credits. Please add credits to your account at https://openrouter.ai/credits';
                }
                throw new Error(errorMessage);
            }
            throw new Error('Failed to parse API response');
        }

        // Check for errors in response
        if (!response.ok) {
            let errorMessage = `API error: ${response.status} ${response.statusText}`;
            
            if (data.error?.message) {
                errorMessage = data.error.message;
            }
            
            // Handle specific error codes
            if (response.status === 402) {
                errorMessage = 'Payment Required: Your OpenRouter account has insufficient credits. Please add credits to your account at https://openrouter.ai/credits';
            } else if (response.status === 401) {
                errorMessage = 'Unauthorized: Invalid API key. Please check your OpenRouter API key.';
            } else if (response.status === 429) {
                errorMessage = 'Rate limit exceeded: Please wait a moment and try again.';
            }
            
            throw new Error(errorMessage);
        }
        
        // Remove loading message
        removeLoadingMessage(loadingElement);

        // Check if response has error
        if (data.error) {
            throw new Error(data.error.message || 'Unknown error from API');
        }

        // Extract assistant's response
        const assistantMessage = data.choices[0]?.message?.content || 'Sorry, I could not generate a response.';
        
        // Add assistant message to conversation
        conversationHistory.push({
            role: 'assistant',
            content: assistantMessage
        });

        // Display assistant's response
        addMessage(assistantMessage, false);

    } catch (error) {
        console.error('Error sending message:', error);
        
        // Remove loading message
        removeLoadingMessage(loadingElement);
        
        // Show user-friendly error message
        let userMessage = `Sorry, I encountered an error: ${error.message}`;
        
        if (error.message.includes('402') || error.message.includes('Payment Required')) {
            userMessage = '⚠️ Payment Required: Your OpenRouter account needs credits to use this service. Please visit https://openrouter.ai/credits to add credits to your account.';
        }
        
        addMessage(userMessage, false);
    } finally {
        // Re-enable input
        setSendButtonState(false);
    }
}

// Handle send button click
chatboxSendBtn.addEventListener('click', () => {
    const message = chatboxInput.value.trim();
    if (message) {
        addMessage(message, true);
        chatboxInput.value = '';
        sendMessage(message);
    }
});

// Handle Enter key press
chatboxInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        chatboxSendBtn.click();
    }
});

// Disable send button while processing
function setSendButtonState(disabled) {
    chatboxSendBtn.disabled = disabled;
    chatboxInput.disabled = disabled;
}

