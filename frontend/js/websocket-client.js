/**
 * WebSocket Client - Handles communication with STT server
 */

class WebSocketClient {
    constructor(options = {}) {
        this.url = options.url || this.getDefaultUrl();
        this.onConnect = options.onConnect || (() => {});
        this.onDisconnect = options.onDisconnect || (() => {});
        this.onTranscript = options.onTranscript || (() => {});
        this.onVadState = options.onVadState || (() => {});
        this.onError = options.onError || console.error;

        this.ws = null;
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
    }

    getDefaultUrl() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.host;
        return `${protocol}//${host}/ws`;
    }

    connect() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            console.log('WebSocket already connected');
            return;
        }

        console.log('Connecting to WebSocket:', this.url);

        try {
            this.ws = new WebSocket(this.url);
            this.ws.binaryType = 'arraybuffer';

            this.ws.onopen = () => {
                console.log('WebSocket connected');
                this.isConnected = true;
                this.reconnectAttempts = 0;
                this.onConnect();
            };

            this.ws.onclose = (event) => {
                console.log('WebSocket closed:', event.code, event.reason);
                this.isConnected = false;
                this.onDisconnect();

                // Attempt reconnection
                if (this.reconnectAttempts < this.maxReconnectAttempts) {
                    this.reconnectAttempts++;
                    console.log(`Reconnecting in ${this.reconnectDelay}ms (attempt ${this.reconnectAttempts})`);
                    setTimeout(() => this.connect(), this.reconnectDelay);
                }
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.onError(error);
            };

            this.ws.onmessage = (event) => {
                this.handleMessage(event);
            };
        } catch (error) {
            console.error('Failed to create WebSocket:', error);
            this.onError(error);
        }
    }

    disconnect() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        this.isConnected = false;
    }

    handleMessage(event) {
        try {
            // Handle text messages (JSON)
            if (typeof event.data === 'string') {
                const msg = JSON.parse(event.data);
                this.handleJsonMessage(msg);
            }
        } catch (error) {
            console.error('Error handling message:', error, event.data);
        }
    }

    handleJsonMessage(msg) {
        switch (msg.type) {
            case 'connected':
                console.log('Server connected:', msg.session_id);
                break;

            case 'vad':
                this.onVadState(msg.state);
                break;

            case 'transcript':
                this.onTranscript({
                    text: msg.text,
                    isFinal: msg.is_final,
                    confidence: msg.confidence,
                });
                break;

            case 'error':
                console.error('Server error:', msg.error);
                this.onError(new Error(msg.error));
                break;

            case 'pong':
                // Latency measurement response
                if (this.pingCallback) {
                    const latency = Date.now() - this.pingTime;
                    this.pingCallback(latency);
                    this.pingCallback = null;
                }
                break;

            case 'reset':
                console.log('Session reset:', msg.message);
                break;

            default:
                console.log('Unknown message type:', msg.type, msg);
        }
    }

    /**
     * Send audio data to server
     * @param {Int16Array} audioData - PCM audio samples
     */
    sendAudio(audioData) {
        if (!this.isConnected || !this.ws) {
            return;
        }

        try {
            // Send as binary data
            this.ws.send(audioData.buffer);
        } catch (error) {
            console.error('Error sending audio:', error);
        }
    }

    /**
     * Send control message to server
     * @param {string} type - Message type
     * @param {object} data - Additional data
     */
    sendControl(type, data = {}) {
        if (!this.isConnected || !this.ws) {
            return;
        }

        const msg = { type, ...data };
        this.ws.send(JSON.stringify(msg));
    }

    /**
     * Reset the session
     */
    reset() {
        this.sendControl('reset');
    }

    /**
     * Measure latency with ping
     * @param {function} callback - Called with latency in ms
     */
    ping(callback) {
        if (!this.isConnected) {
            callback(-1);
            return;
        }

        this.pingTime = Date.now();
        this.pingCallback = callback;
        this.sendControl('ping');
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = WebSocketClient;
}
