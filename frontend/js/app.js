/**
 * STT Application - Main entry point
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const micBtn = document.getElementById('mic-btn');
    const micStatus = document.getElementById('mic-status');
    const connectionStatus = document.getElementById('connection-status');
    const statusText = connectionStatus.querySelector('.status-text');
    const vadStatus = document.getElementById('vad-status');
    const vadText = vadStatus.querySelector('.vad-text');
    const visualizerCanvas = document.getElementById('visualizer');
    const visualizerCtx = visualizerCanvas.getContext('2d');
    const transcript = document.getElementById('transcript');
    const partialTranscript = document.getElementById('partial-transcript');
    const copyBtn = document.getElementById('copy-btn');
    const clearBtn = document.getElementById('clear-btn');

    // Stats elements
    const sessionDuration = document.getElementById('session-duration');
    const audioProcessed = document.getElementById('audio-processed');
    const latencyDisplay = document.getElementById('latency');
    const wordCount = document.getElementById('word-count');

    // Settings elements
    const vadThreshold = document.getElementById('vad-threshold');
    const vadThresholdValue = document.getElementById('vad-threshold-value');
    const silenceTimeout = document.getElementById('silence-timeout');

    // State
    let isRecording = false;
    let fullTranscript = '';
    let sessionStartTime = null;
    let totalAudioMs = 0;
    let statsInterval = null;

    // Initialize WebSocket client
    const wsClient = new WebSocketClient({
        onConnect: () => {
            connectionStatus.classList.remove('disconnected');
            connectionStatus.classList.add('connected');
            statusText.textContent = 'Connected';
            micBtn.disabled = false;
        },
        onDisconnect: () => {
            connectionStatus.classList.remove('connected');
            connectionStatus.classList.add('disconnected');
            statusText.textContent = 'Disconnected';
            micBtn.disabled = true;
            stopRecording();
        },
        onTranscript: handleTranscript,
        onVadState: handleVadState,
        onError: (error) => {
            console.error('WebSocket error:', error);
            micStatus.textContent = 'Error: ' + error.message;
        },
    });

    // Initialize Audio Processor
    const audioProcessor = new AudioProcessor({
        sampleRate: 16000,
        bufferSize: 4096,
        onAudioData: (data) => {
            wsClient.sendAudio(data);
            totalAudioMs += (data.length / 16000) * 1000;
        },
        onVisualizerData: drawVisualizer,
        onError: (error) => {
            console.error('Audio error:', error);
            micStatus.textContent = 'Microphone error';
            stopRecording();
        },
    });

    // Handle transcript updates
    function handleTranscript(result) {
        if (result.isFinal) {
            // Append to full transcript
            if (fullTranscript && !fullTranscript.endsWith(' ')) {
                fullTranscript += ' ';
            }
            fullTranscript += result.text;

            // Update display
            transcript.innerHTML = '';
            transcript.textContent = fullTranscript;

            // Clear partial
            partialTranscript.textContent = '';

            // Update word count
            updateWordCount();
        } else {
            // Show partial transcript
            partialTranscript.textContent = result.text;
        }

        // Scroll to bottom
        transcript.parentElement.scrollTop = transcript.parentElement.scrollHeight;
    }

    // Handle VAD state changes
    function handleVadState(state) {
        if (state === 'speaking') {
            vadStatus.classList.add('speaking');
            vadText.textContent = 'Speaking...';
        } else {
            vadStatus.classList.remove('speaking');
            vadText.textContent = 'Listening...';
        }
    }

    // Draw audio visualizer
    function drawVisualizer(data) {
        const width = visualizerCanvas.width;
        const height = visualizerCanvas.height;
        const barCount = Math.min(data.length, 64);
        const barWidth = width / barCount;

        visualizerCtx.fillStyle = '#0f172a';
        visualizerCtx.fillRect(0, 0, width, height);

        for (let i = 0; i < barCount; i++) {
            const value = data[Math.floor(i * data.length / barCount)] / 255;
            const barHeight = value * height;

            // Gradient color based on value
            const hue = 240 + value * 120; // Blue to purple
            visualizerCtx.fillStyle = `hsl(${hue}, 70%, ${50 + value * 30}%)`;

            visualizerCtx.fillRect(
                i * barWidth,
                height - barHeight,
                barWidth - 1,
                barHeight
            );
        }
    }

    // Draw idle visualizer
    function drawIdleVisualizer() {
        const width = visualizerCanvas.width;
        const height = visualizerCanvas.height;

        visualizerCtx.fillStyle = '#0f172a';
        visualizerCtx.fillRect(0, 0, width, height);

        // Draw a flat line
        visualizerCtx.strokeStyle = '#334155';
        visualizerCtx.lineWidth = 2;
        visualizerCtx.beginPath();
        visualizerCtx.moveTo(0, height / 2);
        visualizerCtx.lineTo(width, height / 2);
        visualizerCtx.stroke();
    }

    // Start recording
    async function startRecording() {
        if (isRecording) return;

        try {
            await audioProcessor.start();
            isRecording = true;
            micBtn.classList.add('recording');
            micStatus.textContent = 'Recording...';
            vadText.textContent = 'Listening...';

            // Start session timer
            sessionStartTime = Date.now();
            startStatsUpdate();

            // Reset session on server
            wsClient.reset();
        } catch (error) {
            console.error('Failed to start recording:', error);
            micStatus.textContent = 'Failed to access microphone';
        }
    }

    // Stop recording
    function stopRecording() {
        if (!isRecording) return;

        audioProcessor.stop();
        isRecording = false;
        micBtn.classList.remove('recording');
        micStatus.textContent = 'Click to start';
        vadStatus.classList.remove('speaking');
        vadText.textContent = 'Waiting...';

        // Stop stats update
        stopStatsUpdate();

        // Draw idle visualizer
        drawIdleVisualizer();
    }

    // Toggle recording
    function toggleRecording() {
        if (isRecording) {
            stopRecording();
        } else {
            startRecording();
        }
    }

    // Update word count
    function updateWordCount() {
        const words = fullTranscript.trim().split(/\s+/).filter(w => w.length > 0);
        wordCount.textContent = words.length;
    }

    // Format time as M:SS
    function formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    // Update stats display
    function updateStats() {
        if (sessionStartTime) {
            const elapsed = (Date.now() - sessionStartTime) / 1000;
            sessionDuration.textContent = formatTime(elapsed);
        }

        audioProcessed.textContent = (totalAudioMs / 1000).toFixed(1) + 's';

        // Measure latency
        wsClient.ping((latency) => {
            if (latency >= 0) {
                latencyDisplay.textContent = latency + 'ms';
            }
        });
    }

    function startStatsUpdate() {
        if (statsInterval) return;
        statsInterval = setInterval(updateStats, 1000);
        updateStats();
    }

    function stopStatsUpdate() {
        if (statsInterval) {
            clearInterval(statsInterval);
            statsInterval = null;
        }
    }

    // Copy transcript to clipboard
    function copyTranscript() {
        if (!fullTranscript) return;

        navigator.clipboard.writeText(fullTranscript).then(() => {
            const originalText = copyBtn.innerHTML;
            copyBtn.innerHTML = '<span style="color: #22c55e">Copied!</span>';
            setTimeout(() => {
                copyBtn.innerHTML = originalText;
            }, 1500);
        });
    }

    // Clear transcript
    function clearTranscript() {
        fullTranscript = '';
        transcript.innerHTML = '<span class="placeholder">Your transcription will appear here...</span>';
        partialTranscript.textContent = '';
        wordCount.textContent = '0';
        totalAudioMs = 0;
        audioProcessed.textContent = '0.0s';
        wsClient.reset();
    }

    // Resize visualizer canvas
    function resizeVisualizer() {
        const container = visualizerCanvas.parentElement;
        visualizerCanvas.width = container.clientWidth;
        if (!isRecording) {
            drawIdleVisualizer();
        }
    }

    // Event Listeners
    micBtn.addEventListener('click', toggleRecording);
    copyBtn.addEventListener('click', copyTranscript);
    clearBtn.addEventListener('click', clearTranscript);

    vadThreshold.addEventListener('input', () => {
        vadThresholdValue.textContent = vadThreshold.value;
    });

    window.addEventListener('resize', resizeVisualizer);

    // Initialize
    wsClient.connect();
    resizeVisualizer();
    drawIdleVisualizer();

    // Handle page visibility
    document.addEventListener('visibilitychange', () => {
        if (document.hidden && isRecording) {
            // Optionally pause when tab is hidden
            // stopRecording();
        }
    });

    // Cleanup on page unload
    window.addEventListener('beforeunload', () => {
        if (isRecording) {
            stopRecording();
        }
        wsClient.disconnect();
    });
});
