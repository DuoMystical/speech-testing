/**
 * Audio Processor - Handles microphone capture and audio processing
 */

class AudioProcessor {
    constructor(options = {}) {
        this.sampleRate = options.sampleRate || 16000;
        this.bufferSize = options.bufferSize || 4096;
        this.onAudioData = options.onAudioData || (() => {});
        this.onVisualizerData = options.onVisualizerData || (() => {});
        this.onError = options.onError || console.error;

        this.audioContext = null;
        this.mediaStream = null;
        this.sourceNode = null;
        this.workletNode = null;
        this.analyserNode = null;
        this.isRecording = false;
    }

    async start() {
        if (this.isRecording) return;

        try {
            // Request microphone access
            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    sampleRate: this.sampleRate,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                },
            });

            // Create audio context
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: this.sampleRate,
            });

            // Create source from microphone
            this.sourceNode = this.audioContext.createMediaStreamSource(this.mediaStream);

            // Create analyser for visualization
            this.analyserNode = this.audioContext.createAnalyser();
            this.analyserNode.fftSize = 256;
            this.analyserNode.smoothingTimeConstant = 0.8;

            // Connect source to analyser
            this.sourceNode.connect(this.analyserNode);

            // Create ScriptProcessor for audio data (fallback if AudioWorklet not available)
            // In production, use AudioWorklet for better performance
            this.scriptNode = this.audioContext.createScriptProcessor(this.bufferSize, 1, 1);

            this.scriptNode.onaudioprocess = (event) => {
                if (!this.isRecording) return;

                const inputData = event.inputBuffer.getChannelData(0);

                // Convert Float32 to Int16
                const int16Data = this.float32ToInt16(inputData);

                // Send to callback
                this.onAudioData(int16Data);
            };

            // Connect nodes
            this.sourceNode.connect(this.scriptNode);
            this.scriptNode.connect(this.audioContext.destination);

            // Start visualizer loop
            this.isRecording = true;
            this.startVisualizer();

            console.log('Audio recording started');
        } catch (error) {
            this.onError(error);
            throw error;
        }
    }

    stop() {
        if (!this.isRecording) return;

        this.isRecording = false;

        // Stop media stream
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.mediaStream = null;
        }

        // Disconnect nodes
        if (this.scriptNode) {
            this.scriptNode.disconnect();
            this.scriptNode = null;
        }

        if (this.sourceNode) {
            this.sourceNode.disconnect();
            this.sourceNode = null;
        }

        if (this.analyserNode) {
            this.analyserNode.disconnect();
            this.analyserNode = null;
        }

        // Close audio context
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }

        console.log('Audio recording stopped');
    }

    startVisualizer() {
        const visualize = () => {
            if (!this.isRecording || !this.analyserNode) return;

            const bufferLength = this.analyserNode.frequencyBinCount;
            const dataArray = new Uint8Array(bufferLength);
            this.analyserNode.getByteFrequencyData(dataArray);

            this.onVisualizerData(dataArray);

            requestAnimationFrame(visualize);
        };

        visualize();
    }

    float32ToInt16(float32Array) {
        const int16Array = new Int16Array(float32Array.length);
        for (let i = 0; i < float32Array.length; i++) {
            // Clamp value between -1 and 1
            const s = Math.max(-1, Math.min(1, float32Array[i]));
            // Convert to 16-bit integer
            int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        return int16Array;
    }

    getAnalyserData() {
        if (!this.analyserNode) return null;

        const bufferLength = this.analyserNode.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        this.analyserNode.getByteFrequencyData(dataArray);
        return dataArray;
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AudioProcessor;
}
