import React, { useState, useRef, useCallback, useEffect, useMemo } from 'react';
import { GoogleGenAI, LiveServerMessage, Modality, Type } from '@google/genai';
import { AppStatus, Language } from './types.ts';
import { LANGUAGES, TRANSCRIPTION_MODELS } from './constants.ts';

const IS_API_KEY_MISSING = !process.env.API_KEY;

// Custom type for a timed transcription segment
interface TranscriptionSegment {
  startTime: number; // in milliseconds
  endTime: number; // in milliseconds
  text: string;
}

// Custom type for a parsed SRT segment
interface SrtSegment {
    index: number;
    timing: string;
    text: string;
}


// Configuration for file chunking
const CHUNK_DURATION_SECONDS = 55; // Process audio in 55-second chunks
const SRT_TRANSLATE_CHUNK_SIZE = 50; // Translate 50 SRT segments at a time
const MAX_AUDIO_FILE_SIZE_BYTES = 100 * 1024 * 1024; // 100 MB
const MAX_SRT_FILE_SIZE_BYTES = 5 * 1024 * 1024; // 5 MB


// Helper function to format milliseconds into SRT timestamp format HH:MM:SS,mmm
const formatSrtTime = (ms: number): string => {
  const date = new Date(0);
  date.setMilliseconds(ms);
  const hours = date.getUTCHours().toString().padStart(2, '0');
  const minutes = date.getUTCMinutes().toString().padStart(2, '0');
  const seconds = date.getUTCSeconds().toString().padStart(3, '0').slice(0, 2);
  const milliseconds = date.getUTCMilliseconds().toString().padStart(3, '0');
  return `${hours}:${minutes}:${seconds},${milliseconds}`;
};

// Base64 encoding function for audio data
function encode(bytes: Uint8Array): string {
  let binary = '';
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

// Base64 decoding function for audio data
function decode(base64: string): Uint8Array {
    const binaryString = atob(base64);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes;
}

/**
 * Decodes raw PCM audio data into an AudioBuffer for playback.
 * The Gemini TTS model returns audio at a 24000Hz sample rate.
 */
async function decodeAudioData(
    data: Uint8Array,
    ctx: AudioContext,
    sampleRate: number,
    numChannels: number,
): Promise<AudioBuffer> {
    const dataInt16 = new Int16Array(data.buffer);
    const frameCount = dataInt16.length / numChannels;
    const buffer = ctx.createBuffer(numChannels, frameCount, sampleRate);

    for (let channel = 0; channel < numChannels; channel++) {
        const channelData = buffer.getChannelData(channel);
        for (let i = 0; i < frameCount; i++) {
            channelData[i] = dataInt16[i * numChannels + channel] / 32768.0;
        }
    }
    return buffer;
}


/**
 * Converts an AudioBuffer to a WAV audio format Blob.
 * This is necessary for sending audio chunks to the API in a recognized format.
 */
const audioBufferToWav = (buffer: AudioBuffer): Blob => {
    const numOfChan = buffer.numberOfChannels;
    const length = buffer.length * numOfChan * 2 + 44;
    const bufferArray = new ArrayBuffer(length);
    const view = new DataView(bufferArray);
    const channels: Float32Array[] = [];
    let i: number, sample: number;
    let offset = 0;
    let pos = 0;

    const setUint16 = (data: number) => {
        view.setUint16(pos, data, true);
        pos += 2;
    };

    const setUint32 = (data: number) => {
        view.setUint32(pos, data, true);
        pos += 4;
    };

    // Write WAVE header
    setUint32(0x46464952); // "RIFF"
    setUint32(length - 8); // file length - 8
    setUint32(0x45564157); // "WAVE"

    // Write fmt chunk
    setUint32(0x20746d66); // "fmt "
    setUint32(16); // chunk length
    setUint16(1); // sample format (1 = PCM)
    setUint16(numOfChan);
    setUint32(buffer.sampleRate);
    setUint32(buffer.sampleRate * 2 * numOfChan); // byte rate
    setUint16(numOfChan * 2); // block align
    setUint16(16); // bits per sample

    // Write data chunk
    setUint32(0x61746164); // "data"
    setUint32(length - pos - 4);

    // Get PCM samples
    for (i = 0; i < buffer.numberOfChannels; i++) {
        channels.push(buffer.getChannelData(i));
    }

    while (pos < length - 44) {
        for (i = 0; i < numOfChan; i++) {
            sample = Math.max(-1, Math.min(1, channels[i][offset])); // clamp
            sample = (0.5 + sample < 0 ? sample * 32768 : sample * 32767) | 0; // scale to 16-bit
            view.setInt16(pos, sample, true);
            pos += 2;
        }
        offset++;
    }

    return new Blob([view], { type: 'audio/wav' });
};


// Icon components defined outside the main component to prevent re-creation on re-renders
const MicIcon = ({ className }: { className?: string }) => (
    <svg className={className} xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M12 14a2 2 0 0 0 2-2V6a2 2 0 0 0-4 0v6a2 2 0 0 0 2 2ZM15.9 12.1a1 1 0 1 0-2 0A2.94 2.94 0 0 1 12 15a2.94 2.94 0 0 1-1.9-4.9 1 1 0 1 0-2 0A5 5 0 0 0 12 17a5 5 0 0 0 3.9-7.9Z" /><path d="M12 2a1 1 0 0 0-1 1v8a1 1 0 0 0 2 0V3a1 1 0 0 0-1-1Z" /><path d="M19 10a1 1 0 0 0-1 1a6 6 0 0 1-12 0 1 1 0 0 0-2 0a8 8 0 0 0 7 7.93V21H9a1 1 0 0 0 0 2h6a1 1 0 0 0 0-2h-2v-2.07A8 8 0 0 0 19 11a1 1 0 0 0-1-1Z" /></svg>
);
const StopIcon = ({ className }: { className?: string }) => (
    <svg className={className} xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2Zm0 18a8 8 0 1 1 8-8a8 8 0 0 1-8 8Z" /><path d="M12 10a2 2 0 1 0 2 2a2 2 0 0 0-2-2Z" /></svg>
);
const UploadIcon = ({ className }: { className?: string }) => (
    <svg className={className} xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M9 16h6v-6h4l-7-7-7 7h4v6zm-4 2h14v2H5v-2z"/></svg>
);
const SpeakerWaveIcon = ({ className }: { className?: string }) => (
    <svg className={className} xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M3 10v4a1 1 0 0 0 1 1h1.41l4.24 4.24a1 1 0 0 0 1.41 0 1 1 0 0 0 0-1.41L7.83 14H5a1 1 0 0 1-1-1v-2a1 1 0 0 1 1-1h2.83l3.24-3.24a1 1 0 0 0 0-1.42 1 1 0 0 0-1.41 0L5.41 9H4a1 1 0 0 0-1 1ZM15.5 5.5a1 1 0 0 0-1.41 1.41 4 4 0 0 1 0 5.66 1 1 0 0 0 1.41 1.41 6 6 0 0 0 0-8.48Z"/><path d="M18.32 2.68a1 1 0 0 0-1.41 1.41 8 8 0 0 1 0 11.32 1 1 0 1 0 1.41-1.41 6 6 0 0 0 0-8.5Z"/></svg>
);
const StopCircleIcon = ({ className }: { className?: string }) => (
    <svg className={className} xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2a10 10 0 1 0 10 10A10.011 10.011 0 0 0 12 2Zm0 18a8 8 0 1 1 8-8 8.009 8.009 0 0 1-8 8Z"/><path d="M15 9H9v6h6Z"/></svg>
);
const SpinnerIcon = ({ className }: { className?: string }) => (
  <svg className={`animate-spin ${className}`} xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
  </svg>
);
const FileTextIcon = ({ className }: { className?: string }) => (
    <svg className={className} xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6zM16 18H8v-2h8v2zm0-4H8v-2h8v2zm-3-5V3.5L18.5 9H13z"/></svg>
);

const ProgressBar = ({ progress, label }: { progress: number; label: string }) => (
    <div className="w-full bg-gray-700 rounded-full h-6 relative overflow-hidden shadow-inner">
        <div
            className="bg-gradient-to-r from-blue-500 to-blue-600 h-full rounded-full transition-all duration-300 ease-out"
            style={{ width: `${progress}%` }}
        />
        <div className="absolute inset-0 flex items-center justify-center">
            <span className="text-white font-semibold text-sm drop-shadow-md">
                {label} ({Math.round(progress)}%)
            </span>
        </div>
    </div>
);

const App: React.FC = () => {
    // General state
    const [status, setStatus] = useState<AppStatus>(AppStatus.IDLE);
    const [error, setError] = useState<string | null>(null);
    const [activeTab, setActiveTab] = useState<'scribe' | 'srtTranslator'>('scribe');

    // Scribe & Translate state
    const [segments, setSegments] = useState<TranscriptionSegment[]>([]);
    const [translatedSegments, setTranslatedSegments] = useState<TranscriptionSegment[]>([]);
    const [currentSegment, setCurrentSegment] = useState<{ text: string; startTime: number | null }>({ text: '', startTime: null });
    const [targetLanguage, setTargetLanguage] = useState<string>(LANGUAGES[1].code);
    const [progress, setProgress] = useState<number>(0);
    const [progressLabel, setProgressLabel] = useState<string>('');
    const [ttsStatus, setTtsStatus] = useState<'idle' | 'loading' | 'playing'>('idle');
    const [transcriptionModel, setTranscriptionModel] = useState<string>(() => {
        return localStorage.getItem('transcriptionModel') || TRANSCRIPTION_MODELS[0];
    });

    // SRT Translator state
    const [srtFileName, setSrtFileName] = useState<string | null>(null);
    const [srtSegments, setSrtSegments] = useState<SrtSegment[]>([]);
    const [translatedSrtContent, setTranslatedSrtContent] = useState<string | null>(null);
    const [srtStatus, setSrtStatus] = useState<'idle' | 'uploading' | 'processing' | 'finished' | 'error'>('idle');
    const [srtError, setSrtError] = useState<string | null>(null);
    const [srtProgress, setSrtProgress] = useState<number>(0);
    const [srtProgressLabel, setSrtProgressLabel] = useState<string>('');

    // Refs
    const recordingStartTimeRef = useRef<number>(0);
    const progressIntervalRef = useRef<number | null>(null);
    const streamRef = useRef<MediaStream | null>(null);
    const audioContextRef = useRef<AudioContext | null>(null);
    const scriptProcessorRef = useRef<ScriptProcessorNode | null>(null);
    const sessionPromiseRef = useRef<Promise<any> | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const srtFileInputRef = useRef<HTMLInputElement>(null);
    const outputAudioContextRef = useRef<AudioContext | null>(null);
    const audioSourceRef = useRef<AudioBufferSourceNode | null>(null);
    const isTranscriptionCancelledRef = useRef<boolean>(false);
    
    const currentSegmentRef = useRef(currentSegment);
    currentSegmentRef.current = currentSegment;

    // Derived state for display
    const transcription = segments.map(s => s.text).join(' ');
    const translation = translatedSegments.map(s => s.text).join(' ');
    const transcriptionForDisplay = (segments.map(s => s.text).join(' ') + ' ' + currentSegment.text).trim();

    const resetScribeState = () => {
        setStatus(AppStatus.IDLE);
        setSegments([]);
        setTranslatedSegments([]);
        setCurrentSegment({ text: '', startTime: null });
        setError(null);
        recordingStartTimeRef.current = 0;
        if (progressIntervalRef.current) clearInterval(progressIntervalRef.current);
        progressIntervalRef.current = null;
        setProgress(0);
        setProgressLabel('');
        isTranscriptionCancelledRef.current = false;
    };
    
    useEffect(() => {
        localStorage.setItem('transcriptionModel', transcriptionModel);
    }, [transcriptionModel]);
    
    const cleanupLiveRecording = useCallback(async () => {
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
            streamRef.current = null;
        }
        if (scriptProcessorRef.current) {
            scriptProcessorRef.current.disconnect();
            scriptProcessorRef.current = null;
        }
        if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
            await audioContextRef.current.close();
            audioContextRef.current = null;
        }

        if (sessionPromiseRef.current) {
            try {
                const session = await sessionPromiseRef.current;
                session.close();
            } catch (e) {
                console.error("Error closing session:", e);
            }
            sessionPromiseRef.current = null;
        }
    }, []);

    // Unmount cleanup
    useEffect(() => {
        return () => {
            cleanupLiveRecording();
            if (progressIntervalRef.current) {
                clearInterval(progressIntervalRef.current);
            }
            if (audioSourceRef.current) {
                audioSourceRef.current.stop();
            }
            if (outputAudioContextRef.current && outputAudioContextRef.current.state !== 'closed') {
                outputAudioContextRef.current.close();
            }
        };
    }, [cleanupLiveRecording]);

    const handleStopRecording = useCallback(async () => {
        setStatus(AppStatus.PROCESSING);

        // Finalize any pending segment from live recording
        if (currentSegmentRef.current.text.trim() && currentSegmentRef.current.startTime !== null) {
            const lastSegment: TranscriptionSegment = {
                text: currentSegmentRef.current.text.trim(),
                startTime: currentSegmentRef.current.startTime,
                endTime: Date.now() - recordingStartTimeRef.current,
            };
            setSegments(prev => [...prev, lastSegment]);
            setCurrentSegment({ text: '', startTime: null });
        }

        await cleanupLiveRecording();
        setStatus(AppStatus.FINISHED);
    }, [cleanupLiveRecording]);

    const handleStartRecording = useCallback(async () => {
        resetScribeState();
        setStatus(AppStatus.RECORDING);
        recordingStartTimeRef.current = Date.now();
        
        try {
            const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
            const audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            streamRef.current = audioStream;

            sessionPromiseRef.current = ai.live.connect({
                model: 'gemini-2.5-flash-native-audio-preview-09-2025',
                callbacks: {
                    onopen: () => console.log('Session opened.'),
                    onmessage: (message: LiveServerMessage) => {
                        if (message.serverContent?.inputTranscription) {
                            const { text } = message.serverContent.inputTranscription;
                             setCurrentSegment(prev => {
                                const isNew = prev.text === '';
                                return {
                                    text: prev.text + text,
                                    startTime: isNew ? Date.now() - recordingStartTimeRef.current : prev.startTime,
                                };
                            });
                        }
                        if (message.serverContent?.turnComplete) {
                            if (currentSegmentRef.current.text.trim() && currentSegmentRef.current.startTime !== null) {
                                const newSegment: TranscriptionSegment = {
                                    text: currentSegmentRef.current.text.trim(),
                                    startTime: currentSegmentRef.current.startTime,
                                    endTime: Date.now() - recordingStartTimeRef.current,
                                };
                                setSegments(prev => [...prev, newSegment]);
                                setCurrentSegment({ text: '', startTime: null });
                            }
                        }
                    },
                    onerror: (e: ErrorEvent) => {
                        console.error('Session error:', e);
                        setError('A real-time connection error occurred. The recording has been stopped.');
                        handleStopRecording();
                    },
                    onclose: (e: CloseEvent) => {
                        console.log('Session closed.');
                    },
                },
                config: {
                    responseModalities: [Modality.AUDIO],
                    inputAudioTranscription: {},
                },
            });
            
            const context = new ((window as any).AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
            audioContextRef.current = context;
            const source = context.createMediaStreamSource(audioStream);
            const processor = context.createScriptProcessor(4096, 1, 1);
            scriptProcessorRef.current = processor;

            processor.onaudioprocess = (audioProcessingEvent) => {
                const inputData = audioProcessingEvent.inputBuffer.getChannelData(0);
                const l = inputData.length;
                const int16 = new Int16Array(l);
                for (let i = 0; i < l; i++) {
                    int16[i] = inputData[i] * 32768;
                }
                const pcmBlob = {
                    data: encode(new Uint8Array(int16.buffer)),
                    mimeType: 'audio/pcm;rate=16000',
                };
                
                sessionPromiseRef.current?.then((session) => {
                    session.sendRealtimeInput({ media: pcmBlob });
                });
            };
            source.connect(processor);
            processor.connect(context.destination);
            
        } catch (err: any) {
            console.error("Failed to start recording:", err);
            let errorMessage = 'Failed to start recording. Please try again.';
            if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
                errorMessage = 'Microphone access denied. Please enable microphone permissions in your browser settings and refresh the page.';
            } else if (err.message) {
                errorMessage = `Failed to start recording: ${err.message}`;
            }
            setError(errorMessage);
            setStatus(AppStatus.ERROR);
        }
    }, [handleStopRecording]);

    const startProgressSimulation = (label: string) => {
        setProgress(0);
        setProgressLabel(label);
        if (progressIntervalRef.current) clearInterval(progressIntervalRef.current);
    
        progressIntervalRef.current = window.setInterval(() => {
            setProgress(p => {
                if (p >= 95) {
                    if (progressIntervalRef.current) clearInterval(progressIntervalRef.current);
                    return 95;
                }
                const increment = p < 50 ? 5 : 10;
                return Math.min(p + increment, 95);
            });
        }, 400);
    };

    const completeProgress = () => {
        if (progressIntervalRef.current) clearInterval(progressIntervalRef.current);
        progressIntervalRef.current = null;
        setProgress(100);
    };
    
    const failProgress = () => {
        if (progressIntervalRef.current) clearInterval(progressIntervalRef.current);
        progressIntervalRef.current = null;
        setProgress(0);
        setProgressLabel('');
    };

    const transcribeAudioChunk = async (mimeType: string, buffer: ArrayBuffer, timeOffsetSeconds: number) => {
        if (isTranscriptionCancelledRef.current) return;
        
        const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
        const base64Data = encode(new Uint8Array(buffer));
        
        const audioPart = { inlineData: { mimeType, data: base64Data } };
        const textPart = { text: `Transcribe this audio file accurately. The output must be a valid JSON array of objects. Each object represents a sentence and must have "startTime", "endTime" (in seconds with 3 decimal places), and "text". Example: [{"startTime": 0.512, "endTime": 2.123, "text": "This is the first sentence."}]` };

        const schema = {
            type: Type.ARRAY,
            items: {
                type: Type.OBJECT,
                properties: {
                    startTime: { type: Type.NUMBER },
                    endTime: { type: Type.NUMBER },
                    text: { type: Type.STRING },
                },
                required: ['startTime', 'endTime', 'text'],
            },
        };

        const response = await ai.models.generateContent({
            model: transcriptionModel,
            contents: { parts: [audioPart, textPart] },
            config: {
                responseMimeType: 'application/json',
                responseSchema: schema,
            },
        });
        
        if (isTranscriptionCancelledRef.current) return;

        const parsedSegments = JSON.parse(response.text);
        const newSegments: TranscriptionSegment[] = parsedSegments.map((s: any) => ({
            text: s.text,
            startTime: Math.round((s.startTime + timeOffsetSeconds) * 1000),
            endTime: Math.round((s.endTime + timeOffsetSeconds) * 1000),
        }));

        setSegments(prev => [...prev, ...newSegments]);
    };

    const chunkAndTranscribeAudio = async (audioBuffer: AudioBuffer) => {
        const totalDuration = audioBuffer.duration;
        const numChunks = Math.ceil(totalDuration / CHUNK_DURATION_SECONDS);

        for (let i = 0; i < numChunks; i++) {
            if (isTranscriptionCancelledRef.current) {
                break;
            }

            setProgressLabel(`Transcribing chunk ${i + 1} of ${numChunks}...`);
            setProgress(((i) / numChunks) * 100);

            const offset = i * CHUNK_DURATION_SECONDS;
            const chunkDuration = Math.min(CHUNK_DURATION_SECONDS, totalDuration - offset);
            
            const frameOffset = Math.floor(offset * audioBuffer.sampleRate);
            const frameCount = Math.floor(chunkDuration * audioBuffer.sampleRate);
            
            const chunkContext = new ((window as any).AudioContext || (window as any).webkitAudioContext)();
            const chunkBuffer = chunkContext.createBuffer(
                audioBuffer.numberOfChannels,
                frameCount,
                audioBuffer.sampleRate
            );

            for (let channel = 0; channel < audioBuffer.numberOfChannels; channel++) {
                const channelData = audioBuffer.getChannelData(channel);
                const chunkChannelData = chunkBuffer.getChannelData(channel);
                chunkChannelData.set(channelData.subarray(frameOffset, frameOffset + frameCount));
            }

            const wavBlob = audioBufferToWav(chunkBuffer);
            const chunkArrayBuffer = await wavBlob.arrayBuffer();
            
            await transcribeAudioChunk('audio/wav', chunkArrayBuffer, offset);
        }
    };

    const processAndTranscribeFile = async (file: File) => {
        resetScribeState();
        setStatus(AppStatus.PROCESSING);
        setProgressLabel('Preparing audio...');
        setProgress(0);
        setError(null);
    
        try {
            const audioCtx = new ((window as any).AudioContext || (window as any).webkitAudioContext)();
            const arrayBuffer = await file.arrayBuffer();

            if (isTranscriptionCancelledRef.current) return;

            const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer.slice(0));
            
            if (audioBuffer.duration <= CHUNK_DURATION_SECONDS) {
                setProgressLabel('Transcribing audio...');
                await transcribeAudioChunk(file.type, arrayBuffer, 0);
            } else {
                await chunkAndTranscribeAudio(audioBuffer);
            }
    
            if (isTranscriptionCancelledRef.current) {
                resetScribeState();
                return;
            }

            completeProgress();
            setTimeout(() => {
                setStatus(AppStatus.FINISHED);
            }, 500);

        } catch (err: any) {
            if (isTranscriptionCancelledRef.current) {
                resetScribeState();
                return;
            }
            console.error("Error processing file:", err);
            const message = err.message || 'An unknown error occurred.';
            setError(`Failed to process the audio file: ${message}. Please check the file format or try again.`);
            setStatus(AppStatus.ERROR);
            failProgress();
        }
    };

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            if (!file.type.startsWith('audio/')) {
                setError('Invalid file type. Please upload an audio file.');
                setStatus(AppStatus.ERROR);
                if (event.target) event.target.value = '';
                return;
            }
            if (file.size > MAX_AUDIO_FILE_SIZE_BYTES) {
                setError(`File is too large. Maximum size is ${MAX_AUDIO_FILE_SIZE_BYTES / 1024 / 1024} MB.`);
                setStatus(AppStatus.ERROR);
                if (event.target) event.target.value = '';
                return;
            }
            processAndTranscribeFile(file);
        }
        if(event.target) {
            event.target.value = '';
        }
    };

    const handleCancelTranscription = () => {
        isTranscriptionCancelledRef.current = true;
        resetScribeState();
    };

    const handleTranslate = async () => {
        if (segments.length === 0) return;
        setStatus(AppStatus.TRANSLATING);
        startProgressSimulation('Translating text...');
        setTranslatedSegments([]);
        setError(null);
        try {
            const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
            const selectedLang = LANGUAGES.find(l => l.code === targetLanguage);

            const prompt = `Translate the "text" value in each object of the following JSON array to ${selectedLang?.name || 'the selected language'}.
Return a valid JSON array with the exact same structure and objects, including the same "startTime" and "endTime" values, but with the "text" values translated.
The number of objects in the output array must match the number of objects in the input array.

Input:
${JSON.stringify(segments)}
`;

            const schema = {
                type: Type.ARRAY,
                items: {
                    type: Type.OBJECT,
                    properties: {
                        startTime: { type: Type.NUMBER },
                        endTime: { type: Type.NUMBER },
                        text: { type: Type.STRING },
                    },
                    required: ['startTime', 'endTime', 'text'],
                },
            };
            
            const response = await ai.models.generateContent({
                model: transcriptionModel,
                contents: prompt,
                 config: {
                    responseMimeType: 'application/json',
                    responseSchema: schema,
                },
            });

            completeProgress();
            setTimeout(() => {
                const parsed = JSON.parse(response.text);
                setTranslatedSegments(parsed);
                setStatus(AppStatus.FINISHED);
            }, 500);
        } catch (err: any) {
            console.error("Translation error:", err);
            const message = err.message || 'An unknown error occurred.';
            setError(`Failed to translate text: ${message}. Please try again.`);
            setStatus(AppStatus.ERROR);
            failProgress();
        }
    };

    const handleStopSpeaking = () => {
        if (audioSourceRef.current) {
            audioSourceRef.current.stop();
            audioSourceRef.current.disconnect();
            audioSourceRef.current = null;
        }
        setTtsStatus('idle');
    };

    const handlePlayTranslation = async () => {
        if (!translation || ttsStatus === 'loading') return;
        if (ttsStatus === 'playing') {
            handleStopSpeaking();
            return;
        }

        setTtsStatus('loading');
        setError(null);

        try {
            const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
            const response = await ai.models.generateContent({
                model: 'gemini-2.5-flash-preview-tts',
                contents: [{ parts: [{ text: translation }] }],
                config: {
                    responseModalities: [Modality.AUDIO],
                },
            });
            
            const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
            if (!base64Audio) {
                throw new Error("No audio data received from API.");
            }

            if (!outputAudioContextRef.current) {
                outputAudioContextRef.current = new ((window as any).AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
            }
            const audioCtx = outputAudioContextRef.current;
            
            const decodedBytes = decode(base64Audio);
            const audioBuffer = await decodeAudioData(decodedBytes, audioCtx, 24000, 1);
            
            if (audioSourceRef.current) {
                audioSourceRef.current.stop();
            }

            const source = audioCtx.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(audioCtx.destination);
            source.onended = () => {
                setTtsStatus('idle');
                audioSourceRef.current = null;
            };
            source.start();
            audioSourceRef.current = source;
            setTtsStatus('playing');

        } catch (err: any) {
            console.error("TTS error:", err);
            const message = err.message || 'An unknown error occurred.';
            setError(`Failed to generate audio for translation: ${message}`);
            setTtsStatus('idle');
        }
    };
    
    const handleDownloadSrt = () => {
        if (translatedSegments.length === 0) return;
    
        let srtContent = '';
        
        for (let i = 0; i < translatedSegments.length; i++) {
            const segment = translatedSegments[i];
            
            const startTime = formatSrtTime(segment.startTime);
            const endTime = formatSrtTime(segment.endTime);
    
            srtContent += `${i + 1}\n`;
            srtContent += `${startTime} --> ${endTime}\n`;
            srtContent += `${segment.text}\n\n`;
        }
    
        const blob = new Blob([srtContent], { type: 'application/x-subrip;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'translation.srt';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    };

    // --- SRT Translator Logic ---
    const parseSrt = (srtContent: string): SrtSegment[] => {
        const segments: SrtSegment[] = [];
        // Normalize line endings and split into blocks, filtering out empty blocks
        const blocks = srtContent.trim().replace(/\r\n/g, '\n').split(/\n\n+/).filter(b => b.trim());

        for (const block of blocks) {
            const lines = block.split('\n');
            let index: number;
            let timing: string;
            let text: string;

            // A valid block must have at least 2 lines (timing and text)
            if (lines.length < 2) continue;

            let lineIndex = 0;
            
            // Check if the first line is a number (the index)
            if (/^\d+$/.test(lines[lineIndex].trim())) {
                index = parseInt(lines[lineIndex].trim(), 10);
                lineIndex++;
            } else {
                // If no index, generate one
                index = segments.length + 1;
            }

            // The next line should be the timing
            if (lines.length > lineIndex && lines[lineIndex].includes('-->')) {
                // Normalize millisecond separator from . to ,
                timing = lines[lineIndex].trim().replace('.', ',');
                lineIndex++;
            } else {
                // If no timing line is found where expected, this is not a valid block
                continue; 
            }

            // The rest of the lines are the text
            text = lines.slice(lineIndex).join('\n').trim();

            if (text) {
                 segments.push({ index, timing, text });
            }
        }
        return segments;
    };

    const reconstructSrt = (segments: SrtSegment[], translatedTexts: string[]): string => {
        return segments.map((segment, i) => {
            const translatedText = translatedTexts[i] || segment.text; // Fallback to original
            return `${segment.index}\n${segment.timing}\n${translatedText}\n\n`;
        }).join('');
    };
    
    const resetSrtState = () => {
        setSrtFileName(null);
        setSrtSegments([]);
        setTranslatedSrtContent(null);
        setSrtStatus('idle');
        setSrtError(null);
        setSrtProgress(0);
        setSrtProgressLabel('');
    };

    const handleSrtFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            resetSrtState();

            if (!file.name.toLowerCase().endsWith('.srt')) {
                setSrtError('Invalid file type. Please upload a .srt file.');
                setSrtStatus('error');
                if (event.target) event.target.value = '';
                return;
            }
            if (file.size > MAX_SRT_FILE_SIZE_BYTES) {
                setSrtError(`File is too large. Maximum size is ${MAX_SRT_FILE_SIZE_BYTES / 1024 / 1024} MB.`);
                setSrtStatus('error');
                if (event.target) event.target.value = '';
                return;
            }

            setSrtFileName(file.name);
            setSrtStatus('uploading');
            const reader = new FileReader();
            reader.onload = (e) => {
                const content = e.target?.result as string;
                const parsed = parseSrt(content);
                if (parsed.length === 0) {
                    setSrtError('Failed to parse SRT file. Please check the file format.');
                    setSrtStatus('error');
                    return;
                }
                setSrtSegments(parsed);
                setSrtStatus('idle');
            };
            reader.onerror = () => {
                 setSrtError('Error reading the SRT file.');
                 setSrtStatus('error');
            };
            reader.readAsText(file);
        }
         if (event.target) {
            event.target.value = '';
        }
    };
    
    const handleSrtTranslate = async () => {
        if (srtSegments.length === 0) return;
        setSrtStatus('processing');
        setSrtError(null);
        setTranslatedSrtContent(null);
        setSrtProgress(0);

        try {
            const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
            const selectedLang = LANGUAGES.find(l => l.code === targetLanguage);
            const allTextsToTranslate = srtSegments.map(s => s.text);
            const allTranslatedTexts: string[] = [];

            const numChunks = Math.ceil(allTextsToTranslate.length / SRT_TRANSLATE_CHUNK_SIZE);

            for (let i = 0; i < numChunks; i++) {
                const chunkStart = i * SRT_TRANSLATE_CHUNK_SIZE;
                const chunkEnd = chunkStart + SRT_TRANSLATE_CHUNK_SIZE;
                const chunk = allTextsToTranslate.slice(chunkStart, chunkEnd);
                
                setSrtProgressLabel(`Translating chunk ${i + 1} of ${numChunks}...`);
                setSrtProgress((i / numChunks) * 100);

                const prompt = `Translate each string in the following JSON array to ${selectedLang?.name || 'the selected language'}.
Return a valid JSON array containing only the translated strings, in the exact same order as the input. The number of strings in your output array must be exactly ${chunk.length}.

Input:
${JSON.stringify(chunk)}
`;
                const schema = {
                    type: Type.ARRAY,
                    items: { type: Type.STRING },
                };

                const response = await ai.models.generateContent({
                    model: transcriptionModel,
                    contents: prompt,
                    config: {
                        responseMimeType: 'application/json',
                        responseSchema: schema,
                    },
                });
                
                const translatedChunk = JSON.parse(response.text);
                if (translatedChunk.length !== chunk.length) {
                    throw new Error(`Translation API returned a mismatching number of items for chunk ${i + 1}. Expected ${chunk.length}, got ${translatedChunk.length}.`);
                }
                allTranslatedTexts.push(...translatedChunk);
            }
            
            setSrtProgress(100);
            setSrtProgressLabel('Translation complete!');

            if (allTranslatedTexts.length !== srtSegments.length) {
                throw new Error("Final translation returned a different number of segments than the input.");
            }
            
            const newSrtContent = reconstructSrt(srtSegments, allTranslatedTexts);
            setTranslatedSrtContent(newSrtContent);
            setSrtStatus('finished');

        } catch (err: any) {
             console.error("SRT Translation error:", err);
            const message = err.message || 'An unknown error occurred.';
            setSrtError(`Failed to translate SRT file: ${message}.`);
            setSrtStatus('error');
            setSrtProgress(0);
            setSrtProgressLabel('');
        }
    };

    const handleDownloadTranslatedSrt = () => {
        if (!translatedSrtContent || !srtFileName) return;
        const blob = new Blob([translatedSrtContent], { type: 'application/x-subrip;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        const nameParts = srtFileName.split('.');
        const extension = nameParts.pop();
        const baseName = nameParts.join('.');
        a.download = `${baseName}_${targetLanguage}.${extension}`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    };


    const isProcessing = status === AppStatus.PROCESSING || status === AppStatus.TRANSLATING;
    const canTranslate = status === AppStatus.FINISHED && segments.length > 0;
    const canDownload = status === AppStatus.FINISHED && translatedSegments.length > 0;
    const tabClass = (tabName: string) => `px-4 py-3 text-sm font-medium rounded-t-lg transition-colors focus:outline-none ${activeTab === tabName ? 'bg-gray-800 text-white' : 'bg-gray-700/50 text-gray-400 hover:bg-gray-800/60'}`;

    return (
        <div className="min-h-screen bg-gray-900 text-gray-200 flex flex-col items-center justify-center p-4 font-sans">
            <div className="w-full max-w-3xl mx-auto space-y-8">
                <header className="text-center">
                    <h1 className="text-4xl md:text-5xl font-bold text-white tracking-tight">Audio Scribe & Translate</h1>
                    <p className="mt-4 text-lg text-gray-400">Record, upload, transcribe, translate, and export with the power of Gemini.</p>
                </header>

                <main className="bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-2xl shadow-2xl">
                    {IS_API_KEY_MISSING && (
                        <div className="bg-yellow-900/50 border border-yellow-700 text-yellow-300 px-4 py-3 rounded-lg text-center m-6">
                            <p><strong>Configuration Error:</strong> The API key is not set. Please ensure the <code>API_KEY</code> environment variable is configured to use the application.</p>
                        </div>
                    )}
                    
                    <div className="flex border-b border-gray-700 px-6">
                        <button onClick={() => setActiveTab('scribe')} className={tabClass('scribe')}>Audio Scribe & Translate</button>
                        <button onClick={() => setActiveTab('srtTranslator')} className={tabClass('srtTranslator')}>SRT Translator</button>
                    </div>

                    <div className="p-6 md:p-8 space-y-8">
                        {activeTab === 'scribe' && (
                            <div className="space-y-8">
                                <div className="border-b border-gray-700 pb-6 space-y-4">
                                    <h3 className="text-lg font-semibold text-center text-white">Settings</h3>
                                    <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
                                        <label htmlFor="model-select" className="font-medium text-gray-300 shrink-0">AI Model:</label>
                                        <select
                                            id="model-select"
                                            value={transcriptionModel}
                                            onChange={(e) => setTranscriptionModel(e.target.value)}
                                            className="w-full sm:w-auto bg-gray-700 border-gray-600 text-white rounded-md px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition"
                                            disabled={isProcessing}
                                        >
                                            {TRANSCRIPTION_MODELS.map((model) => (
                                                <option key={model} value={model}>{model}</option>
                                            ))}
                                        </select>
                                    </div>
                                </div>

                                <div className="flex flex-col md:flex-row items-center justify-center gap-4">
                                    {status !== AppStatus.RECORDING ? (
                                        <>
                                            <button
                                                onClick={handleStartRecording}
                                                disabled={isProcessing || IS_API_KEY_MISSING}
                                                className="w-full md:w-auto flex items-center justify-center gap-2 px-6 py-3 bg-blue-600 text-white font-semibold rounded-full shadow-lg hover:bg-blue-700 focus:outline-none focus:ring-4 focus:ring-blue-500 focus:ring-opacity-50 transition-all duration-300 ease-in-out transform hover:scale-105 disabled:bg-gray-500 disabled:cursor-not-allowed disabled:scale-100"
                                            >
                                                <MicIcon className="w-6 h-6" />
                                                Record Audio
                                            </button>

                                            <span className="text-gray-400 font-medium">OR</span>

                                            <button
                                                onClick={() => fileInputRef.current?.click()}
                                                disabled={isProcessing || IS_API_KEY_MISSING}
                                                className="w-full md:w-auto flex items-center justify-center gap-2 px-6 py-3 bg-indigo-600 text-white font-semibold rounded-full shadow-lg hover:bg-indigo-700 focus:outline-none focus:ring-4 focus:ring-indigo-500 focus:ring-opacity-50 transition-all duration-300 ease-in-out transform hover:scale-105 disabled:bg-gray-500 disabled:cursor-not-allowed disabled:scale-100"
                                            >
                                                <UploadIcon className="w-6 h-6" />
                                                Upload Audio File
                                            </button>
                                            <input
                                                type="file"
                                                ref={fileInputRef}
                                                onChange={handleFileChange}
                                                accept="audio/*"
                                                className="hidden"
                                            />
                                        </>
                                    ) : (
                                        <button
                                            onClick={handleStopRecording}
                                            className="w-full md:w-auto flex items-center justify-center gap-2 px-6 py-3 bg-red-600 text-white font-semibold rounded-full shadow-lg hover:bg-red-700 focus:outline-none focus:ring-4 focus:ring-red-500 focus:ring-opacity-50 transition-all duration-300 ease-in-out transform hover:scale-105 animate-pulse"
                                        >
                                            <StopIcon className="w-6 h-6" />
                                            Stop Recording
                                        </button>
                                    )}
                                </div>
                                
                                {status === AppStatus.PROCESSING && (
                                    <div className="w-full px-2 md:px-0 space-y-3">
                                        <ProgressBar progress={progress} label={progressLabel} />
                                        <div className="flex justify-center">
                                            <button
                                                onClick={handleCancelTranscription}
                                                className="px-4 py-1.5 bg-gray-600 text-white font-semibold rounded-md shadow-md hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-opacity-50 transition"
                                            >
                                                Cancel
                                            </button>
                                        </div>
                                    </div>
                                )}
                                
                                {status === AppStatus.TRANSLATING && (
                                    <div className="w-full px-2 md:px-0">
                                        <ProgressBar progress={progress} label={progressLabel} />
                                    </div>
                                )}

                                {error && (
                                    <div className="bg-red-900/50 border border-red-700 text-red-300 px-4 py-3 rounded-lg text-center">
                                        <p><strong>Error:</strong> {error}</p>
                                    </div>
                                )}
                                
                                {(transcriptionForDisplay || translation) && (
                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                        <div className="space-y-2">
                                            <h3 className="font-semibold text-lg text-white">Transcription</h3>
                                            <div className="bg-gray-900 rounded-lg p-4 h-48 overflow-y-auto text-gray-300 min-h-[12rem]">
                                                {transcriptionForDisplay || <span className="text-gray-500">Your transcribed text will appear here...</span>}
                                            </div>
                                        </div>
                                        <div className="space-y-2">
                                            <div className="flex items-center justify-between">
                                                <h3 className="font-semibold text-lg text-white">Translation</h3>
                                                {translation && (
                                                    <button
                                                        onClick={handlePlayTranslation}
                                                        disabled={ttsStatus === 'loading' || IS_API_KEY_MISSING}
                                                        className="p-1.5 rounded-full text-gray-400 hover:bg-gray-700 hover:text-white focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-800 focus:ring-white transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                                                        aria-label={ttsStatus === 'playing' ? 'Stop translation' : 'Listen to translation'}
                                                        title={ttsStatus === 'playing' ? 'Stop translation' : 'Listen to translation'}
                                                    >
                                                        {ttsStatus === 'loading' && <SpinnerIcon className="w-5 h-5" />}
                                                        {ttsStatus === 'playing' && <StopCircleIcon className="w-5 h-5 text-green-400" />}
                                                        {ttsStatus === 'idle' && <SpeakerWaveIcon className="w-5 h-5" />}
                                                    </button>
                                                )}
                                            </div>
                                            <div className="bg-gray-900 rounded-lg p-4 h-48 overflow-y-auto text-gray-300 min-h-[12rem]">
                                                {translation || <span className="text-gray-500">Your translated text will appear here...</span>}
                                            </div>
                                        </div>
                                    </div>
                                )}

                                {(status === AppStatus.FINISHED && segments.length > 0) && (
                                    <div className="border-t border-gray-700 pt-6 space-y-6">
                                        <div className="space-y-3">
                                            <h3 className="font-semibold text-lg text-white" id="translate-heading">Translate Transcription</h3>
                                            <div className="flex flex-col sm:flex-row items-center gap-3" role="group" aria-labelledby="translate-heading">
                                                <select
                                                    value={targetLanguage}
                                                    onChange={(e) => setTargetLanguage(e.target.value)}
                                                    className="w-full sm:w-auto flex-grow bg-gray-700 border-gray-600 text-white rounded-md px-3 py-2.5 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition"
                                                >
                                                    {LANGUAGES.map((lang) => (
                                                        <option key={lang.code} value={lang.code}>{lang.name}</option>
                                                    ))}
                                                </select>
                                                <button
                                                    onClick={handleTranslate}
                                                    disabled={!canTranslate || isProcessing}
                                                    className="w-full sm:w-auto px-6 py-2.5 bg-green-600 text-white font-semibold rounded-md shadow-md hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-opacity-50 transition disabled:bg-gray-500 disabled:cursor-not-allowed"
                                                >
                                                    Translate
                                                </button>
                                            </div>
                                        </div>

                                        <div className="space-y-3">
                                            <h3 className="font-semibold text-lg text-white" id="actions-heading">Final Actions</h3>
                                            <div className="flex flex-col sm:flex-row items-center gap-3" role="group" aria-labelledby="actions-heading">
                                                <button
                                                    onClick={handleDownloadSrt}
                                                    disabled={!canDownload}
                                                    className="w-full sm:w-auto flex-grow px-6 py-2.5 bg-purple-600 text-white font-semibold rounded-md shadow-md hover:bg-purple-700 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-opacity-50 transition disabled:bg-gray-500 disabled:cursor-not-allowed"
                                                >
                                                    Download .SRT
                                                </button>
                                                <button
                                                    onClick={resetScribeState}
                                                    className="w-full sm:w-auto px-6 py-2.5 bg-gray-600 text-white font-semibold rounded-md shadow-md hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-opacity-50 transition"
                                                >
                                                    Start Over
                                                </button>
                                            </div>
                                        </div>
                                    </div>
                                )}
                            </div>
                        )}
                        {activeTab === 'srtTranslator' && (
                             <div className="space-y-8">
                                <div className="text-center">
                                    <h3 className="text-xl font-bold text-white">SRT File Translator</h3>
                                    <p className="text-gray-400 mt-1">Upload a subtitle file to translate its content.</p>
                                </div>

                                <div className="border-b border-gray-700 pb-6 space-y-4">
                                    <h3 className="text-lg font-semibold text-center text-white">Settings</h3>
                                    <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
                                        <label htmlFor="srt-model-select" className="font-medium text-gray-300 shrink-0">AI Model:</label>
                                        <select
                                            id="srt-model-select"
                                            value={transcriptionModel}
                                            onChange={(e) => setTranscriptionModel(e.target.value)}
                                            className="w-full sm:w-auto bg-gray-700 border-gray-600 text-white rounded-md px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition"
                                            disabled={srtStatus === 'processing'}
                                        >
                                            {TRANSCRIPTION_MODELS.map((model) => (
                                                <option key={model} value={model}>{model}</option>
                                            ))}
                                        </select>
                                    </div>
                                </div>
                                
                                <div className="bg-gray-900/50 p-6 rounded-lg border border-gray-700 space-y-4">
                                    <div className="flex flex-col items-center justify-center gap-4">
                                        <button
                                            onClick={() => srtFileInputRef.current?.click()}
                                            disabled={srtStatus === 'processing' || IS_API_KEY_MISSING}
                                            className="w-full sm:w-auto flex items-center justify-center gap-2 px-6 py-3 bg-indigo-600 text-white font-semibold rounded-full shadow-lg hover:bg-indigo-700 focus:outline-none focus:ring-4 focus:ring-indigo-500 focus:ring-opacity-50 transition-all duration-300 ease-in-out transform hover:scale-105 disabled:bg-gray-500 disabled:cursor-not-allowed"
                                        >
                                            <FileTextIcon className="w-6 h-6" />
                                            {srtFileName ? 'Upload a Different SRT File' : 'Upload SRT File'}
                                        </button>
                                        <input
                                            type="file"
                                            ref={srtFileInputRef}
                                            onChange={handleSrtFileChange}
                                            accept=".srt"
                                            className="hidden"
                                        />
                                        {srtFileName && <p className="text-gray-300 text-sm">Selected: <span className="font-medium text-white">{srtFileName}</span></p>}
                                    </div>

                                    {srtError && (
                                        <div className="bg-red-900/50 border border-red-700 text-red-300 px-4 py-3 rounded-lg text-center">
                                            <p><strong>Error:</strong> {srtError}</p>
                                        </div>
                                    )}

                                    {srtSegments.length > 0 && (
                                        <div className="border-t border-gray-700 pt-4 space-y-4">
                                            <div className="flex flex-col sm:flex-row items-center gap-3">
                                                <select
                                                    value={targetLanguage}
                                                    onChange={(e) => setTargetLanguage(e.target.value)}
                                                    className="w-full sm:w-auto flex-grow bg-gray-700 border-gray-600 text-white rounded-md px-3 py-2.5 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition"
                                                    disabled={srtStatus === 'processing'}
                                                >
                                                    {LANGUAGES.map((lang) => (
                                                        <option key={lang.code} value={lang.code}>{lang.name}</option>
                                                    ))}
                                                </select>
                                                <button
                                                    onClick={handleSrtTranslate}
                                                    disabled={srtStatus === 'processing'}
                                                    className="w-full sm:w-auto px-6 py-2.5 bg-green-600 text-white font-semibold rounded-md shadow-md hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-opacity-50 transition disabled:bg-gray-500 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                                                >
                                                    {srtStatus === 'processing' && <SpinnerIcon className="w-5 h-5" />}
                                                    Translate File
                                                </button>
                                            </div>
                                        </div>
                                    )}
                                    
                                    {srtStatus === 'processing' && (
                                        <div className="pt-2">
                                            <ProgressBar progress={srtProgress} label={srtProgressLabel} />
                                        </div>
                                    )}

                                    {srtStatus === 'finished' && translatedSrtContent && (
                                        <div className="border-t border-gray-700 pt-4 space-y-3 text-center">
                                            <p className="text-green-400 font-semibold">Translation successful!</p>
                                            <button
                                                onClick={handleDownloadTranslatedSrt}
                                                className="w-full sm:w-auto px-6 py-2.5 bg-purple-600 text-white font-semibold rounded-md shadow-md hover:bg-purple-700 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-opacity-50 transition"
                                            >
                                                Download Translated .SRT
                                            </button>
                                        </div>
                                    )}
                                </div>
                            </div>
                        )}
                    </div>
                </main>
            </div>
        </div>
    );
};

export default App;