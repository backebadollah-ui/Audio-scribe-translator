import React, { useState, useRef, useCallback, useEffect, useMemo } from 'react';
import { GoogleGenAI, LiveServerMessage, Modality, Type } from '@google/genai';
import { AppStatus, Language } from './types';
import { LANGUAGES, TRANSCRIPTION_MODELS } from './constants';

const IS_API_KEY_MISSING = !process.env.API_KEY;

// Custom type for a timed transcription segment
interface TranscriptionSegment {
  startTime: number; // in milliseconds
  endTime: number; // in milliseconds
  text: string;
}

// Configuration for file chunking
const CHUNK_DURATION_SECONDS = 55; // Process audio in 55-second chunks

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
    <svg className={className} xmlns="http://www.w.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M12 14a2 2 0 0 0 2-2V6a2 2 0 0 0-4 0v6a2 2 0 0 0 2 2ZM15.9 12.1a1 1 0 1 0-2 0A2.94 2.94 0 0 1 12 15a2.94 2.94 0 0 1-1.9-4.9 1 1 0 1 0-2 0A5 5 0 0 0 12 17a5 5 0 0 0 3.9-7.9Z" /><path d="M12 2a1 1 0 0 0-1 1v8a1 1 0 0 0 2 0V3a1 1 0 0 0-1-1Z" /><path d="M19 10a1 1 0 0 0-1 1a6 6 0 0 1-12 0 1 1 0 0 0-2 0a8 8 0 0 0 7 7.93V21H9a1 1 0 0 0 0 2h6a1 1 0 0 0 0-2h-2v-2.07A8 8 0 0 0 19 11a1 1 0 0 0-1-1Z" /></svg>
);
const StopIcon = ({ className }: { className?: string }) => (
    <svg className={className} xmlns="http://www.w.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2Zm0 18a8 8 0 1 1 8-8a8 8 0 0 1-8 8Z" /><path d="M12 10a2 2 0 1 0 2 2a2 2 0 0 0-2-2Z" /></svg>
);
const UploadIcon = ({ className }: { className?: string }) => (
    <svg className={className} xmlns="http://www.w.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M9 16h6v-6h4l-7-7-7 7h4v6zm-4 2h14v2H5v-2z"/></svg>
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
    const [status, setStatus] = useState<AppStatus>(AppStatus.IDLE);
    const [segments, setSegments] = useState<TranscriptionSegment[]>([]);
    const [translatedSegments, setTranslatedSegments] = useState<TranscriptionSegment[]>([]);
    const [currentSegment, setCurrentSegment] = useState<{ text: string; startTime: number | null }>({ text: '', startTime: null });
    const [targetLanguage, setTargetLanguage] = useState<string>(LANGUAGES[1].code);
    const [error, setError] = useState<string | null>(null);
    const [progress, setProgress] = useState<number>(0);
    const [progressLabel, setProgressLabel] = useState<string>('');
    const [transcriptionModel, setTranscriptionModel] = useState<string>(() => {
        return localStorage.getItem('transcriptionModel') || TRANSCRIPTION_MODELS[0];
    });

    const recordingStartTimeRef = useRef<number>(0);
    const progressIntervalRef = useRef<number | null>(null);
    const streamRef = useRef<MediaStream | null>(null);
    const audioContextRef = useRef<AudioContext | null>(null);
    const scriptProcessorRef = useRef<ScriptProcessorNode | null>(null);
    const sessionPromiseRef = useRef<Promise<any> | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    
    const currentSegmentRef = useRef(currentSegment);
    currentSegmentRef.current = currentSegment;

    const ai = useMemo(() => new GoogleGenAI({ apiKey: process.env.API_KEY }), []);

    // Derived state for display
    const transcription = segments.map(s => s.text).join(' ');
    const translation = translatedSegments.map(s => s.text).join(' ');
    const transcriptionForDisplay = (segments.map(s => s.text).join(' ') + ' ' + currentSegment.text).trim();

    const resetState = () => {
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
        resetState();
        setStatus(AppStatus.RECORDING);
        recordingStartTimeRef.current = Date.now();
        
        try {
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
    }, [ai, handleStopRecording]);

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
        resetState();
        setStatus(AppStatus.PROCESSING);
        setProgressLabel('Preparing audio...');
        setProgress(0);
        setError(null);
    
        try {
            const audioCtx = new ((window as any).AudioContext || (window as any).webkitAudioContext)();
            const arrayBuffer = await file.arrayBuffer();
            const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer.slice(0));
            
            if (audioBuffer.duration <= CHUNK_DURATION_SECONDS) {
                setProgressLabel('Transcribing audio...');
                await transcribeAudioChunk(file.type, arrayBuffer, 0);
            } else {
                await chunkAndTranscribeAudio(audioBuffer);
            }
    
            completeProgress();
            setTimeout(() => {
                setStatus(AppStatus.FINISHED);
            }, 500);

        } catch (err: any) {
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
            processAndTranscribeFile(file);
        }
        if(event.target) {
            event.target.value = '';
        }
    };

    const handleTranslate = async () => {
        if (segments.length === 0) return;
        setStatus(AppStatus.TRANSLATING);
        startProgressSimulation('Translating text...');
        setTranslatedSegments([]);
        setError(null);
        try {
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

    const isProcessing = status === AppStatus.PROCESSING || status === AppStatus.TRANSLATING;
    const canTranslate = status === AppStatus.FINISHED && segments.length > 0;
    const canDownload = status === AppStatus.FINISHED && translatedSegments.length > 0;

    return (
        <div className="min-h-screen bg-gray-900 text-gray-200 flex flex-col items-center justify-center p-4 font-sans">
            <div className="w-full max-w-3xl mx-auto space-y-8">
                <header className="text-center">
                    <h1 className="text-4xl md:text-5xl font-bold text-white tracking-tight">Audio Scribe & Translate</h1>
                    <p className="mt-4 text-lg text-gray-400">Record, upload, transcribe, translate, and export with the power of Gemini.</p>
                </header>

                <main className="bg-gray-800 rounded-2xl shadow-2xl p-6 md:p-8 space-y-8">
                    {IS_API_KEY_MISSING && (
                        <div className="bg-yellow-900/50 border border-yellow-700 text-yellow-300 px-4 py-3 rounded-lg text-center">
                            <p><strong>Configuration Error:</strong> The API key is not set. Please ensure the <code>API_KEY</code> environment variable is configured to use the application.</p>
                        </div>
                    )}
                    
                    <div className="border-b border-gray-700 pb-6 space-y-4">
                        <h3 className="text-lg font-semibold text-center text-white">Settings</h3>
                        <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
                            <label htmlFor="model-select" className="font-medium text-gray-300 shrink-0">File Transcription & Translation Model:</label>
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
                                    Upload File
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
                    
                    {isProcessing && (
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
                                 <h3 className="font-semibold text-lg text-white">Translation</h3>
                                 <div className="bg-gray-900 rounded-lg p-4 h-48 overflow-y-auto text-gray-300 min-h-[12rem]">
                                     {translation || <span className="text-gray-500">Your translated text will appear here...</span>}
                                 </div>
                             </div>
                         </div>
                     )}

                    {(status === AppStatus.FINISHED && segments.length > 0) && (
                        <div className="border-t border-gray-700 pt-6 space-y-6">
                            {/* --- Translation Group --- */}
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

                            {/* --- Final Actions Group --- */}
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
                                        onClick={resetState}
                                        className="w-full sm:w-auto px-6 py-2.5 bg-gray-600 text-white font-semibold rounded-md shadow-md hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-opacity-50 transition"
                                    >
                                        Start Over
                                    </button>
                                </div>
                            </div>
                        </div>
                    )}
                </main>
            </div>
        </div>
    );
};

export default App;
