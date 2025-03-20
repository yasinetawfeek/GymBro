import { useMeeting } from "@videosdk.live/react-sdk";
import Hls from "hls.js";
import { useEffect, useRef, useState } from "react";
import { AlertCircle, RefreshCw, Volume2, VolumeX, Play } from "lucide-react";
import { motion } from "framer-motion";

function ViewerComponent({ playbackUrl, onPlay }) {
    const { hlsState } = useMeeting();
    const videoRef = useRef(null);
    const [hlsInstance, setHlsInstance] = useState(null);
    const [loaded, setLoaded] = useState(false);
    const [error, setError] = useState(null);
    const [isBuffering, setIsBuffering] = useState(false);
    const [isMuted, setIsMuted] = useState(false);
    const [showControls, setShowControls] = useState(true);
    const [isWaitingForPlay, setIsWaitingForPlay] = useState(false);
    
    const streamUrl = hlsState?.playbackHlsUrl || playbackUrl;

    const handlePlaybackError = (message) => {
        setError(message);
        setLoaded(false);
        setIsBuffering(false);
        setIsWaitingForPlay(true);
    };

    const startPlayback = async () => {
        if (!videoRef.current) return;
        
        try {
            setIsBuffering(true);
            setError(null);
            await videoRef.current.play();
            setLoaded(true);
            setIsBuffering(false);
            setIsWaitingForPlay(false);
            onPlay?.();
        } catch (err) {
            console.error("Playback failed:", err);
            handlePlaybackError("Click play to start the stream");
        }
    };

    const initializeHls = async (autoplay = true) => {
        if (!streamUrl || !videoRef.current) return;

        setError(null);
        setIsBuffering(true);

        try {
            if (Hls.isSupported()) {
                // Cleanup existing instance
                if (hlsInstance) {
                    hlsInstance.destroy();
                }

                const hls = new Hls({
                    maxBufferLength: 30,
                    liveSyncDuration: 10,
                    debug: false,
                    startLevel: -1, // Auto quality selection
                });

                // Setup error handling
                hls.on(Hls.Events.ERROR, (event, data) => {
                    console.warn("HLS error:", data);
                    if (data.fatal) {
                        switch(data.type) {
                            case Hls.ErrorTypes.NETWORK_ERROR:
                                console.log("Attempting to recover network error...");
                                hls.startLoad();
                                break;
                            case Hls.ErrorTypes.MEDIA_ERROR:
                                console.log("Attempting to recover media error...");
                                hls.recoverMediaError();
                                break;
                            default:
                                handlePlaybackError("Stream playback error. Please try refreshing.");
                                hls.destroy();
                                break;
                        }
                    }
                });

                // Setup success handling
                hls.on(Hls.Events.MANIFEST_PARSED, () => {
                    console.log("HLS manifest parsed successfully");
                    if (autoplay) {
                        startPlayback();
                    } else {
                        setIsBuffering(false);
                        setIsWaitingForPlay(true);
                    }
                });

                // Load the stream
                await new Promise((resolve, reject) => {
                    hls.loadSource(streamUrl);
                    hls.attachMedia(videoRef.current);
                    hls.on(Hls.Events.MANIFEST_PARSED, resolve);
                    hls.on(Hls.Events.ERROR, (event, data) => {
                        if (data.fatal) reject(data);
                    });
                });

                setHlsInstance(hls);
            } else if (videoRef.current.canPlayType("application/vnd.apple.mpegurl")) {
                // Native HLS support (Safari)
                videoRef.current.src = streamUrl;
                if (autoplay) {
                    await startPlayback();
                } else {
                    setIsBuffering(false);
                    setIsWaitingForPlay(true);
                }
            }
        } catch (err) {
            console.error("HLS initialization failed:", err);
            handlePlaybackError("Failed to initialize stream. Please try again.");
        }
    };

    // Initialize when stream URL changes
    useEffect(() => {
        if (streamUrl) {
            initializeHls(true);
        }
        return () => {
            if (hlsInstance) {
                hlsInstance.destroy();
            }
        };
    }, [streamUrl]);

    // Controls visibility
    useEffect(() => {
        let hideTimeout;
        const handleMouseMove = () => {
            setShowControls(true);
            clearTimeout(hideTimeout);
            hideTimeout = setTimeout(() => setShowControls(false), 3000);
        };

        document.addEventListener('mousemove', handleMouseMove);
        return () => {
            document.removeEventListener('mousemove', handleMouseMove);
            clearTimeout(hideTimeout);
        };
    }, []);

    const handleRetry = () => {
        setIsWaitingForPlay(false);
        initializeHls(true);
    };

    const handlePlayClick = () => {
        if (isWaitingForPlay) {
            startPlayback();
        }
    };

    return (
        <div className="relative w-full h-full bg-black">
            <video
                ref={videoRef}
                className="w-full h-full"
                playsInline
                controls={false}
            />

            {/* Play Button Overlay */}
            {isWaitingForPlay && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/50">
                    <motion.button
                        whileHover={{ scale: 1.1 }}
                        whileTap={{ scale: 0.95 }}
                        onClick={handlePlayClick}
                        className="w-20 h-20 rounded-full bg-purple-600 flex items-center justify-center text-white"
                    >
                        <Play className="w-10 h-10" />
                    </motion.button>
                </div>
            )}

            {/* Controls Overlay */}
            <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: showControls ? 1 : 0 }}
                className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black/80 to-transparent"
            >
                <div className="flex items-center justify-between">
                    <button
                        onClick={() => {
                            setIsMuted(!isMuted);
                            if (videoRef.current) {
                                videoRef.current.muted = !isMuted;
                            }
                        }}
                        className="text-white hover:text-purple-300 transition-colors"
                    >
                        {isMuted ? <VolumeX className="w-6 h-6" /> : <Volume2 className="w-6 h-6" />}
                    </button>

                    {error && (
                        <div className="flex items-center gap-2 text-red-400">
                            <AlertCircle className="w-5 h-5" />
                            <span className="text-sm">{error}</span>
                            <button
                                onClick={handleRetry}
                                className="ml-2 p-1 hover:bg-white/10 rounded-full transition-colors"
                            >
                                <RefreshCw className="w-5 h-5" />
                            </button>
                        </div>
                    )}
                </div>
            </motion.div>

            {/* Loading Spinner */}
            {isBuffering && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/50">
                    <motion.div
                        animate={{ rotate: 360 }}
                        transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                        className="w-12 h-12 border-4 border-purple-500 border-t-transparent rounded-full"
                    />
                </div>
            )}
        </div>
    );
}

export default ViewerComponent;