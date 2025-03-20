import { MeetingProvider } from "@videosdk.live/react-sdk";
import { useEffect, useState } from "react";
import { Loader2, AlertCircle, Play } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import ViewerComponent from "../components/Player";
function Meeting() {
    const [streamInfo, setStreamInfo] = useState(null);
    const [token, setToken] = useState(null);
    const [error, setError] = useState(null);
    const [isReady, setIsReady] = useState(false);
    const [isPlaying, setIsPlaying] = useState(false);

    const fetchWithRetry = async (url, options, retries = 3) => {
        try {
            const res = await fetch(url, options);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            return await res.json();
        } catch (err) {
            if (retries > 0) return fetchWithRetry(url, options, retries - 1);
            throw err;
        }
    };

    useEffect(() => {
        const loadData = async () => {
            try {
                // Using the API endpoints
                const [streamData, tokenData] = await Promise.all([
                    fetchWithRetry("http://localhost:8000/api/stream-info/"),
                    fetchWithRetry("http://localhost:8000/api/get-token/")
                ]);
                console.log("Successfully fetched stream data and token from API endpoints");
                
                setStreamInfo(streamData);
                
                if (!tokenData?.token) {
                    throw new Error("No token received from server");
                }
                
                setToken(String(tokenData.token));
            } catch (err) {
                console.error("Initialization failed:", err);
                setError("Failed to initialize stream. Please try again.");
            }
        };

        loadData();
    }, []);

    useEffect(() => {
        if (token) {
            setTimeout(() => setIsReady(true), 100);
        }
    }, [token]);

    if (error) {
        return (
            <div className="min-h-screen bg-gradient-to-br from-purple-900 to-purple-950 flex items-center justify-center p-4">
                <motion.div 
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="bg-white/10 backdrop-blur-lg rounded-xl p-6 flex items-center gap-3 text-red-300"
                >
                    <AlertCircle className="w-6 h-6" />
                    <span>{error}</span>
                </motion.div>
            </div>
        );
    }

    if (!streamInfo || !token || !isReady) {
        return (
            <div className="min-h-screen bg-gradient-to-br from-purple-900 to-purple-950 flex items-center justify-center">
                <motion.div 
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="text-purple-300 flex items-center gap-3"
                >
                    <Loader2 className="w-6 h-6 animate-spin" />
                    <span>Preparing your stream...</span>
                </motion.div>
            </div>
        );
    }

    const tokenString = typeof token === 'string' ? token.trim() : String(token).trim();
    
    const meetingConfig = {
        meetingId: "stream1",
        mode: "VIEWER",
        token: tokenString,
        name: "Viewer",
        micEnabled: false,
        webcamEnabled: false,
        participantId: "user-" + Date.now()
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-purple-900 to-purple-950">
            <div className="max-w-6xl mx-auto p-4">
                <motion.div 
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="rounded-2xl overflow-hidden bg-black/30 backdrop-blur-lg"
                >
                    <div className="relative aspect-video">
                        <MeetingProvider config={meetingConfig} token={tokenString}>
                            <ViewerComponent 
                                playbackUrl={streamInfo?.playback_url}
                                onPlay={() => setIsPlaying(true)}
                            />
                        </MeetingProvider>

                        <AnimatePresence>
                            {!isPlaying && (
                                <motion.div 
                                    initial={{ opacity: 0 }}
                                    animate={{ opacity: 1 }}
                                    exit={{ opacity: 0 }}
                                    className="absolute inset-0 flex items-center justify-center bg-black/50"
                                >
                                    <motion.button
                                        whileHover={{ scale: 1.1 }}
                                        whileTap={{ scale: 0.95 }}
                                        className="w-20 h-20 rounded-full bg-purple-600 flex items-center justify-center text-white"
                                        onClick={() => setIsPlaying(true)}
                                    >
                                        <Play className="w-10 h-10" />
                                    </motion.button>
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </div>

                    <div className="p-4 text-purple-200">
                        <h1 className="text-2xl font-semibold">Live Stream</h1>
                        <p className="text-purple-400 mt-1">
                            {streamInfo?.title || "Welcome to the live stream"}
                        </p>
                    </div>
                </motion.div>
            </div>
        </div>
    );
}

export default Meeting;