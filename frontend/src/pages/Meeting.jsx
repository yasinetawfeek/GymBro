import { MeetingProvider } from "@videosdk.live/react-sdk";
import { useEffect, useState } from "react";
import ViewerComponent from "../components/Player";

// Rename the function to match the file name
function Meeting() {
    const [streamInfo, setStreamInfo] = useState(null);
    const [token, setToken] = useState(null);
    const [error, setError] = useState(null);
    // Add this state to track if token is fully ready
    const [isReady, setIsReady] = useState(false);

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
                // Parallel fetching
                const [streamData, tokenData] = await Promise.all([
                    fetchWithRetry("http://localhost:8000/stream-info/"),
                    fetchWithRetry("http://localhost:8000/get-token/")
                ]);
                
                console.log("Stream data:", streamData);
                console.log("Token data:", tokenData);
                
                setStreamInfo(streamData);
                
                // Make sure we're actually getting a token value
                if (!tokenData || !tokenData.token) {
                    throw new Error("No token received from server");
                }
                
                console.log("Setting token:", tokenData.token);
                setToken(tokenData.token);
            } catch (err) {
                console.error("Initialization failed:", err);
                setError("Failed to initialize stream. Please try again.");
            }
        };

        loadData();
    }, []);

    // Add effect to set ready state after token is loaded
    useEffect(() => {
        if (token) {
            // Give React a chance to fully update the state
            setTimeout(() => {
                setIsReady(true);
            }, 100);
        }
    }, [token]);

    // Add debugging logs
    useEffect(() => {
        console.log("Current token value:", token);
    }, [token]);
    
    useEffect(() => {
        console.log("Current streamInfo value:", streamInfo);
    }, [streamInfo]);

    if (error) return <div className="error">{error}</div>;
    if (!streamInfo || !token || !isReady) return <div>Loading stream...</div>;

    // Double check that we have a token before rendering the MeetingProvider
    console.log("Rendering with token:", token);
    
    // Use the token directly without any transformations
    return (
        <MeetingProvider
            config={{
                meetingId: "stream1",
                mode: "VIEWER",
                token: token.trim(), // Make sure there are no whitespace characters
                streamConfig: { // Changed from hls to streamConfig
                    ...(streamInfo.hls_config || {}),
                    autoStart: true,
                    controls: true
                }
            }}
        >
            <ViewerComponent playbackUrl={streamInfo.playback_url} />
        </MeetingProvider>
    );
}

// Export with the matching name
export default Meeting;