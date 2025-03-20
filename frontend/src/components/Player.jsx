import { useMeeting } from "@videosdk.live/react-sdk";
import Hls from "hls.js";
import { useEffect, useRef } from "react";
function ViewerComponent() {
  const { hlsState } = useMeeting();
  const videoRef = useRef(null);

  useEffect(() => {
    if (hlsState?.playbackHlsUrl && videoRef.current) {
      if (Hls.isSupported()) {
        const hls = new Hls({
          maxBufferLength: 30,
          liveSyncDuration: 10
        });
        
        hls.loadSource(hlsState.playbackHlsUrl);
        hls.attachMedia(videoRef.current);
      } else if (videoRef.current.canPlayType("application/vnd.apple.mpegurl")) {
        videoRef.current.src = hlsState.playbackHlsUrl;
      }
    }
  }, [hlsState]);

  return <video ref={videoRef} controls autoPlay style={{ width: "100%" }} />;
}

export default ViewerComponent;