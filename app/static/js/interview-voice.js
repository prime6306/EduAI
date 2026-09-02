/**
 * VoiceIO — wraps the browser-native Web Speech API for Interview Prep.
 * No API key, no server round-trip for voice itself: SpeechRecognition
 * does STT client-side (Chrome/Edge), SpeechSynthesis does TTS — each
 * interviewer persona gets a slightly different pitch/rate passed into
 * speak() so Priya and Arjun don't sound identical even when the same
 * browser voice is used for both.
 * Degrades gracefully: if unsupported, callers fall back to typing.
 *
 * Chrome has a well-known quirk where `continuous: true` still stops the
 * recognizer on its own after a short silence (sometimes under a second),
 * even though nothing asked it to stop. We tell the difference between
 * "the user clicked stop" and "Chrome stopped it on its own" and, in the
 * latter case, transparently restart so a normal pause between sentences
 * doesn't end the answer.
 */
(function (global) {
  const SpeechRecognitionImpl = window.SpeechRecognition || window.webkitSpeechRecognition;
  const supported = !!SpeechRecognitionImpl && !!window.speechSynthesis;

  let recognizer = null;
  let listening = false;
  let userRequestedStop = false;
  let fatalErrorOccurred = false;
  let finalTranscript = "";
  let restartAttempts = 0;
  const MAX_AUTO_RESTARTS = 20; // generous — a thoughtful answer can have many pauses

  function speak(text, onEnd, opts) {
    if (!window.speechSynthesis) { if (onEnd) onEnd(); return; }
    window.speechSynthesis.cancel();
    const utter = new SpeechSynthesisUtterance(text);
    utter.rate = (opts && opts.rate) || 1.0;
    utter.pitch = (opts && opts.pitch) || 1.0;
    utter.onend = () => { if (onEnd) onEnd(); };
    utter.onerror = () => { if (onEnd) onEnd(); };
    window.speechSynthesis.speak(utter);
  }

  function _attachHandlers(cb) {
    recognizer.onstart = () => {
      listening = true;
      if (restartAttempts === 0 && cb.onStart) cb.onStart();
    };

    recognizer.onresult = (event) => {
      let interim = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const chunk = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          finalTranscript += chunk + " ";
        } else {
          interim += chunk;
        }
      }
      if (cb.onInterim) cb.onInterim((finalTranscript + interim).trim());
    };

    recognizer.onerror = (event) => {
      // 'no-speech' fires constantly on normal pauses between sentences —
      // treat it as a reason to restart, not a real error to surface.
      if (event.error === "no-speech" || event.error === "aborted") {
        return;
      }
      // Fatal errors (mic permission denied, no mic hardware, etc.) must
      // not trigger the auto-restart loop in onend below.
      fatalErrorOccurred = true;
      listening = false;
      if (cb.onError) cb.onError(new Error(event.error || "speech recognition error"));
    };

    recognizer.onend = () => {
      listening = false;
      if (userRequestedStop || fatalErrorOccurred) {
        if (cb.onFinal) cb.onFinal(finalTranscript.trim());
        if (cb.onStop) cb.onStop();
        return;
      }
      // Chrome stopped the recognizer on its own — restart transparently
      // so a brief pause doesn't end the answer.
      if (restartAttempts < MAX_AUTO_RESTARTS) {
        restartAttempts += 1;
        try {
          recognizer.start();
        } catch (e) {
          // start() can throw if called too soon after stop(); retry shortly.
          setTimeout(() => {
            try { recognizer.start(); } catch (e2) { /* give up quietly */ }
          }, 150);
        }
      } else if (cb.onFinal) {
        cb.onFinal(finalTranscript.trim());
        if (cb.onStop) cb.onStop();
      }
    };
  }

  function startListening({ onInterim, onFinal, onError, onStart, onStop }) {
    if (!supported) {
      if (onError) onError(new Error("Speech recognition not supported in this browser. Try Chrome desktop."));
      return;
    }
    if (listening) return;

    finalTranscript = "";
    userRequestedStop = false;
    fatalErrorOccurred = false;
    restartAttempts = 0;

    recognizer = new SpeechRecognitionImpl();
    recognizer.continuous = true;
    recognizer.interimResults = true;
    recognizer.lang = "en-US";

    _attachHandlers({ onInterim, onFinal, onError, onStart, onStop });
    recognizer.start();
  }

  function stopListening() {
    if (recognizer && listening) {
      userRequestedStop = true;
      recognizer.stop();
    }
  }

  global.VoiceIO = { supported, speak, startListening, stopListening };
})(window);
