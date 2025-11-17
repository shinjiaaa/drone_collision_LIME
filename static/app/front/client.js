// static/client.js

document.addEventListener('DOMContentLoaded', (event) => {
    // UI 요소 참조
    const riskInfo = document.getElementById('riskInfo');
    const riskText = document.getElementById('riskText');
    const riskPercentage = document.getElementById('riskPercentage');
    const focusWarning = document.getElementById('focusWarning');
    
    let dataWs;
    let isFocused = false;
    let audioContext;
    let audioInitialized = false; // 오디오 초기화 상태 추적

    // --- 웹소켓 연결 및 메시지 처리 ---

    function connectWebSocket() {
        const host = window.location.host;
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        
        dataWs = new WebSocket(`${wsProtocol}//${host}/ws/data`);
        dataWs.onopen = () => console.log("[WS] Connected.");
        
        dataWs.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                // 데이터 수신 확인용 로그 (필요시 주석 해제)
                // console.log("[WS DEBUG] Data received:", data); 
                handleDataMessage(data);
            } catch (e) {
                console.error("[WS] Error parsing message:", e);
            }
        };
        
        dataWs.onclose = () => {
            console.log("[WS] Disconnected. Reconnecting...");
            setTimeout(connectWebSocket, 3000);
        };
    }

    function handleDataMessage(data) {
        if (data.type === 'telemetry') {
            updateTelemetry(data);
        }
    }

    // --- 텔레메트리 및 위험도 업데이트 (Problem 1 해결) ---

    function updateTelemetry(data) {
        const risk = data.risk;
        
        // 데이터 유효성 검사 강화
        if (!risk || typeof risk.max_conf !== 'number' || !risk.level || !risk.text) {
            console.warn("[Telemetry] Received invalid or incomplete risk data:", data);
            return;
        }

        // 실시간 위험 정보 업데이트
        riskText.textContent = risk.text;
        riskPercentage.textContent = `${(risk.max_conf * 100).toFixed(1)}%`;

        // 위험도에 따른 색상 변경 (className을 직접 설정하여 확실한 반영 보장)
        const newClassName = `alert text-center shadow-sm risk-${risk.level}`;
        if (riskInfo.className !== newClassName) {
            riskInfo.className = newClassName;
            // UI 변경 시점 확인 로그
            console.log(`[UI Update] Risk Level Changed: ${risk.level} (${(risk.max_conf * 100).toFixed(1)}%)`);
        }

        // 알림 이벤트 처리
        if (risk.alert_event) {
            handleAlertEvent(risk.alert_event);
        }
    }

    // --- 알림 처리 (사운드 및 TTS) (Problem 2 해결) ---

    // [강화됨] 사용자 상호작용 기반 오디오 및 TTS 초기화 (브라우저 정책 대응)
    function initAudio() {
        if (audioInitialized) return;

        try {
            if (!audioContext) {
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
            }

            const initialize = () => {
                 console.log("[Audio] AudioContext initialized. State:", audioContext.state);
                 audioInitialized = true;
                 
                 // TTS 음성 비동기 로딩 처리
                 if ('speechSynthesis' in window) {
                    const loadVoices = () => {
                        if (window.speechSynthesis.getVoices().length > 0) {
                            console.log("[TTS] Voices loaded.");
                            window.speechSynthesis.onvoiceschanged = null; // 리스너 제거
                            return true;
                        }
                        return false;
                    };

                    if (!loadVoices()) {
                        // 음성이 로드되지 않았으면 로드될 때까지 대기
                        window.speechSynthesis.onvoiceschanged = loadVoices;
                        window.speechSynthesis.getVoices(); // 로드 트리거
                    }
                 }
            };

            // AudioContext 상태에 따른 처리 (Promise 사용하여 비동기 완료 대기)
            if (audioContext.state === 'suspended') {
                console.log("[Audio] Attempting to resume AudioContext...");
                // resume()이 완료된 후 initialize 호출
                audioContext.resume().then(initialize).catch(e => console.error("[Audio] Failed to resume AudioContext:", e));
            } else if (audioContext.state === 'running') {
                initialize();
            }
        } catch (e) {
            console.error("[Audio] Initialization failed:", e);
        }
    }
    
    // 알림 이벤트 핸들러
    function handleAlertEvent(event) {
        // 알림 발생 및 오디오 상태 확인 로그
        console.log(`%c[Alert Triggered] ${event.level}. Audio Initialized: ${audioInitialized}`, 'color: red; font-weight: bold;');

        if (!audioInitialized) {
            console.warn("[Alert] Ignored: Audio not initialized. Please interact with the page.");
            return;
        }
        
        // 사운드 및 TTS 재생
        if (event.sound) playSound(event.sound);
        if (event.tts) speak(event.tts);
    }

    // 사운드 재생 (Web Audio API)
    function playSound(soundType) {
        if (!audioContext || !audioInitialized) return;

        // 재생 전 다시 한번 resume 확인 (안전 장치)
        if (audioContext.state === 'suspended') {
             audioContext.resume();
        }

        // 요구사항에 따른 사운드 재생 로직
        switch (soundType) {
            case 'alert_low_1': // 주의 (Conf >= 0.50): 톤이 낮은 알림음 1회 (A4=440Hz)
                playBeep(440, 0.2, 0.5);
                break;
            case 'alert_mid_2': // 경고 (Conf >= 0.60): 중간 톤의 알림음 2회 (E5=660Hz)
                playBeep(660, 0.2, 0.5);
                setTimeout(() => playBeep(660, 0.2, 0.5), 300);
                break;
            case 'alert_high_repeat': // 위험 (Conf >= 0.80): 높은 톤의 알림음 반복 (C6=1046.5Hz, 3회 패턴)
                playBeep(1046.5, 0.15, 0.6);
                setTimeout(() => playBeep(1046.5, 0.15, 0.6), 200);
                setTimeout(() => playBeep(1046.5, 0.15, 0.6), 400);
                break;
        }
    }

    function playBeep(frequency, duration, volume) {
        try {
            const oscillator = audioContext.createOscillator();
            const gainNode = audioContext.createGain();
            oscillator.connect(gainNode);
            gainNode.connect(audioContext.destination);
            oscillator.frequency.setValueAtTime(frequency, audioContext.currentTime);
            gainNode.gain.setValueAtTime(volume, audioContext.currentTime);
            oscillator.start(audioContext.currentTime);
            oscillator.stop(audioContext.currentTime + duration);
        } catch (e) {
            console.error("[Audio] Error playing beep:", e);
        }
    }

    // TTS 재생
    function speak(text) {
        if ('speechSynthesis' in window) {
            const voices = window.speechSynthesis.getVoices();
            // 음성이 로드되었는지 확인
            if (voices.length === 0) {
                console.warn("[TTS] Failed: Voices not loaded yet.");
                return;
            }

            window.speechSynthesis.cancel(); // 이전 TTS 중지
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.lang = 'ko-KR';
            
            // 한국어 음성 명시적 선택 시도
            const koreanVoice = voices.find(voice => voice.lang === 'ko-KR' || voice.lang === 'ko_KR');
            if (koreanVoice) {
                utterance.voice = koreanVoice;
            }
            
            window.speechSynthesis.speak(utterance);
        }
    }

    // --- 드론 제어 (키보드) ---
    // (키보드 제어 및 포커스 관리 로직은 변경 없음)
    
    const keyState = {};
    let controlInterval = null;

    function sendInstantCommand(command) {
        if (dataWs && dataWs.readyState === WebSocket.OPEN) {
            dataWs.send(JSON.stringify({ type: 'control_command', command: command }));
        }
    }

    window.addEventListener('keydown', (e) => {
        if (!isFocused) return;
        const key = e.key.toLowerCase();
        keyState[key] = true;
        
        if (key === 't') {
            sendInstantCommand('takeoff');
            e.preventDefault();
        } else if (key === 'l') {
            sendInstantCommand('land');
            e.preventDefault();
        }

        if (['arrowup', 'arrowdown', 'arrowleft', 'arrowright', 'w', 'a', 's', 'd'].includes(key)) {
            e.preventDefault();
        }
    });

    window.addEventListener('keyup', (e) => {
        if (!isFocused) return;
        keyState[e.key.toLowerCase()] = false;
    });

    function sendRCControl() {
        const controls = {
            lr: (keyState['a'] ? -1 : 0) + (keyState['d'] ? 1 : 0),
            fb: (keyState['s'] ? -1 : 0) + (keyState['w'] ? 1 : 0),
            ud: (keyState['arrowdown'] ? -1 : 0) + (keyState['arrowup'] ? 1 : 0),
            yv: (keyState['arrowleft'] ? -1 : 0) + (keyState['arrowright'] ? 1 : 0),
        };
        
        if (dataWs && dataWs.readyState === WebSocket.OPEN) {
            dataWs.send(JSON.stringify({ type: 'control_command', command: 'rc_control', data: controls }));
        }
    }

    function handleFocus() {
        isFocused = true;
        focusWarning.style.display = 'none';
        initAudio(); // 포커스 시 오디오 활성화 시도
        if (!controlInterval) {
            controlInterval = setInterval(sendRCControl, 50); // 20Hz
        }
    }

    function handleBlur() {
        isFocused = false;
        focusWarning.style.display = 'block';
        if (controlInterval) {
            clearInterval(controlInterval);
            controlInterval = null;
        }
        Object.keys(keyState).forEach(k => keyState[k] = false);
        sendRCControl(); // 정지 명령 전송
    }

    window.addEventListener('focus', handleFocus);
    window.addEventListener('blur', handleBlur);
    // [중요] 클릭 이벤트를 통해 사용자 인터랙션을 확실히 보장하고 오디오 초기화
    document.body.addEventListener('click', initAudio, { once: true });

    // 초기화
    connectWebSocket();
    if (document.hasFocus()) {
        handleFocus();
    } else {
        handleBlur();
    }
});