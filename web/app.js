/* ==========================================================================
   Spaghetti3D Web Monitor • Pure Client-Side JavaScript Logic
   ========================================================================== */

// Configuración de rutas WASM para TensorFlow.js TFLite
if (window.tflite) {
  tflite.setWasmPath('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.9/dist/');
} else {
  console.error("TensorFlow.js TFLite no pudo ser cargado desde el CDN.");
}

// Elements DOM
const elWebcam = document.getElementById('webcam');
const elCanvas = document.getElementById('canvas');
const elLaser = document.getElementById('laser');
const elViewportContainer = document.getElementById('viewport-container');
const elModelStatusBadge = document.getElementById('model-status');
const elStateBadge = document.getElementById('state-badge');
const elStateBadgeText = document.getElementById('state-badge-text');
const elAlertScreen = document.getElementById('alert-screen');

const btnStart = document.getElementById('btn-start');
const btnStop = document.getElementById('btn-stop');
const btnRecalibrate = document.getElementById('btn-recalibrate');
const btnSimulate = document.getElementById('btn-simulate');
const btnDismissAlert = document.getElementById('btn-dismiss-alert');
const selectCamera = document.getElementById('camera-select');

const elScoreText = document.getElementById('score-text');
const elScoreBar = document.getElementById('score-bar');
const elValThreshold = document.getElementById('val-threshold');
const elValBaseline = document.getElementById('val-baseline');
const elValFrames = document.getElementById('val-frames');
const elValFps = document.getElementById('val-fps');
const elHysIndicator = document.getElementById('hys-indicator');
const elHysValText = document.getElementById('hys-val-text');

// State Variables
let model = null;
let stream = null;
let isRunning = false;
let processedFrames = 0;
let calibrationLimit = 45;
let maxBaselineScore = 0.0;
let threshold = 0.35;
let isCalibrated = false;

let rollingQueue = [];
const queueMaxSize = 15;
const alertThresholdPercent = 0.6; // 60%
let isAlertActive = false;

let lastFrameTime = performance.now();
let fpsCounter = 0;
let currentFps = 0;

let audioCtx = null;
let alarmInterval = null;

// --- 1. CARGA DEL MODELO TFLITE ---
async function loadModel() {
  try {
    elModelStatusBadge.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Cargando Modelo...';
    // Cargamos el modelo estático desde la carpeta pública hospedada
    model = await tflite.loadTFLiteModel('models/anomaly_efficientnet.tflite');
    
    elModelStatusBadge.classList.add('loaded');
    elModelStatusBadge.innerHTML = '<i class="fa-solid fa-circle-check"></i> IA Lista (6.1MB)';
    console.log("Modelo TFLite cargado correctamente en navegador.");
    btnStart.disabled = false;
  } catch (error) {
    console.error("Error al cargar el modelo TFLite:", error);
    elModelStatusBadge.style.color = '#ff007f';
    elModelStatusBadge.innerHTML = '<i class="fa-solid fa-circle-xmark"></i> Error Carga Modelo';
  }
}

// Inicializar carga del modelo al arrancar
window.addEventListener('DOMContentLoaded', loadModel);

// --- 2. CONTROL DE CÁMARA (GetUserMedia) ---
async function startCamera() {
  if (stream) stopCamera();

  const facingModeValue = selectCamera.value;
  const constraints = {
    video: {
      facingMode: facingModeValue,
      width: { ideal: 640 },
      height: { ideal: 480 }
    },
    audio: false
  };

  try {
    stream = await navigator.mediaDevices.getUserMedia(constraints);
    elWebcam.srcObject = stream;
    
    // Esperar a que el video esté listo para ajustar el canvas
    elWebcam.onloadedmetadata = () => {
      elCanvas.width = elWebcam.videoWidth;
      elCanvas.height = elWebcam.videoHeight;
      console.log(`Cámara activa. Resolución real: ${elWebcam.videoWidth}x${elWebcam.videoHeight}`);
      
      isRunning = true;
      elViewportContainer.classList.add('scanning');
      updateStateBadge('CALIBRANDO', 'state-calibrating');
      
      btnStart.disabled = true;
      btnStop.disabled = false;
      btnRecalibrate.disabled = false;
      selectCamera.disabled = true;
      
      // Reiniciar variables de calibración y colas
      resetCalibration();
      
      // Iniciar bucle de procesamiento
      requestAnimationFrame(processFrameLoop);
    };
  } catch (error) {
    console.error("No se pudo acceder a la cámara:", error);
    alert("Error de cámara: Asegurate de dar permisos e iniciar desde HTTPS o Localhost.");
  }
}

function stopCamera() {
  isRunning = false;
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
  }
  elWebcam.srcObject = null;
  elViewportContainer.classList.remove('scanning');
  updateStateBadge('DESCONECTADO', 'state-disconnected');
  
  btnStart.disabled = false;
  btnStop.disabled = true;
  btnRecalibrate.disabled = true;
  selectCamera.disabled = false;
  
  // Limpiar Canvas
  const ctx = elCanvas.getContext('2d');
  ctx.clearRect(0, 0, elCanvas.width, elCanvas.height);
  
  stopAlarmSound();
  elAlertScreen.classList.remove('active');
  isAlertActive = false;
}

// --- 3. PROCESAMIENTO EN TIEMPO REAL CON TENSORFLOW.JS ---
async function processFrameLoop() {
  if (!isRunning) return;

  // Medición de FPS
  const now = performance.now();
  fpsCounter++;
  if (now - lastFrameTime >= 1000) {
    currentFps = fpsCounter;
    fpsCounter = 0;
    lastFrameTime = now;
    elValFps.textContent = currentFps;
  }

  const ctx = elCanvas.getContext('2d');
  
  // Dibujar el video en nuestro lienzo para manipulación
  ctx.clearRect(0, 0, elCanvas.width, elCanvas.height);
  
  // A. Calcular ROI (Cama de impresión) - Idéntico a detector.py
  const w = elCanvas.width;
  const h = elCanvas.height;
  const bx = Math.floor(w * 0.1);
  const by = Math.floor(h * 0.15);
  const bw = Math.floor(w * 0.8);
  const bh = Math.floor(h * 0.75);

  // Dibujar rectángulo verde de la zona de monitoreo
  ctx.strokeStyle = '#00ff87';
  ctx.lineWidth = 3;
  ctx.strokeRect(bx, by, bw, bh);
  
  // Etiqueta del visor
  ctx.fillStyle = '#00ff87';
  ctx.font = 'bold 12px Inter';
  ctx.fillText('CAMA BAJO MONITOREO', bx + 5, by - 8);

  // B. Recortar la zona para la IA y correr Inferencia
  try {
    // tf.tidy evita fugas de memoria limpiando tensores temporales automáticamente
    const score = await tf.tidy(() => {
      // 1. Obtener píxeles del video
      const tfImg = tf.browser.fromPixels(elWebcam);
      
      // 2. Recortar (slice) la ROI definida
      const cropped = tf.slice(tfImg, [by, bx, 0], [bh, bw, 3]);
      
      // 3. Redimensionar a 640x640 (formato YOLOv8 esperado)
      const resized = tf.image.resizeBilinear(cropped, [640, 640]);
      
      // 4. Normalizar píxeles de 0-255 a flotantes 0.0-1.0
      const normalized = resized.toFloat().div(tf.scalar(255.0));
      
      // 5. Expandir dimensiones a [1, 640, 640, 3] para lote
      const batched = normalized.expandDims(0);
      
      // 6. Inferencia
      const outputTensor = model.predict(batched);
      
      // 7. Leer salida de puntuación (Formato YOLOv8 [1, 8, 8400] o plano)
      const outShape = outputTensor.shape;
      if (outShape.length === 3 && outShape[1] === 8) {
        // En YOLOv8 las puntuaciones de las clases están a partir del índice 4 en la segunda dimensión
        const scoresSlice = tf.slice(outputTensor, [0, 4, 0], [1, 4, outShape[2]]);
        const maxScoreTensor = tf.max(scoresSlice);
        return maxScoreTensor.arraySync();
      } else {
        const flatTensor = tf.reshape(outputTensor, [-1]);
        return flatTensor.gather([0]).arraySync()[0];
      }
    });

    // C. Lógica de Autocalibración y Alertas
    handleScoring(score);

  } catch (error) {
    console.error("Error durante procesamiento de frame por la IA:", error);
  }

  // Siguiente iteración
  requestAnimationFrame(processFrameLoop);
}

// --- 4. MANEJO DE PUNTUACIÓN (Scoring, Calibración & Media Móvil) ---
function handleScoring(score) {
  // Asegurar límites numéricos válidos
  if (isNaN(score)) score = 0.0;
  
  // Actualizar barra de puntuación instantánea
  const scorePct = Math.min(100, Math.max(0, score * 100));
  elScoreBar.style.width = `${scorePct}%`;
  elScoreText.textContent = `${scorePct.toFixed(2)}%`;
  
  // Cambiar color de la barra en base a gravedad
  if (score < threshold) {
    elScoreBar.style.background = 'linear-gradient(90deg, var(--neon-green) 0%, var(--neon-cyan) 100%)';
    elScoreText.className = 'text-green';
  } else if (score < threshold + 0.15) {
    elScoreBar.style.background = 'linear-gradient(90deg, var(--neon-cyan) 0%, var(--neon-orange) 100%)';
    elScoreText.className = 'text-orange';
  } else {
    elScoreBar.style.background = 'linear-gradient(90deg, var(--neon-orange) 0%, var(--neon-pink) 100%)';
    elScoreText.className = 'text-pink';
  }

  // --- A. FASE DE AUTOCALIBRACIÓN ---
  if (!isCalibrated) {
    processedFrames++;
    elValFrames.textContent = `${processedFrames} / ${calibrationLimit}`;
    
    if (score > maxBaselineScore) {
      maxBaselineScore = score;
      elValBaseline.textContent = maxBaselineScore.toFixed(4);
    }
    
    // Al finalizar calibración inicial
    if (processedFrames >= calibrationLimit) {
      threshold = Math.min(0.88, Math.max(0.20, maxBaselineScore + 0.15));
      isCalibrated = true;
      elValThreshold.textContent = threshold.toFixed(4);
      updateStateBadge('SANO', 'state-ok');
      console.log(`Calibración completada. Ruido Base: ${maxBaselineScore.toFixed(4)} | Umbral Ajustado: ${threshold.toFixed(4)}`);
    }
    return;
  }

  // --- B. FASE DE MONITOREO ACTIVO ---
  // Clasificación binaria instantánea
  const prediction = score > threshold ? 1 : 0;
  
  // Añadir a la cola circular de media móvil
  rollingQueue.push(prediction);
  if (rollingQueue.length > queueMaxSize) {
    rollingQueue.shift();
  }
  
  // Calcular promedio móvil
  const sum = rollingQueue.reduce((a, b) => a + b, 0);
  const avgFail = sum / rollingQueue.length;
  
  elHysValText.textContent = avgFail.toFixed(2);
  
  // Dibujar estado de promedio (Histéresis)
  if (avgFail >= alertThresholdPercent) {
    elHysIndicator.className = 'dot-indicator dot-pink';
    if (!isAlertActive) {
      triggerEmergencyAlert(avgFail);
    }
  } else if (avgFail > 0.2) {
    elHysIndicator.className = 'dot-indicator dot-orange';
    updateStateBadge('ADVERTENCIA', 'state-calibrating');
  } else {
    elHysIndicator.className = 'dot-indicator dot-green';
    updateStateBadge('SANO', 'state-ok');
    
    // Desactivar alerta por histéresis si cae abajo del 50% del umbral de alerta (0.3)
    if (isAlertActive && avgFail < (alertThresholdPercent * 0.5)) {
      dismissEmergencyAlert();
    }
  }
}

// --- 5. ALERTAS CRÍTICAS (Visual, Sonido y Vibración) ---
function triggerEmergencyAlert(avg) {
  isAlertActive = true;
  updateStateBadge('FALLO CRÍTICO', 'state-error');
  elAlertScreen.classList.add('active');
  
  // Vibración táctil si se corre desde el celular
  if (navigator.vibrate) {
    navigator.vibrate([200, 100, 200, 100, 300]);
  }
  
  // Iniciar sonido de alarma
  startAlarmSound();
}

function dismissEmergencyAlert() {
  elAlertScreen.classList.remove('active');
  isAlertActive = false;
  rollingQueue = []; // Limpiamos la cola para resetear
  stopAlarmSound();
  updateStateBadge('SANO', 'state-ok');
}

// --- 6. UTILS & HELPERS ---
function updateStateBadge(text, className) {
  elStateBadge.className = `status-badge ${className}`;
  elStateBadgeText.textContent = text;
}

function resetCalibration() {
  processedFrames = 0;
  maxBaselineScore = 0.0;
  threshold = 0.35;
  isCalibrated = false;
  rollingQueue = [];
  isAlertActive = false;
  
  elValFrames.textContent = `0 / ${calibrationLimit}`;
  elValBaseline.textContent = '0.00';
  elValThreshold.textContent = '0.35';
  elHysValText.textContent = '0.00';
  elHysIndicator.className = 'dot-indicator dot-green';
}

// Generador de alarma sonora sintética (Web Audio API)
function startAlarmSound() {
  if (alarmInterval) return;
  
  // Inicializa contexto de audio solo tras interacción del usuario
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  
  if (audioCtx.state === 'suspended') {
    audioCtx.resume();
  }

  alarmInterval = setInterval(() => {
    try {
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      
      osc.type = 'sawtooth';
      osc.frequency.setValueAtTime(880, audioCtx.currentTime); // Pitch alto de sirena
      gain.gain.setValueAtTime(0.08, audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.35);
      
      osc.start();
      osc.stop(audioCtx.currentTime + 0.35);
    } catch (e) {
      console.warn("Audio no permitido aún por políticas del navegador:", e);
    }
  }, 500);
}

function stopAlarmSound() {
  if (alarmInterval) {
    clearInterval(alarmInterval);
    alarmInterval = null;
  }
}

// --- 7. LISTENERS DE INTERACCIÓN ---
btnStart.addEventListener('click', startCamera);
btnStop.addEventListener('click', stopCamera);

btnRecalibrate.addEventListener('click', () => {
  resetCalibration();
  updateStateBadge('CALIBRANDO', 'state-calibrating');
});

// Botón de Simulación de fallo para tests rápidos sin filamento real
let simulationInterval = null;
btnSimulate.addEventListener('click', () => {
  if (simulationInterval) {
    clearInterval(simulationInterval);
    simulationInterval = null;
    btnSimulate.innerHTML = '<i class="fa-solid fa-flask"></i> Simular Fallo (Prueba)';
    btnSimulate.className = 'btn btn-warning';
    resetCalibration();
    isCalibrated = true;
    updateStateBadge('SANO', 'state-ok');
    dismissEmergencyAlert();
  } else {
    btnSimulate.innerHTML = '<i class="fa-solid fa-flask-slash"></i> Detener Prueba';
    btnSimulate.className = 'btn btn-danger';
    
    // Forzamos calibración para saltearla rápido
    isCalibrated = true;
    threshold = 0.30;
    elValThreshold.textContent = threshold.toFixed(4);
    
    // Inyectamos valores altos simulando detección de espagueti cada 100ms
    simulationInterval = setInterval(() => {
      const simulatedScore = 0.65 + Math.random() * 0.25;
      handleScoring(simulatedScore);
    }, 150);
  }
});

btnDismissAlert.addEventListener('click', () => {
  if (simulationInterval) {
    // Si estaba simulando, apagamos el generador
    clearInterval(simulationInterval);
    simulationInterval = null;
    btnSimulate.innerHTML = '<i class="fa-solid fa-flask"></i> Simular Fallo (Prueba)';
    btnSimulate.className = 'btn btn-warning';
  }
  dismissEmergencyAlert();
});
