#!/bin/bash
echo "--- INICIANDO SISTEMA PROFESIONAL DE MONITOREO 3D (YOLOv11) ---"
echo ""
echo "Ingrese el nombre del video (ej. videoimp1.mp4) o escriba 0 para usar la camara web:"
read video_input

if [ -z "$video_input" ]; then
    video_input="0"
fi

echo ""
echo "[*] Iniciando motor de Inteligencia Artificial..."
python3 main.py --camera "$video_input" --classifier_model models/spaghetti_pro.pt
