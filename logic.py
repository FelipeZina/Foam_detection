from ultralytics import YOLO
import cv2
import time
import torch 

MODEL_PATH = "Michelin.pt"
VIDEO_PATH = "ultimo.mp4"
TARGET_CLASS = "residuo"
TARGET_CLASS_LOWER = TARGET_CLASS.lower()
INPUT_IMG_SIZE = 320  

COLORS = {
    TARGET_CLASS_LOWER: (0, 0, 255)
}
DEFAULT_COLOR = (0, 255, 0)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f" Usando dispositivo: {DEVICE}")

try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    exit()

class_names = model.names
print("Classes que o modelo foi treinado para detectar:", class_names)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Erro: Não foi possível abrir o vídeo em '{VIDEO_PATH}'")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_original = cap.get(cv2.CAP_PROP_FPS)

print(f"Dimensões originais do vídeo: {frame_width}x{frame_height}")
print(f"FPS original: {fps_original}")

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  

FRAME_SKIP = 2  
frame_counter = 0

print("Reproduzindo vídeo. Pressione 'q' para sair, 'p' para pausar.")

frame_count = 0
start_time = time.time()
paused = False

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("Fim do vídeo ou falha ao ler o frame.")
            break
        
        frame_counter += 1
        
        if frame_counter % FRAME_SKIP != 0:
            continue
            
        frame_count += 1
        
        results = model.track(
            source=frame,
            persist=True,
            conf=0.15,  
            iou=0.5,
            verbose=False,
            imgsz=INPUT_IMG_SIZE,  
            device=DEVICE,
            half=True if DEVICE != 'cpu' else False, 
        )
        
        output_frame = frame.copy()  
        target_detectado_neste_frame = False

        if results and results[0].boxes and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu()
            class_ids = results[0].boxes.cls.cpu()
            confs = results[0].boxes.conf.cpu()
            
            ids = []
            if results[0].boxes.id is not None:
                ids = results[0].boxes.id.cpu()

            for i in range(len(boxes)):
                x1, y1, x2, y2 = map(int, boxes[i])
                class_id = int(class_ids[i])
                confidence = float(confs[i])
                
                class_name = class_names.get(class_id, "Desconhecido")
                color = COLORS.get(class_name.lower(), DEFAULT_COLOR)

                label = f"{class_name} {confidence:.0%}"
                if ids is not None and len(ids) > i:
                    label = f"{label}"

                cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(output_frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)  
                
                if class_name.lower() == TARGET_CLASS_LOWER:
                    target_detectado_neste_frame = True
        
        elapsed_time = time.time() - start_time
        current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        if target_detectado_neste_frame:
            status_text = "Residuo detectado!"
            status_color = (0, 255, 0)
        else:
            status_text = f"Nenhum {TARGET_CLASS} detectado"
            status_color = (0, 0, 255)
            
        font_scale = max(0.5, frame_width / 1000) 
        thickness = max(1, int(frame_width / 400))
        
        cv2.putText(output_frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, status_color, thickness)
        


        cv2.imshow("Detector de Resíduos - Tamanho Real (Pressione 'p' para pausar)", output_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        paused = not paused
        print("Pausado" if paused else "Continuando")
    elif key == ord('f'):  
        cv2.setWindowProperty("Detector de Resíduos - Tamanho Real (Pressione 'p' para pausar)", 
                             cv2.WND_PROP_FULLSCREEN, 
                             not cv2.getWindowProperty("Detector de Resíduos - Tamanho Real (Pressione 'p' para pausar)", cv2.WND_PROP_FULLSCREEN))

end_time = time.time()
total_time = end_time - start_time
print(f"\n--- Estatísticas Finais ---")
print(f"Total de frames processados: {frame_count}")
print(f"Tempo total de execução: {total_time:.2f}s")
print(f"FPS real de processamento: {frame_count / total_time:.2f}")

cap.release()
cv2.destroyAllWindows()
