import cv2
import mediapipe as mp
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROCESSED_FOLDER = "/app/processed"
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

use_gpu = cv2.cuda.getCudaEnabledDeviceCount() > 0

if use_gpu:
    logging.info("GPU доступен. Используется CUDA.")
else:
    logging.info("GPU недоступен. Используется CPU.")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def detect_person(frame, min_size=0.1):
    """Функция для детектирования человека в кадре и проверки минимального размера объекта"""
    
    if use_gpu:
        frame_gpu = cv2.cuda_GpuMat()
        frame_gpu.upload(frame)
        frame_rgb_gpu = cv2.cuda.cvtColor(frame_gpu, cv2.COLOR_BGR2RGB)
        frame_rgb = frame_rgb_gpu.download()
    else:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = pose.process(frame_rgb)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        person_width = abs(right_shoulder.x - left_shoulder.x)
        person_height = abs(right_hip.y - left_shoulder.y)
        
        confidences = [landmark.visibility for landmark in landmarks]
        avg_confidence = sum(confidences) / len(confidences)

        logging.info(f"Обнаружены человек: ширина {person_width:.2f}, высота {person_height:.2f}, средняя уверенность: {avg_confidence:.2f}")

        if person_width > min_size or person_height > min_size:
            return True, avg_confidence 
        else:
            return False, avg_confidence
    return False, 0.0

def filter_video(input_video, threshold=0.5, min_size=0.1, task_id=None):
    from main import tasks

    cap = cv2.VideoCapture(input_video)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    output_filename = os.path.join(PROCESSED_FOLDER, f"processed_{os.path.basename(input_video)}")
    out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    removed_frames = 0
    removed_segments = []  # Для хранения начала и конца вырезанных фрагментов

    logging.info(f"Начало обработки видео: {input_video}")
    logging.info(f"Общее количество кадров: {total_frames}")

    if task_id:
        tasks[task_id]['total_frames'] = total_frames

    segment_start = None

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.info("Достигнут конец видео.")
            break

        person_detected, confidence = detect_person(frame, min_size=min_size)

        if not person_detected or confidence < threshold:
            if segment_start is not None:
                removed_segments.append((segment_start / fps, (frame_count - 1) / fps))  # Изменение здесь
                segment_start = None
            out.write(frame)
        else:
            removed_frames += 1
            #Будет писать, когда человек будет обнаружен
            # logging.info(f"Кадр {frame_count} удален: человек обнаружен с уверенностью {confidence:.2f}")
            if segment_start is None:
                segment_start = frame_count

        frame_count += 1

        if task_id:
            tasks[task_id]['processed_frames'] = frame_count
            tasks[task_id]['percent_complete'] = (frame_count / total_frames) * 100 

    if segment_start is not None:
        removed_segments.append((segment_start / fps, (frame_count - 1) / fps))  # Изменение здесь

    cap.release()
    out.release()
    
    logging.info(f"Обработка завершена. Удалено кадров: {removed_frames}, сохраненный файл: {output_filename}")
    
    return {'output_path': output_filename, 'removed_frames': removed_frames, 'removed_segments': removed_segments}
