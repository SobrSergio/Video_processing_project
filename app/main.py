from flask import Flask, render_template, request, jsonify, send_file
import os
import logging
import uuid
import threading
from video_processing import filter_video

app = Flask(__name__)

UPLOAD_FOLDER = "/app/uploads"
PROCESSED_FOLDER = "/app/processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

tasks = {}

def process_video(input_path, threshold, min_size, task_id):
    logging.info(f"Начало обработки видео: {input_path}")
    result = filter_video(input_path, threshold=threshold, min_size=min_size, task_id=task_id)

    if result:
        logging.info(f"Видео обработано: {result['output_path']}")
        tasks[task_id]['status'] = 'SUCCESS'
        tasks[task_id]['output_path'] = result['output_path']
        tasks[task_id]['removed_frames'] = result['removed_frames']
        tasks[task_id]['removed_segments'] = result['removed_segments']  # Добавили информацию о сегментах
    else:
        tasks[task_id]['status'] = 'FAILURE'
        tasks[task_id]['error'] = 'Не удалось обработать видео.'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    video = request.files.get('video')
    threshold = float(request.form.get('threshold', 0.5))
    min_size = float(request.form.get('min_size', 0.1))

    if video:
        filename = f"{uuid.uuid4()}_{video.filename}"
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        video.save(input_path)

        logging.info(f"Видео сохранено: {input_path}")

        task_id = str(uuid.uuid4())
        tasks[task_id] = {
            'status': 'PENDING',
            'processed_frames': 0,
            'total_frames': 0
        }
        thread = threading.Thread(target=process_video, args=(input_path, threshold, min_size, task_id))
        thread.start()

        return jsonify({'task_id': task_id}), 202

    return "Ошибка при загрузке видео", 400

@app.route('/status/<task_id>')
def task_status(task_id):
    task = tasks.get(task_id)
    if task:
        response = {
            'state': task.get('status', 'PENDING'),
            'processed_frames': task.get('processed_frames', 0),
            'total_frames': task.get('total_frames', 1),
            'removed_frames': task.get('removed_frames', 0),
            'removed_segments': task.get('removed_segments', []),  # Добавили информацию о сегментах
            'percent_complete': task.get('percent_complete', 0),
            'result': task.get('output_path', '')
        }
        return jsonify(response)
    else:
        return jsonify({'state': 'PENDING', 'processed_frames': 0, 'total_frames': 1}), 200

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(PROCESSED_FOLDER, filename), as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
