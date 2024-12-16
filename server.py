from quart import Quart, request, jsonify, Response
import cv2
import numpy as np
import torch
from models.superpoint import SuperPoint
from models.superglue import SuperGlue
import os
import io


app = Quart(__name__)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


superpoint_config = {'weights': 'superpoint'}
superglue_config = {'weights': 'outdoor'}
superpoint = SuperPoint(superpoint_config).to(DEVICE).eval()
superglue = SuperGlue(superglue_config).to(DEVICE).eval()


def load_image_from_bytes(image_bytes, device, max_size=640):
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Не удалось декодировать изображение")
    scale = max_size / max(img.shape)
    new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
    img_resized = cv2.resize(img, new_size)
    img_tensor = torch.from_numpy(img_resized / 255.).float()[None, None].to(device)
    return img_tensor, img_resized


@app.route('/process-images', methods=['POST'])
async def process_images():
    files = await request.files

    if 'map_image' not in files or 'satellite_image' not in files:
        return jsonify({'error': 'Оба изображения (map_image, satellite_image) должны быть загружены'}), 400

    map_image = files['map_image'].read()
    satellite_image = files['satellite_image'].read()

    try:
        map_tensor, map_resized = load_image_from_bytes(map_image, DEVICE)
        satellite_tensor, satellite_resized = load_image_from_bytes(satellite_image, DEVICE)

        def extract_features(model, image):
            with torch.no_grad():
                pred = model({'image': image})
            return (
                pred['keypoints'][0].cpu().numpy(),
                pred['descriptors'][0].cpu().numpy(),
                pred['scores'][0].cpu().numpy()
            )

        keypoints_map, descriptors_map, scores_map = extract_features(superpoint, map_tensor)
        keypoints_sat, descriptors_sat, scores_sat = extract_features(superpoint, satellite_tensor)


        superglue_input = {
            'keypoints0': torch.from_numpy(keypoints_map).float().unsqueeze(0).to(DEVICE),
            'keypoints1': torch.from_numpy(keypoints_sat).float().unsqueeze(0).to(DEVICE),
            'descriptors0': torch.from_numpy(descriptors_map).float().unsqueeze(0).to(DEVICE),
            'descriptors1': torch.from_numpy(descriptors_sat).float().unsqueeze(0).to(DEVICE),
            'scores0': torch.from_numpy(scores_map).float().unsqueeze(0).to(DEVICE),
            'scores1': torch.from_numpy(scores_sat).float().unsqueeze(0).to(DEVICE),
            'image0': map_tensor,
            'image1': satellite_tensor
        }

        with torch.no_grad():
            matches = superglue(superglue_input)

        matches0 = matches['matches0'][0].cpu().numpy()
        valid_matches = matches0 != -1
        matched_keypoints_map = keypoints_map[valid_matches]
        matched_keypoints_sat = keypoints_sat[matches0[valid_matches]]

        if len(matched_keypoints_map) == 0:
            return jsonify({'message': 'Не найдено совпадений'}), 200

        x_coords = matched_keypoints_map[:, 0]
        y_coords = matched_keypoints_map[:, 1]
        x_min, y_min = int(np.min(x_coords)), int(np.min(y_coords))
        x_max, y_max = int(np.max(x_coords)), int(np.max(y_coords))

        map_with_bbox = cv2.cvtColor(map_resized, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(map_with_bbox, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', map_with_bbox)
        byte_stream = io.BytesIO(buffer)

        return Response(byte_stream.getvalue(), mimetype='image/jpeg')

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(port=5000)
