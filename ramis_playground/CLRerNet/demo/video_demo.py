import argparse
import cv2
import tempfile
import os
from mmdet.apis import init_detector
from libs.api.inference import inference_one_image
from libs.utils.visualizer import visualize_lanes

def parse_args():
    parser = argparse.ArgumentParser(description='CLRerNet video demo')
    parser.add_argument('video', help='input video file')
    parser.add_argument('config', help='config file')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out-file', default='result.avi', help='output video file')
    parser.add_argument('--device', default='cuda:0', help='device')
    return parser.parse_args()

def main(args):
    model = init_detector(args.config, args.checkpoint, device=args.device)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        tmp_path = tmp.name

    # get output size from first frame
    ret, frame = cap.read()
    cv2.imwrite(tmp_path, frame)
    src, preds = inference_one_image(model, tmp_path)
    dst = visualize_lanes(src, preds)
    out_h, out_w = dst.shape[:2]
    print(f"Output size: {out_w}x{out_h}, first frame lanes: {len(preds)}")

    # use XVID codec into .avi — most reliable with OpenCV
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(args.out_file, fourcc, fps, (out_w, out_h))
    out.write(dst)

    try:
        frame_idx = 1
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            print(f'Processing frame {frame_idx}/{total}', end='\r')
            cv2.imwrite(tmp_path, frame)
            src, preds = inference_one_image(model, tmp_path)
            dst = visualize_lanes(src, preds)
            out.write(dst)
    finally:
        cap.release()
        out.release()
        os.unlink(tmp_path)

    print(f'\nDone! Saved to {args.out_file}')

if __name__ == '__main__':
    args = parse_args()
    main(args)
