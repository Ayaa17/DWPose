from annotator.dwpose import DWposeDetector
import cv2
import os
import matplotlib.pyplot as plt


def pred_pose(image):
    pose = DWposeDetector()
    oriImg = cv2.imread(image)  # B,G,R order
    out = pose(oriImg)
    plt.imsave('result.jpg', out)


def pred_images(input_folder, output_folder):
    # 初始化模型
    pose = DWposeDetector()

    # 確保輸出資料夾存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍歷輸入資料夾中的所有圖片
    for filename in os.listdir(input_folder):
        # 確保是圖片檔案
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            # 讀取圖片
            input_path = os.path.join(input_folder, filename)
            oriImg = cv2.imread(input_path)  # B, G, R order

            if oriImg is None:
                print(f"跳過無法讀取的檔案: {input_path}")
                continue

            # 執行姿勢檢測
            out = pose(oriImg)

            # 保存結果
            output_path = os.path.join(output_folder, filename)
            plt.imsave(output_path, out)
            print(f"處理完成: {input_path} -> {output_path}")


if __name__ == "__main__":
    image = 'test_imgs/anime3.jpg'

    images_input_dir = 'test_imgs'
    images_output_dir = 'output'

    pred_pose(image)
    pred_images(images_input_dir, images_output_dir)
