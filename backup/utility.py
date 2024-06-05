temp_iq_dir = "../temp_iq_diagrams"

import os
import shutil

# 원본 및 대상 디렉토리 경로 설정
source_dir = '../temp_iq_diagrams'
target_dir = '../datasets/rfml2/constellation_selected_snrmod/constellation/size64'

# 원본 디렉토리 내의 모든 하위 디렉토리를 탐색
for subdir in os.listdir(source_dir):
    # 원본 및 대상 하위 디렉토리 경로 생성
    source_subdir_path = os.path.join(source_dir, subdir)
    target_subdir_path = os.path.join(target_dir, subdir)

    # 대상 하위 디렉토리가 없다면 생성
    if not os.path.exists(target_subdir_path):
        os.makedirs(target_subdir_path)

    # 각 하위 디렉토리에서 처음 100개의 이미지 파일을 대상 디렉토리로 복사
    for img_file in range(1, 101):
        source_img_path = os.path.join(source_subdir_path, f"{img_file}.png")
        target_img_path = os.path.join(target_subdir_path, f"{img_file}.png")

        # 파일 복사
        shutil.copy(source_img_path, target_img_path)