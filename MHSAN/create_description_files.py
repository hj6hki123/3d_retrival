import os

def create_description_files(root_dir):
    # 取得所有類別（資料夾名稱）
    categories = sorted(os.listdir(root_dir))
    count = 0
    map_list = set()
    for category in categories:
        category_path = os.path.join(root_dir, category)
        if not os.path.isdir(category_path):
            continue

        # 對每個分割資料夾,通常是 "train" 與 "test"
        for split in ["train", "test"]:
            split_path = os.path.join(category_path, split)
            if not os.path.exists(split_path):
                continue

            # 取得每個物件的資料夾
            for obj_name in os.listdir(split_path):
                obj_path = os.path.join(split_path, obj_name)
                if not os.path.isdir(obj_path):
                    continue

                # description.txt 的完整路徑
                desc_path = os.path.join(obj_path, "description.txt")
                # # 若檔案已存在,就跳過
                # if os.path.exists(desc_path):
                #     continue

                # 產生描述文字,你可以修改成你想要的描述內容
                text = f"{category}"
                map_list.add(text)
                # 寫入檔案
                with open(desc_path, "w", encoding="utf-8") as f:
                    f.write(text)
                count += 1
                print(f"建立描述檔: {desc_path}")

    print(f"總共建立了 {count} 個 description.txt 檔案。")
    print(map_list)

if __name__ == "__main__":
    # 根據你提供的路徑修改
    root_dir = "modelnet40-princeton-3d-object-dataset/rendered_views_12"
    create_description_files(root_dir)
