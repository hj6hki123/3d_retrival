import time
import torch
import cv2 as cv

#======================================================================
from preprocess import img_convert

model = torch.jit.load('model_scripted_32.pt')

#======================================================================

cate = {
    'A': 0, 'BA': 1, 'CHI': 2, 'DA': 3, 'E': 4, 'FU': 5, 'HA': 6, 'HE': 7, 'HI': 8, 'HO': 9, 'I': 10, 'JI': 11,
    'KA': 12, 'KE': 13, 'KI': 14, 'KO': 15, 'KU': 16, 'MA': 17, 'ME': 18, 'MI': 19, 'MO': 20, 'MU': 21, 'N': 22,
    'NA': 23, 'NE': 24, 'NI': 25, 'NO': 26, 'NU': 27, 'O': 28, 'PI': 29, 'RA': 30, 'RE': 31, 'RI': 32, 'RO': 33,
    'RU': 34, 'SA': 35, 'SE': 36, 'SHI': 37, 'SO': 38, 'SU': 39, 'TA': 40, 'TE': 41, 'TO': 42, 'TSU': 43, 'U':
    44, 'WA': 45, 'WO': 46, 'YA': 47, 'YO': 48, 'YU': 49
}
inv_cate = {v: k for k, v in cate.items()}    #label_dict_inv 是將數值標籤轉換回字母標籤的字典


cap = cv.VideoCapture(0)

# 設定框架的初始位置和大小
x, y, w, h = 170, 100, 300, 300

while(cap.isOpened()):
    ret,frame = cap.read()
    
    # 顯示影像
    cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # 在視窗中繪製框架
    cv.imshow("vedio",frame)
    
    # 按下 's' 鍵進行截圖
    if cv.waitKey(1) == ord('s'):
        
        # 儲存影像
        filepath = './screenshotq.jpg'
        cv.imwrite(filepath, frame)
        
        # 擷取框架內的圖片
        frame_screenshot = frame[y:y+h, x:x+w]  

        # 存儲圖片
        filepath2 = './frame_screenshot.jpg'
        cv.imwrite(filepath2 , frame_screenshot)  

        # 讀取圖片檔案
        frame_img = cv.imread(filepath2)
        
        # 顯示圖片
        cv.imshow(filepath2, frame_img)
        cv.waitKey(0)
        
        #量測起始時間
        start_time = time.time()  # 開始計時
        
        #呼叫資料前處裡函式q
        trans_frame_image = img_convert(frame_img)
               
        # 添加批次維度 (batch dimension)：模型預測需要批次的資料
        input_image = trans_frame_image.unsqueeze(0)

        # 進行預測
        with torch.no_grad():
            output = model(input_image)
            probabilities = torch.softmax(output, dim=1)
            
            max_prob, predicted = torch.max(probabilities, 1)

        end_time = time.time()  # 結束計時
        elapsed_time = end_time - start_time  # 計算花費時間

        # 取得預測結果
        prediction = predicted.item()
        print("預測結果:", prediction, inv_cate[prediction])
        prediction = predicted.item()
        prob = max_prob.item()
        print(f"預測結果: {prediction} {inv_cate[prediction]} 機率: {prob:.4f}")

        #結束時間,印出計算時間   
        print(f'預測時間: {elapsed_time:.3f} 秒')
        
    # 按下q離開
    if(cv.waitKey(1) == ord('q')):
        break
#關閉視窗
cap.release()
cv.destroyAllWindows()