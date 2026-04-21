from ultralytics import YOLO
import os
import csv

if __name__ == "__main__":
    # 指定权重文件夹"C:\Users\zql\Desktop\chehaoshibieketi\allruns\attention_runs\yolo11"C:\Users\zql\Desktop\chehaoshibieketi\allruns\attention_runs\mobilenetv4
    weights_folder = r"C:\Users\zql\Desktop\ccpd"
    # 指定数据集文件列表
    data_files = [
        r"E:\CPDD_test\data1.yaml",
        r"E:\CPDD_test\data2.yaml",
        r"E:\CPDD_test\data3.yaml",
        r"E:\CPDD_test\data4.yaml",
        r"E:\CPDD_test\data5.yaml"

    ]

    # 查找weights目录下的所有 last.pt 文件C:\Users\zql\Desktop\chehaoshibieketi\allruns\attention_runs\CCPD111
    weight_files = []
    for root, dirs, files in os.walk(weights_folder):
        for file in files:
            if file == "last.pt":
                weight_files.append(os.path.join(root, file))

    # 用于保存结果
    results = []

    # 循环权重文件和数据集
    for weight_path in weight_files:
        weight_name = os.path.basename(os.path.dirname(weight_path))  # 上级目录名作为权重名称
        for data_path in data_files:
            data_name = os.path.splitext(os.path.basename(data_path))[0]  # 文件名作为数据集名称
            record_name = f"{weight_name}_{data_name}"  # 自定义命名
            print(f"Evaluating: {record_name}")

            model = YOLO(weight_path)
            metrics = model.val(
                val=True,
                data=data_path,
                split='test',
                batch=1,
                imgsz=640,
                device='',
                workers=0,  # <--- 关键修改，Windows下不使用多进程
                save_json=False,
                conf=0.001,
                iou=0.6,
                project='runs/valbuchongshiyan',
                seed=1,
                name='yolo_eval',
                max_det=300,
                half=False,
                dnn=False,
                plots=False,
            )

            map50_95 = metrics.box.map * 100
            map50 = metrics.box.map50 * 100
            print(f"mAP50-95: {map50_95:.2f}, mAP50: {map50:.2f}")

            results.append({
                "record_name": record_name,
                "weight": weight_path,
                "data": data_path,
                "mAP50-95": map50_95,
                "mAP50": map50
            })

    # 保存CSV"E:\crndtest\data.yaml"
    output_csv = r"C:\Users\zql\Desktop\ccpd\cpdd_eval_results.csv"
    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["record_name", "weight", "data", "mAP50-95", "mAP50"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"All results saved to {output_csv}")