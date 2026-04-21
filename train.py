from PIL.ImageOps import scale

from ultralytics import YOLO
from ultralytics import RTDETR



if __name__ == '__main__':
    # 加载模型
    model = YOLO(r'cfg/models/11/ours4_22yolo11.yaml')  # 不使用预训练权重训练
    #model= YOLO(r'C:\Users\zql\Desktop\ultralytics-main-yolov11\yolo11n.pt')
    # 训练参数 ----------------------------------------------------------------------------------------------
    model.train(

        data=r"F:\OSDar23\reorganized_dataset\data.yaml",
        epochs=10,  # (int) 训练的周期数
        patience=200,  # (int) 等待无明显改善以进行早期停止的周tiao期数
        batch=16,  # (int) 每批次的图像数量（-1 为自动批处理）
        imgsz=640,  # (int) 输入图像的大小，整数或w，h
        save=True,  # (bool) 保存训练检查点和预测结果
        save_period=-1,  # (int) 每x周期保存检查点（如果小于1则禁用）
        cache=False,  # (bool) True/ram、磁盘或False。使用缓存加载数据
        device='',  # (int | str | list, optional) 运行的设备，例如 cuda device=0 或 device=0,1,2,3 或 device=cpu
        workers=8,  # (int) 数据加载的工作线程数（每个DDP进程）
        project='runs/train4',  # (str, optional) 项目名称
        name='osd',  # (str, optional) 实验名称，结果保存在'project/name'目录下
        exist_ok=False,  # (bool) 是否覆盖现有实验
        pretrained=True,  # (bool | str) 是否使用预训练模型（bool），或从中加载权重的模型（str）
        optimizer='SGD',  # (str) 要使用的优化器，选择=[SGD，Adam，Adamax，AdamW，NAdam，RAdam，RMSProp，auto]
        verbose=True,  # (bool) 是否打印详细输出
        seed=0,  # (int) 用于可重复性的随机种子
        deterministic=True,  # (bool) 是否启用确定性模式
        single_cls=False,  # (bool) 将多类数据训练为单类
        rect=False,  # (bool) 如果mode='train'，则进行矩形训练，如果mode='val'，则进行矩形验证
        cos_lr=False,  # (bool) 使用余弦学习率调度器
        close_mosaic=0,  # (int) 在最后几个周期禁用马赛克增强
        resume=False,  # (bool) 从上一个检查点恢复训练
        amp=True,  # (bool) 自动混合精度（AMP）训练，选择=[True, False]，True运行AMP检查
        fraction=1.0,  # (float) 要训练的数据集分数（默认为1.0，训练集中的所有图像）
        profile=False,  # (bool) 在训练期间为记录器启用ONNX和TensorRT速度
        freeze=None,  # (int | list, 可选) 在训练期间冻结前 n 层，或冻结层索引列表。

    )



