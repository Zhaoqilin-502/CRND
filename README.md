# CRND（Chinese Railway Number Dataset）
## Dataset Descriptions
- **1**: CRND comprises over 12,000 unique railway train number images with comprehensive annotations.
- **2**: Images were collected from Chinese railway stations and operational rail lines.
- **3**: Position information for all train number characters and corresponding train number data were manually annotated using YOLO format annotation files.
- **4**: Images representing diverse conditions were selected and manually classified into several subdatasets within CRND, including Clear(5200), Normal(3700), Missing(390), Obscured(340), and Blur(2300) subdatasets. A new CRND-Complex(259) sub-dataset has been added.
The dataset is roughly illustrated below：
<img width="1289" height="213" alt="屏幕截图 2026-06-30 163652" src="https://github.com/user-attachments/assets/3442c523-74be-44cb-97a0-ebbb21dfd2fc" />
<img width="1168" height="313" alt="屏幕截图 2026-06-30 164255" src="https://github.com/user-attachments/assets/2854ebc0-6000-4508-b4c6-4eb69a7cf23d" />
## Dataset Annotations
All comments are in txt document format.
The content of the txt document is as follows:   (2 0.772577 0.554206 0.0903717 0.218302). 
The explanation is as follows:
- **Character types**: The character type of the train number. There are a total of 23 types： 1 2 3 4 5 6 7 8 9 C E A K H N B T P Q L G Z. Encode sequentially from 0, with the last Z encoded as 22.
- **Bounding box coordinates**: The (x, y) coordinates of the center point and the height (H) and width (W) dimensions.

The dataset can be downloaded from:
- **Quark Drive**: https://pan.quark.cn/s/8486735bc5b7    
- **Google Drive**: Coming soon.
- To obtain the extraction code for the USB drive, please fill out a data application registration form. The extraction code will be automatically sent to your email address after completion. Your privacy will be protected. If you encounter any problems, please contact us via email: 18702868351@163.com. We will reply as soon as possible. You can choose Google Forms or Questionnaire Star (a survey software).
- The download address for the newly added CRND-Complex subset dataset is as follows:https://pan.baidu.com/s/1PC4sffB9XMgJrVPDQmvoZw
- **Google Forms**: https://docs.google.com/forms/d/e/1FAIpQLSdWIyFADrDkegUp37dIfA6br5JXpuRdp-sEPj9bBPq8nihvCQ/viewform?usp=dialog
- **Questionnaire**: 
# MMGA（Our Attention for Train Number Recognition）
- The project's code is currently being organized and released gradually.
- MMGA's source code has been open-sourced, and the one-click operation of the entire project is being developed.（20260630）

# Model effect display
https://github.com/user-attachments/assets/90e802b3-bac6-47ee-b262-40ecc1a6e76e

