# CRND（Chinese Railway Number Dataset）
## Dataset Descriptions
- **1** CRND comprises over 12,000 unique railway train number images with comprehensive annotations.
- **2** Images were collected from Chinese railway stations and operational rail lines
- **3** Position information for all train number characters and corresponding train number data were manually annotated using YOLO format annotation files.
- **4** Images representing diverse conditions were selected and manually classified into several subdatasets within CRND, including Clear(5200), Normal(3700), Missing(390), Obscured(340), and Blur(2300) subdatasets

## Dataset Annotations
All comments are in txt document format 
The content of the txt document is as follows:(2 0.772577 0.554206 0.0903717 0.218302).The explanation is as follows
- **Character types**: The character type of the train number. There are a total of 23 types： 1 2 3 4 5 6 7 8 9 C E A K H N B T P Q L G Z. Encode sequentially from 0, with the last Z encoded as 22.
- **Bounding box coordinates**: The (x, y) coordinates of the center point and the height (H) and width (W) dimensions.

The dataset can be downloaded from: 
