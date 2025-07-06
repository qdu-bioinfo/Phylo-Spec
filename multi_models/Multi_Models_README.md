# Instructions

## Package requirement

```
python3.8
torch
pandas
numpy
scikit-learn
imbalanced-learn
ete3
matplotlib
biopython
openpyxl
```



## Installation environment

```
1.git clone https://github.com/qdu-bioinfo/Phylo-Spec.git
2.cd Phylo-Spec/multi_models
3.conda create -n PhyloSpec python=3.8
4.conda activate PhyloSpec
5.sh init.sh
```



## For a single disease classification:

1. model 1 MetaDR (Source is MetaDR  Folder): 

   ```
   cd multi_models
   python MetaDR/MetaDR.py -c input_for_all_models/Synthetic_Dataset_1/Synthetic_Dataset_1.csv -t input_for_all_models/Synthetic_Dataset_1/Synthetic_Dataset_1.nwk
   ```

2. mdoel 2 RF: 

   ```
   cd multi_models
   python RF/RF.py -c input_for_all_models/Synthetic_Dataset_1/Synthetic_Dataset_1.csv
   ```

3. model 3 PM-CNN: 

   ```
   cd multi_models
   python PMCNN/PMCNN.py -c input_for_all_models/Synthetic_Dataset_1/Synthetic_Dataset_1.csv -list input_for_all_models/Synthetic_Dataset_1/PMCNN_Synthetic_Dataset_1.csv
   ```

4. model 4 DeepPhylo: 

   ```
   cd multi_models
   python DeepPhylo/main_all.py -xnpy input_for_all_models/Synthetic_Dataset_1/Synthetic_Dataset_1_DeepPhylo_X.npy -ynpy input_for_all_models/Synthetic_Dataset_1/Synthetic_Dataset_1_DeepPhylo_y.npy -embed input_for_all_models/Synthetic_Dataset_1/Synthetic_Dataset_1_DeepPhylo_embeding.npy
   ```

5. model 5 CNN: 

   ```
   cd multi_models
   python CNN/CNN.py -c input_for_all_models/Synthetic_Dataset_1/Synthetic_Dataset_1.csv
   ```

6. Run all models with one click: 

   ```
   python Multi-comparison-models.py -c input_for_all_models/Synthetic_Dataset_1/Synthetic_Dataset_1.csv -t input_for_all_models/Synthetic_Dataset_1/Synthetic_Dataset_1.nwk -l input_for_all_models/Synthetic_Dataset_1/PMCNN_Synthetic_Dataset_1.csv -npy input_for_all_models/Synthetic_Dataset_1/Synthetic_Dataset_1_DeepPhylo_X.npy input_for_all_models/Synthetic_Dataset_1/Synthetic_Dataset_1_DeepPhylo_y.npy input_for_all_models/Synthetic_Dataset_1/Synthetic_Dataset_1_DeepPhylo_embeding.npy
   ```

   

## For multiple disease classification:

1. model 1 MetaDR (Source is MetaDR  Folder): 

   ```
   cd multi_models
   python MetaDR/MetaDR.py -c input_for_all_models/Real_Dateset_Multi_classification/Multi-classification.csv -t input_for_all_models/Real_Dateset_Multi_classification/Multi-classification.nwk
   ```

2. mdoel 2 RF: 

   ```
   cd multi_models
   python RF/RF-multi.py -c input_for_all_models/Real_Dateset_Multi_classification/Multi-classification.csv
   ```

3. model 3 PM-CNN: 

   ```
   cd multi_models
   python PMCNN/PMCNN-multi.py -c input_for_all_models/Real_Dateset_Multi_classification/Multi-classification.csv -list input_for_all_models/Real_Dateset_Multi_classification/PMCNN_Multi-classification.csv
   ```

4. model 4 DeepPhylo: 

   ```
   cd multi_models
   python DeepPhylo/main_all-multi.py -xnpy input_for_all_models/Real_Dateset_Multi_classification/Multi-classification_DeepPhylo_X.npy -ynpy input_for_all_models/Real_Dateset_Multi_classification/Multi-classification_DeepPhylo_y.npy -embed input_for_all_models/Real_Dateset_Multi_classification/Multi-classification_DeepPhylo_embeding.npy
   ```

5. model 5 CNN: 

   ```
   cd multi_models
   python CNN/CNN-multi.py -c input_for_all_models/Real_Dateset_Multi_classification/Multi-classification.csv
   ```



Note: Due to source code limitations, PM-CNN and MetaDR need to manually modify the parameter size of the first layer of full connection
