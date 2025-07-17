# Instructions

## Package requirement

```
python3.8
torch >= 2.3.1
pandas >= 2.2.2
numpy >= 1.26.4
scikit-learn >= 1.4.2
imbalanced-learn >= 0.12.3
ete3 >= 3.1.3
matplotlib >= 3.7.2
biopython >= 1.83
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

   Manual modification of the fully connected layer dimensions: line (#99)

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

   Manual modification of the fully connected layer dimensions: line (#71)

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

   Manual modification of the fully connected layer dimensions: line (#65 and #607)

   ```
   python Multi-comparison-models.py -c input_for_all_models/Synthetic_Dataset_1/Synthetic_Dataset_1.csv -t input_for_all_models/Synthetic_Dataset_1/Synthetic_Dataset_1.nwk -l input_for_all_models/Synthetic_Dataset_1/PMCNN_Synthetic_Dataset_1.csv -npy input_for_all_models/Synthetic_Dataset_1/Synthetic_Dataset_1_DeepPhylo_X.npy input_for_all_models/Synthetic_Dataset_1/Synthetic_Dataset_1_DeepPhylo_y.npy input_for_all_models/Synthetic_Dataset_1/Synthetic_Dataset_1_DeepPhylo_embeding.npy
   ```

   

## For multiple disease classification:

1. model 1 MetaDR (Source is MetaDR  Folder): 

   Manual modification of the fully connected layer dimensions: line (#99)

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

   Manual modification of the fully connected layer dimensions: line (#53)

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
