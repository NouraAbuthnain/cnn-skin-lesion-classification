# Medical Image Classification Using Convolutional Neural Networks (CNNs)
Access the full notebook here: [Open in Google Colab](https://colab.research.google.com/drive/18lHUP1k-WC3lDxHbxN8bpMR2yM9Gl9nZ?usp=sharing)

## Project Overview
This project develops an automated system for classifying seven types of pigmented skin lesions using Convolutional Neural Networks (CNNs) and transfer learning with ResNet50. Because dermatoscopic image datasets are often small and lack diversity, automated diagnosis has historically been challenging; however, the HAM10000 dataset helped overcome this limitation by providing a large, diverse collection of dermatoscopic images suitable for deep learning research. The goal of this project is to build a reliable classification model capable of assisting clinicians in distinguishing benign from malignant lesions—particularly melanoma—through automated dermatoscopic image analysis.

### Why Automated Skin Lesion Diagnosis Matters?
- Early detection: Faster and more objective screening for melanoma when it is most treatable.
- Monitoring progression: Enables serial tracking of lesions to detect subtle changes over time.
- Decision support: Helps prioritize referrals and optimize clinical workflow.
- Reduced diagnostic error: Provides a consistent “second opinion,” mitigating human variability.

## Dataset (HAM10000)
The HAM10000 ("Human Against Machine") dataset contains 10,015 dermatoscopic images collected from:
- The Department of Dermatology, Medical University of Vienna (Austria)
- Skin Cancer Practice of Cliff Rosendahl, Queensland (Australia)

Kaggle Source: [HAM10000 Skin Lesion Dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

### Seven Diagnostic Categories
| Label | Description |
|--------|-------------|
| **akiec** | Actinic keratoses / intraepithelial carcinoma (Bowen's disease) |
| **bcc**   | Basal cell carcinoma |
| **bkl**   | Benign keratosis-like lesions |
| **df**    | Dermatofibroma |
| **nv**    | Melanocytic nevi |
| **mel**   | Melanoma |
| **vasc**  | Vascular lesions |

### Basic preprocessing steps included:
- Balancing all classes to 500 images each
- Resizing images to 224×224
- Normalization using preprocess_input
- Data augmentation (rotation, shifting, flipping, zoom)

## Model Architecture and Methods
The model is built using ResNet50 with transfer learning, where I freeze the pretrained convolutional base and train only a small custom classification head. After extracting features using Global Average Pooling, I pass them through a fully connected layer with 512 units (ReLU) and apply a 0.5 dropout rate for regularization, followed by a Softmax layer to predict the seven lesion classes. To improve generalization, I apply data augmentation, use a lesion-based split to prevent data leakage, and train the model with the Adam optimizer (learning rate 0.0001) and Categorical Cross-Entropy loss.

## Results
The final model achieved a test accuracy of 65.02%. It performed particularly well on the vasc class (F1 = 0.97) and the nv class (F1 = 0.78), while the df (F1 = 0.47) and akiec (F1 = 0.46) classes were more challenging. The confusion matrix shows that many melanoma (mel) cases were misclassified as nv, which is consistent with clinical difficulty and the visual similarity between these two lesion types. This also reflects the dataset imbalance, where nv is the dominant class.

## References
To explain and facilitate the use of the dataset, the following resources were used:
- https://arxiv.org/ftp/arxiv/papers/1803/1803.10417.pdf
- https://www.kaggle.com/code/sitadib/skin-disease-classification-using-deep-learning-ip
- https://www.kaggle.com/code/foroughgh95/skin-cancer-detection-efficientnetb3-ham10000
