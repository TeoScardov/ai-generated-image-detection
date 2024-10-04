# Detection and Classification of Artificially Generated Images using Deep Residual Networks and Multimodal Language Models
This research explores two different technologies that can be applied to fake image detection, both of which were trained and tested on large scale datasets. The CNN part focuses on ResNet, while the LLM part leverages TinyLLaVA Factory to finetune the multimodal model. 

## Dependencies
```
pip install numpy scikit-learn seaborn pillow

pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 --extra-index-url https://download.pytorch.org/whl/cu102

git clone https://github.com/TinyLLaVA/TinyLLaVA_Factory.git

git clone https://github.com/jacobgil/pytorch-grad-cam.git
```

## Data
### [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
This dataset contains 120k images of 32x32 pixels resolution, and it is used to quickly experiment on different configurations of the ResNet architecture.

![cifake_samples](https://github.com/user-attachments/assets/26932fb4-fa68-47c4-924e-12a7c5282c31)
### [GenImage](https://drive.google.com/drive/folders/1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS)
This dataset contains around 2.5M images of up to 1024x1024 pixels resolution, and it is used to extensively test ResNet and the TinyLLaVA model. Before training and testing, the images are compressed into the JPEG format and resized to 256x256.

![genimage_examples](https://github.com/user-attachments/assets/c56af88a-9e35-4f8c-b1b2-45162ffb2886)

## ResNet
ResNet was modified by replacing the last layer with a double output layer predicting if an image was real or fake, while simultaneously predicting the object class.  
ResNet18 was modified this way and finetuned on the CIFAKE dataset, obtaining a test error of 3.06% for the real vs fake prediction and of 11.27% for the object prediction.

![resnet_cifake](https://github.com/user-attachments/assets/4baf32a6-be2c-4574-a8fd-6886c7e1ddbf)

Regarding GenImage, due to the increase in the number of images, in the image resolution, and because the number of classes was 1000 for object prediction and 9 for real vs fake (real, and 8 different model that generated the images) it was necessary to scale up the model and use ResNet50.  
The final model achieved an error of 5.00% in predicting the exact generating model, which translates to a 0.56% error in the real vs fake prediction. The error for the object classification was close to the performance of the baseline ResNet50.

![resnet_genimage](https://github.com/user-attachments/assets/408e31c3-d4a1-4f51-beb1-64c9349a880a)

A model inspection with Grad-CAM revealed that the features considered by ResNet during classification are not useful for human interpretability, as it often looks at subtle details in the background, instead of the main subject of the image.

![gradcam_examples_revised](https://github.com/user-attachments/assets/9d74492d-b337-41bf-9261-2f0d50da3694)

## TinyLLaVA
There are no specific tests for MLM evaluation on artificially generated images, so a [custom VQA benchmark](https://github.com/TeoScardov/ai-generated-image-detection/tree/main/llm/data/eval/custom-vqa) was developed to guide the finetuning process, using 50 true-false questions, and to test the final model performance, with 100 multiple-choice questions.

![mlm_binary_accuracy](https://github.com/user-attachments/assets/735faa9f-a88e-43d9-bb5e-e9574d927ce7)

The TinyLLaVA-Phi-2-SigLIP-3.1B model was finetuned for 6 epochs on [100 conversation samples](https://github.com/TeoScardov/ai-generated-image-detection/tree/main/llm/data/finetune) that were manually crafted. The final model lowered the error on the multiple choice questionnaire from 47% to 38%, managing to identify artificial images more reliably.

## Conclusion
The CNN ResNet has a very low error, but it doesn't help humans understand why an image is fake.  
The MLM TinyLLaVA has an higher error rate, but it is able to explain in natural language the motivation behind its classification.  
The choice of which technology is better ultimately depends on its intended application, balancing the trade-off between accuracy and interpretability.

![mlm_chat4x4](https://github.com/user-attachments/assets/0b813c7d-f643-407f-a66b-969cac22804b)

## License
This project is licensed under the [Apache 2.0 License](LICENSE).
