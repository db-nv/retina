# RetiDetect: AI for Early Detection of Diabetic Retinopathy

RetiDetect is a project aimed at building a reliable, AI-driven tool for the early detection and management of diabetic retinopathy. By implementing Convolutional Neural Network (CNN) models, RetiDetect is designed to streamline the process of diabetic retinopathy detection, classifying retina scans, verifying their authenticity, and estimating the individual's age. 

## Current Modules

### Classification of Retina Scans

- The task is to classify retina scans on a scale of 1-5, indicating the severity of diabetic retinopathy.
- Various EfficientNet architectures have been used for this classification task.
- Both Mean Squared Error (MSE) and Cross-Entropy have been tested as loss functions.
- The model utilizing Cross-Entropy has currently yielded the best results, with an accuracy of 86%.

### Prediction of the Person's Age

- The objective here is to predict the age of an individual based on their retina scan.
- We plan to use the model developed for the classification of retina scans as a pre-trained model for this task.
- We acknowledge a potential bias in the model due to the skew in age distribution in our training data.

### Determination of Whether an Image is a Retina Scan or Not

- This module is focused on developing a model that not only verifies if an image is a retina scan but also assesses its quality.
- An ongoing challenge is ensuring that the model can effectively analyze all types of retina scans.
- The question of how well our model can determine the quality of different scans and their compatibility with our predictive models is under exploration.

## How to Contribute

We welcome contributions to the RetiDetect project! You can contribute in several ways:

1. **Submitting pull requests**: If you've fixed a bug or have a feature you've added, just create a new pull request. If you're new to Github, [here is a good guide](https://opensource.guide/how-to-contribute/#opening-a-pull-request) on how to create a pull request.

2. **Reporting issues or bugs**: If you notice any bugs or issues, please report them.

3. **Suggesting enhancements**: If you have an idea to make this project better, we'd love to hear it! Please just open an issue as a place for discussing the feature.

4. **Data contribution**: If you have access to relevant datasets that could improve the performance of RetiDetect and can share them, please contact us.

Please read the [CONTRIBUTING.md](https://github.com/YourRepo/CONTRIBUTING.md) for more details on how to contribute.

## License

This project is licensed under the Creative Commons Attribution-NonCommercial (CC BY-NC) license. This license allows others to remix, adapt, and build upon this work non-commercially, and although their new works must also acknowledge the original creator and be non-commercial, they don’t have to license their derivative works on the same terms. Commercial use requires permission from the original creator. For more details, see the LICENSE.md file.