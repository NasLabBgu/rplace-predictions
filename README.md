# r/place Engagement Prediction

This repository contains most of the code used to build different engagment predictions models. The engagment is based on **Community** level. We rely on the r/place social experiment that took place on April 2017. The fulll research description and modeling process is propsed in the following paper:

[This Must Be the Place: Predicting Engagement of Online
Communities in a Large-scale Distributed Campaign](https://arxiv.org/pdf/2201.05334.pdf). Abraham Israeli, Alexander Kremiansky, Oren Tsur
In Proceedings of the 2022 The Web Conference, WWW 2022. 

```bibtex
@article{israeli2022must,
  title={This Must Be the Place: Predicting Engagement of Online Communities in a Large-scale Distributed Campaign},
  author={Israeli, Abraham and Kremiansky, Alexander and Tsur, Oren},
  journal={arXiv preprint arXiv:2201.05334},
  year={2022}
}
```
## Code 
Most of the code that was used to load data and train the models is under two folders:

*  **data_loaders:**
A folder that contains all the code we used to load data from various sources.

* **r_place_drawing_classifier:**
A folder that contains mode of the code we userd to train the classification models.

## Data
The raw level data is taken from the open Pushift repository (https://files.pushshift.io/reddit/).
The annotated data (community level) can be found as as a csv file under the "annotated_data" name.

