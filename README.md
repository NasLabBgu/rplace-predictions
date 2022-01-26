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

## r/place Resources
* **Wikipedia Page:** A wiki page describing the experiment: https://en.wikipedia.org/wiki/Place_(Reddit)
* **Experiment Video:** A fast-forward video of the 72-hours timelapse of the experiment:  https://www.youtube.com/watch?v=XnRCZK3KjUY
* **Subreddit:** The r/place subreddit in Reddit: https://www.reddit.com/r/place/

## Code 
Most of the code that was used to load data and train the models is under two folders:

*  **data_loaders:**
A folder that contains all the code we used to load data from various sources.

* **r_place_drawing_classifier:**
A folder that contains mode of the code we userd to train the classification models.

## Data
The raw level data is taken from the open Pushift repository (https://files.pushshift.io/reddit/).
The annotated data (community level) can be found as as a csv file under the "annotated_data" name.

## r/place Over Time
<table>
  <tr>
     <td>**2 Hours** after r/place was launched.</td>
     <td>7 Hours after r/place was launched.</td>
     <td>25 Hours after r/place was launched.</td>
     <td>72 Hours after r/place was launched.</td>
  </tr>
  <tr>
    <td><img src="pics/1490986860.png" width=240 height=240></td>
    <td><img src="pics/1490986860.png" width=240 height=240></td>
    <td><img src="pics/1491116860.png" width=240 height=240></td>
    <td><img src="pics/1491226860.png" width=240 height=240></td>
  </tr>
 </table>
