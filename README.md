# r/place Prediction

This repository is home to our research efforts dealing with the r/place experiment. It currently contains code and data for two different papers we published:

a. The 'r_place_success_prediction' directory contains data and code for the **success level prediction model**, which was presented at ICWSM 2024 (see citations below).
b. The  'r_place_drawing_classifier' directory contains the data and code used for the **participation prediction model**, which was presented at WWW 2022 (see citation below). 

Both projects focus on the **Community** level prediction. We rely on the r/place social experiment that took place in April 2017. The two full research descriptions and modeling processes are proposed in the following paper:

[With Flying Colors: Predicting Community Success in Large-scale Collaborative Campaigns](https://ojs.aaai.org/index.php/ICWSM/article/download/31344/33504). Abraham Israeli and Oren Tsur
In Proceedings of the International AAAI Conference on Web and Social Media. Vol. 18. 2024.

[This Must Be the Place: Predicting Engagement of Online
Communities in a Large-scale Distributed Campaign](https://arxiv.org/pdf/2201.05334.pdf). Abraham Israeli, Alexander Kremiansky, Oren Tsur
In Proceedings of the 2022 The Web Conference, WWW 2022. 

```bibtex
@inproceedings{israeli2024flying,
  title={With Flying Colors: Predicting Community Success in Large-scale Collaborative Campaigns},
  author={Israeli, Abraham and Tsur, Oren},
  booktitle={Proceedings of the International AAAI Conference on Web and Social Media},
  volume={18},
  pages={691--703},
  year={2024}
}
```

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
     <td><b>2 Hours</b> after r/place was launched.</td>
     <td><b>7 Hours</b> after r/place was launched.</td>
     <td><b>25 Hours</b> after r/place was launched.</td>
     <td><b>72 Hours</b> after r/place was launched.</td>
  </tr>
  <tr>
    <td><img src="pics/1490986860.png" width=200 height=200></td>
    <td><img src="pics/1491066860.png" width=200 height=200></td>
    <td><img src="pics/1491116860.png" width=200 height=200></td>
    <td><img src="pics/1491226860.png" width=200 height=200></td>
  </tr>
 </table>
