# Video Segmentation of Clouds ☁️
---
![unknown](https://user-images.githubusercontent.com/25123131/148730294-becc9d82-6500-462f-b45d-6fadd18e3f87.png)

Team Members: Sean Perry, Jenny Lam Jason Liang, Aran Punniamoorthy

This repo includes an implementation of a semantic segmentation model which predicts the semantic class label for each pixel in an image across all images in a cloud image dataset.

An end-to-end model training + inference notebook can be found in the `src` folder.

## Data

Our data comes from WSISEG (click [here](https://github.com/CV-Application/WSISEG-Database)) and HYTA Databases (click [here](https://github.com/Soumyabrata/HYTA)) to download our dataset.

You can then structure the data by putting it into a folder structure as such:

|__ src <br />
&nbsp;&nbsp;&nbsp;&nbsp;|__ data <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__ train <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__ val <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__ test <br />

## Credits

We would like to thank [Jonathan Zamora](https://github.com/jonzamora) and [Joseph Liu](https://github.com/kelpabc123) from [ACM AI at UCSD](https://ai.acmucsd.com/) for advising our team on this project as part of the ACM Projects Cohort for Fall 2021.
