---
layout: defaultPaper
title: SparsePoser Real-time Full-body Motion Reconstruction from Sparse Data
permalink: /
---

<center>
<figure style="display:inline-block;margin:0;padding:0"><img src="assets/img/teaser.jpg" alt="teaser" width="100%" /><figcaption style="text-align:center"></figcaption></figure>
</center>

<br>

<div class="img_horizontal_container">
	<a href="https://doi.org/10.1145/3625264">
	<div class="img-with-text">
		<img src="assets/img/article_icon.svg" alt="paper" />
		<p><b>Paper</b></p>
	</div>
	</a>
	<a href="https://github.com/UPC-ViRVIG/SparsePoser">
	<div class="img-with-text">
		<img src="assets/img/github_icon.svg" alt="code" />
		<p><b>Code</b></p>
	</div>
	</a>
	<a href="https://github.com/UPC-ViRVIG/SparsePoser#data">
	<div class="img-with-text">
		<img src="assets/img/database_icon.svg" alt="data" />
		<p><b>Data</b></p>
	</div>
	</a>
	<a href="https://youtu.be/BAi4KoHtehY">
	<div class="img-with-text">
		<img src="assets/img/video_icon.svg" alt="video" />
		<p><b>Video</b></p>
	</div>
	</a>
</div>

------

<h3><center><b>
Abstract
</b></center></h3>

<div style="text-align: justify;">
We introduce SparsePoser, a novel deep learning-based solution for reconstructing a full-body pose from a reduced set of six tracking devices. Our system incorporates a convolutional-based autoencoder that synthesizes high-quality continuous human poses by learning the human motion manifold from motion capture data. Then, we employ a learned IK component, made of multiple lightweight feed-forward neural networks, to adjust the hands and feet towards the corresponding trackers. We extensively evaluate our method on publicly available motion capture datasets and with real-time live demos. We show that our method outperforms state-of-the-art techniques using IMU sensors or 6-DoF tracking devices, and can be used for users with different body dimensions and proportions.</div>

------

<div class="row">
	<figure style="display:inline-block;margin:0;padding:0">
		<video width="100%" autoplay muted loop>
		  <source src="assets/img/intro.mp4" type="video/mp4">
		Your browser does not support the video tag.
		</video>
	</figure>
</div>

--------

<br>

<h3><b>
Method
</b></h3>
We propose a deep learning-based framework for animating human avatars from a sparse set of input sensors
Our system can be divided into two parts: 

- **Generator** is a convolutional-based autoencoder that extracts the main features from the sensors and reconstructs the user poses for a set of contiguous frames.
- **Learned IK** is a set of feedforward neural networks that adjust the positions of the end-effectors to attain the target points.

<center>
<figure style="display:inline-block;margin:0;padding:0"><img src="assets/img/pipeline.png" alt="pipeline" width="100%" /><figcaption style="text-align:center"></figcaption></figure>
</center>

<br>

<div style="background-color:rgba(247, 255, 255, 1.0); vertical-align: middle; padding:10px 20px; text-align: justify;">
<h3><b>
Generator
</b></h3>
The generator takes three inputs: \( \mathbf{S} \) (static component, contains the skeleton offsets), \( \mathbf{Q} \) 
(dynamic component, root space local rotations and translations of all joints), and \( \mathbf{D} \) 
(displacement component, root displacement between two frames), 
and produces a full-body pose. It consists of a Static Encoder \( se \), Dynamic Encoder \( de \), and Decoder \( d \). 
The Static Encoder extracts learned features from \( \mathbf{S} \), while the Dynamic Encoder encodes the primal skeleton 
using \( \mathbf{D} \) and a subset of \( \mathbf{Q} \). The Decoder reconstructs the pose using the learned features and the 
primal skeleton. To ensure stable training, dual quaternions and Mean Squared Error (MSE) reconstruction loss are used 
instead of Forward Kinematics-based (FK) losses.

<center>
	<figure style="display:inline-block;margin:0;padding:0">
		<video width="100%" autoplay muted loop>
		  <source src="assets/img/generator.mp4" type="video/mp4">
		Your browser does not support the video tag.
		</video>
	</figure>
</center>

</div>

<br>

<div style="background-color:rgba(247, 255, 247, 1.0); vertical-align: middle; padding:10px 20px; text-align: justify;">
<h3><b>
Learned IK
</b></h3>

The learned IK stage improves end-effector accuracy in the synthesized human poses by the generator. 
Feedforward neural networks are trained for each limb to make slight adjustments 
based on the dynamic and static components, improving pose precision. Two losses, 
\( \mathcal{L}_{S} \) and \( \mathcal{L}_{Reg} \), ensure accurate end-effectors and maintain 
pose quality. The final loss is a weighted combination, controlling the tradeoff 
between accuracy and pose quality.

<center>
	<figure style="display:inline-block;margin:0;padding:0">
		<video width="100%" autoplay muted loop>
		  <source src="assets/img/learnedik.mp4" type="video/mp4">
		Your browser does not support the video tag.
		</video>
	</figure>
</center>

</div>

<br>

-----

<h3><center><b>
Overview Video
</b></center></h3>

<center>
<div class="video_wrapper">
	<iframe frameborder="0" width="100%" height="100%"
	src="https://www.youtube.com/embed/BAi4KoHtehY">
	</iframe>
</div>
</center>

-----

<br>

<h3><b>
Citation
</b></h3>
<div style="background-color:rgba(0, 0, 0, 0.03); vertical-align: middle; padding:10px 20px;">
@article {ponton2023sparseposer, <br>
	journal = {ACM Trans. Graph.}, <br>
	{% raw %}
	title = {{SparsePoser: Real-time Full-body Motion Reconstruction from Sparse Data}}, <br>
	{% endraw %}
	author = {Ponton, Jose Luis and Yun, Haoran and Aristidou, Andreas and Andujar, Carlos and Pelechano, Nuria}, <br>
	year = {2023}, <br>
	volume = {}, <br>
	number = {}, <br>
	pages = {}, <br>
	ISSN = {0730-0301}, <br>
	DOI = {10.1145/3625264} <br>
}
</div>