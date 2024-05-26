---
title: "Physics-Informed Deep Learning for Transformer Based Radiotherapy Dose Prediction"
author: "Gijs de Jong, Jakob Kaiser, Macha Meijer, Thijmen Nijdam and Derck Prinzhorn"
date: '27-05-2024'
---

# Physics-Informed Deep Learning for Transformer Based Radiotherapy Dose Prediction
**Gijs de Jong, Jakob Kaiser, Macha Meijer, Thijmen Nijdam and Derck Prinzhorn**
**Teaching Assistant: Stefanos Achlatis**


---
In this blogpost, the paper ["TrDosePred: A deep learning dose prediction algorithm based on transformers for head and neck cancer radiotherapy"](#4) is discussed, reproduced and extended upon. This paper shows the application of 3D Vision Transformers on the field of radiation dose treatment planning. The goal of this blogpost is to explain the research done in the field of radiation dose prediction, reproduce the work done in the original paper, and extend the TrDosePred architecture by incorporating physics-based elements in the training process.

---
# Introduction
<!-- Start with what radiation therapy is, how it is commonly used -->
Radiation therapy is a crucial method in cancer treatment, where about 50% of all patients receive this kind of treatment [[1]](#1). It is used to either shrink a tumor, before using surgery to remove the tumor, or to target cancer cells that are possibly left after the surgery.

Before radiation therapy can be delivered to treat cancer, a treatment plan must be formulated. This plan represents the most optimal configuration of the treatment device, aiming to maximize radiation to the planned target volume (PTV) while minimizing exposure to the organs at risk (OARs). The PTV determines different regions that should receive a minimum percentage of radiation, where regions closer to the tumor should receive more radiation. An example of different PTVs is illustrated in Figure 1. OARs are organs that are sensitive to radiation and should thus receive minimal radiation.

<!-- ![Schematic-diagram-of-radiotherapy-irradiation-volumes](https://hackmd.io/_uploads/S1thHMHX0.jpg)
Figure 1: Example of a planned target volume (PTV) [[8]](#8) -->

<div style="text-align: center;">
    <img src="figs/Schematic-diagram-of-radiotherapy-irradiation-volumes.png" alt="PTV"/>
    <p>Figure 1: Example of a planned target volume (PTV) <a href="#8">[8]</a></p>
</div>


<!-- Generating treatment plans is hard, explanation of treatment plans -->
<!-- Treatment planning often requires manual adjustments by medical physicists or dosimetrists, introducing variability in plan quality that can affect treatment outcomes, and is very time consuming. Efforts to reduce this manual intervention is called automated treatment planning and can be divided into two categories, objective-based planning (OBP) and knowledge-based planning (KBP). OBP relies on optimization algorithms that adjust preset objectives to achieve the established clinical goals. KBP  uses a library of plans from previous patients to predict dose–volume objectives for the new patient. KBP methods are generally formulated as two-stage pipelines (see Figure 2). In the knowledge-based planning approach to treatment planning (see Figure 2), the first step is to create an initial dose distribution, which is typically predicted first using contoured CT images of the patient. The dose distribution contains the amount of radiation that should be given to every individual voxel. After this,, an optimization model develops a treatment plan based on this predicted dose distribution. The accuracy of the initial dose distribution can significantly reduce the time required in the overall optimization process [[7]](#7). This research focuses on predicting the initial 3D dose distribution, the first step of the KBP pipeline. -->
Treatment planning often requires manual adjustments by medical physicists or dosimetrists, introducing variability in plan quality that can affect treatment outcomes, and is very time consuming. To reduce the amount of manual intervention, knowledge-based planning (KBP) can be used. KBP methods are generally formulated as two-stage pipelines (see Figure 2). In the knowledge-based planning approach to treatment planning (see Figure 2), the first step is to create an initial dose distribution, which is typically predicted first using contoured CT images of the patient. The dose distribution contains the amount of radiation that should be given to every individual voxel. After this, an optimization model develops a treatment plan based on this predicted dose distribution. The accuracy of the initial dose distribution can significantly reduce the time required in the overall optimization process [[7]](#7). This research focuses on predicting the initial 3D dose distribution, the first step of the KBP pipeline.

<!-- ![OpenKBP_fig1](https://hackmd.io/_uploads/B1FrEGSmR.jpg)
Figure 2: Overview of a complete knowledge-based planning pipeline [[7]](#7) -->
<div style="text-align: center;">
    <img src="figs/OpenKBP_fig1.jpg" alt="Data"/>
    <p>Figure 2: Overview of a complete knowledge-based planning pipeline <a href="#7">[7]</a></p>
</div>

<!-- Currently treament plan generation with monte carlo, too slow, use DL-->
A popular approach to obtain the initial 3D dose prediction is the Monte Carlo dose calculation [[9]](#9), which uses a combination of simulated and measured beams to compute a dose. While accurate, this approach is slow and it can take minutes to hours for a single dose calculation [[10]](#10). Previous work has improved the speed of these calculations [[10]](#10) [[11]](#11), but it remains a labour-intensive task to create the final treatment plans. Therefore, recent studies have explored deep learning techniques such as Vision Transformers (ViTs) to predict 3D dose distributions that can be used to evaluate or generate treatment plans [[4]](#4) [[12]](#12).

<!-- Related work, what is already done -->
Previous research on dose distribution prediction has investigated a variety of architectural designs. The field was initially advanced by the introduction of the U-Net architecture [[13]](#13), devised for biomedical image segmentation. Its integration of both local and contextual information through convolutional layers and a U-shaped encode-decode architecture makes it suitable for dose prediction applications. The introduction of U-Net gave room to a wide variety of model developments [[14]](#14) [[15]](#15) [[16]](#16). Significant advancements include the transition from predicting 2D dose slices to full 3D dose predictions. Another type of model used for dose prediction is a generative adversarial network (GAN) based architecture [[7]](#7), which achieved state-of-the-art performance and generated more accurate dose distribution predictions.
<!-- Introduction of TrDosePred -->
More recently, with advancements in ViTs [[17]](#17), TrDosePred, a 3D transformer-based algorithm was proposed for dose predictions [[4]](#4). ViTs are capable of modeling long-range dependencies in image data through self-attention mechanisms, enabling them to capture more global context compared to the local receptive fields of convolutional neural networks (CNNs) [[17]](#17).

<!-- Table of contents, what will we talk about -->
In this research, a physics-based approach on dose prediction is presented. This work is based on TrDosePred [[4]](#4) and aims to extend the current research by extending the original framework with physics-based elements, allowing the network to produce dose predictions that are in line with the physics of radiation.

In this blogpost, the key components of TrDosePred will be [explained](#TrDosePred) and [analysed](#Analysis-of-TrDosePred). Thereafter, we introduce the [physics-based components](#TrDosePred---A-physics-based-approach) that were used to extend the original TrDosePred. We analyze the [results](#Results) of the original TrDosePred architecture and compare it with the physics network. Lastly, these results are [discussed](#Conclusion) and possibilities for future research are described.

# TrDosePred
<!-- What is the main idea of TrDosePred -->
TrDosePred leverages a transformer-based approach to predict 3D dose distributions for radiotherapy. Key factors in its success include the ability to model long-range dependencies in the input data by employing 3D Swin Transformer blocks, along with convolutional patch embedding and expanding blocks. The patch embedding block transforms the input into patches that the 3D Swin Transformer blocks process. The patch expanding block reconstructs the final patches into a high-resolution 3D dose distribution. Another crucial factor contributing to TrDosePred's effectiveness is data augmentation, where the limited dataset of 200 training samples is augmented extensively (see the [data](#Data) section for a more extensive description). Additionally, five-fold cross-validation is performed to create an ensemble model, which averages the outputs of five individual models for greater robustness [[4]](#4).

Another component of success is that during inference, robustness of the the model is further enhanced through test-time augmentation, which has proved effective in previous research [[20]](#20). This is done by flipping the input over the different axes to create 4 augmented inputs. These augmented inputs are then passed to each of the five models in the ensemble. After inference, outputs from each model are reverted to their original orientations, resulting in a total of 20 different outputs that are then averaged [[4]](#4). The following sections will provide a detailed explanation of the data, model architecture and limitations that this approach has.

<!-- Explanation of dataset + dataset preprocessing -->
## Data
The data used for training and evaluation is publicly available from the OpenKBP Challenge, designed to standardize benchmarks for dose distribution prediction models. This dataset, known as OpenKBP [[7]](#7), includes 340 head and neck cancer (HNC) patients treated with 6MV IMRT using nine coplanar beams. Each patient has at least a CT scan, one PTV, up to seven OARs, and a dose distribution generated by a Computational Environment for Radiotherapy Research (CERR), which is used as the ground truth dose distribution for the patient [[4]](#4). The CT scan contains information on the on the density of the tissue at each location. The PTVs and OARs are binary masks indicating what regions correspond to the corresponding PTV or OAR. Each of these features has a shape of 128 × 128 × 128 (D x W x H) and an approximate resolution of 3.5 mm × 3.5 mm × 2 mm.

The three-channel input volume is a concatenation of the planning CT, OARs, and PTVs resulting in a final shape of 3 x 128 x 128 x 128 (C x D x W x H). Preprocessing is done as in [[4]](#4), the specifics of each channel are as follows:

- **Planning CT Channel:** CT values are cropped to range from -1024 to 1500 and then divided by 1000.
- **PTV Channel:** Each voxel inside PTVs is assigned the corresponding prescription dose and then for each voxel the maximum value over all PTVs is taken. The resulting channel is normalized by 70 Gy.
- **OAR Channel:** Seven critical OAR masks are labeled with distinct integers and merged into a single channel by summing them: 1 for the brain stem, 2 for the spinal cord, 3 for the right parotid, 4 for the left parotid, 5 for the esophagus, 6 for the larynx, and 7 for the mandible.

To enhance the robustness of TrDosePred, data augmentations were applied during training. These included random flipping along the inferior-superior and right-left axes, as well as random translation (up to 5 voxels along each axis). Additionally, random rotations around the inferior-superior axis were performed, with rotation degrees chosen randomly from a list of 0°, 40°, 80°, 120°, 160°, 200°, 240°, 280°, and 320°.

The dataset was divided as is done in the OpenKBP challenge [[7]](#7): patients 1-200 for training, patients 201-240 for validation, and patients 241-340 for testing.

<!-- Explanation of architecture -->
## Model architecture

Figure 3 shows the overall SWIN-based architecture of the proposed TrDosePred. With a three-channel feature of contoured CT as input, a patch embedding block first projects it into a sequence of patch tokens. A transformer-based encoder and decoder then build the relationship between embedded input features and dose maps. Finally, a patch expanding block generates the 3D dose distribution. The individual components are further elaborated on in the [appendix](##Swin-Components).

<div style="text-align: center;">
    <img src="figs/architecture.png" alt="architecture"/>
    <p>Figure 3: Overview of architecture of TrDosePred <a href="#4">[4]</a></p>
</div>

## Metrics
Dose-volume histograms (DVHs) are commonly used to evaluate treatment plans [[6]](#6). DVHs are used to quantify the dose distribution around a target. They display the absorbed dose of an organ, over the relative volume of the organ that reached this dose. An example is shown in Figure 6.

<div style="text-align: center;">
    <img src="figs/DVH.jpg" alt="DVH example"/>
    <p>Figure 6: An example of a dose-volume histogram. Here, the x-axis displays the absorbed dose, while the y-axis explains the volume of the organ that absorbed that dose. Every line represents a different structure.</p>
</div>

The DVH score is computed as the MAE of a set of specific criterea. These criterea are the $D_{mean}$ and $D_{0.1cc}$ for the seven OARs and the dose received by 1%, 95% and 99% ($D_1$, $D_{95}$, $D_{99}$) of the voxels within the target volumes for the three PTVs, where $D_{0.1cc}$ is the maximum dose received by the most exposed 0.1 cubic centimeters (cc) of a specified volume.

Another metric that is used for evaluation is the dose score, which is computed by taking the MAE between predicted dose and target dose.

Both metrics are expressed in the Gray (Gy) unit, which is the International System of Units measurement for absorbed dose of ionizing radiation. For both metrics lower scores indicate better adherence to clinical objectives.

<!-- Explanation of 2 main limitation: no code and no physics -->
## Limitations
TrDosePred achieves promising results using a machine learning-based approach. However, a critical limitation of TrDosePred is the absence of inherent physics knowledge. The architecture lacks explicit information about the physics of radiation and dose prediction, despite the fact that the predicted dose is highly dependent on how radiation particles move and interact with matter. Incorporating more physics information into the model and training it in a way that aligns with physical principles might lead to dose predictions that are more accurate in relation to the actual treatment plan.

The radiation that is applied during radiotherapy is administered through beams. This means that the dosage of voxels that these beams pass through are very strongly related. For each beam, it should be the case that the amount of dosage reduces as the beam passes through tissue and is absorbed. However, this information is not something that is currently incorporated in the model architecture or loss function.

Another limitation of TrDosePred is the unavailability of the code [[4]](#4). This makes it difficult to reproduce the architecture and continue research in this direction.

# Reproduction
To ensure reproducibility of the results, the SWIN-based architecture was initially implemented. However, due to insufficient information regarding the parameters of this architecture such as the number of multihead attention blocks and the number of layers in the MLP, the implemented model failed to achieve proper learning. Therefore, an alternative architecture incorporating transformers, namely the UNETR architecture, was used [[22]](#22).

## UNETR
UNTER, which stands for UNEt TRansformers, is a hybrid model combining the strengths of convolutional neural networks (CNNs), specifically the U-Net architecture, with transformer-based attention mechanisms.

The UNETR architecture first embeds non overlapping patches with a dimensionality of 16 from the image. These patches are then projected into a K dimensional embedding space using a linear layer. A learnable one dimensional embedding is then added to preserve positional information. The final embeddings then have a size of 768. These embeddings are then passed through a stack of 12 transformer blocks, each constisting of a layer of multi-head self-attention (MSA) and multilayer perceptrons (MLPs). Following:

$$\textbf{z}'_i = \text{MSA}(\text{Norm}(\textbf{z}_{1}))$$

<--+ \textbf{z}_{i-1}$$ -->

$$\textbf{z}'_i = \text{MLP}(\text{Norm}(\textbf{z}_{i})) + \textbf{z}'_{i}$$

Where $\text{Norm}$ indicates Layer Norm and each MLP constist of two linear layers with a $\text{GELU}$ activation function. Each MSA sublayer includes $n$  self-attention (SA) heads. Each SA block is a parameterized function that maps queries ($q$) to corresponding keys ($k$) and values ($v$) within a sequence $z \in \mathbb{R}^{N \times K}$. To compute the attention wheights the similarity between elements in $z$ and their key-value pairs is evaluated as follows:

$$A = \text{Softmax}(\frac{\textbf{qk}^T}{\sqrt K_h})$$

where $K_h$ is used as a scaling factor. This attention can the be used compute the output of a single SA block as follows:

$$ \text{SA}(\textbf{v}) = A\textbf{v}$$

The output of the entire MSA block can then be formulated as follows:

$$\text{MSA}(\textbf{z}) = [\text{SA}_1(\textbf{z}), \text{SA}_2(\textbf{z}), ...,\text{SA}_n(\textbf{z}) ]W_{\text{msa}}$$

where $W_{\text{msa}}$ are trainable parameter weights for the MSA block.

Features from different levels of resolution from the encoder are then merged with the decoder to form a U-net inspired architecture. At each resolution, the features are projected from the embedding space of that resolution into the input space by utilizing 3x3x3 convolutional layers, followed by batch normalization layers. This is then concatenated with the of the following transformer block that is upsampeled using a deconvolutional layer.

After all the features from all resolutions have been combined, the resulting features are fed into a 1x1x1 convolutional layer followed by a softmax activation function to generate the final dose prediction.

## Training
Training has been setup using hugging face such that data preprocessing is fully parallelized. During training use was made of the PyTorch `DataParallel` module in order to train in parallel on multiple GPUs. Each model has been trained for 300 epochs using NVIDIA A1000 GPUs with a batch size of 4 and a learning rate of $4 \times 10^{-4}$.

<!-- Introduction to contributions -->
# TrDosePred - A physics-based approach
<!-- Describe your novel contribution. -->
This work introduces a physics-based approach of TrDosePred. Specifically, the TrDosePred framework is augmented with physics-based elements. The expectation is that these elements will make the model perform better, given the inherently physics-dependent nature of radiation.

Multiple methdologies exist for integrating physics into neural networks [[18]](#18). In this research, two specific approaches are explored. Firstly, the loss function is augmented with a physics-based component. Secondly, an autoregressive strategy is employed to capture dependencies between different segments of the prediction. The subsequent sections provide a detailed explanation of these two methodologies.

<!-- Explanation of loss variants-->
<!-- First of all, training the model with a physics-inspired loss function might lead to increased physics-awareness of the model. In other fields, using these type of loss functions lead to increased performance and stability of the network [[2]](#2) [[3]](#3). This technique is part of Physics-Informed Neural Networks (PINNs). -->
## Physics-based loss
In fields where physics plays a critical role, such as turbulence modelling, the use of physics-inspired loss functions has demonstrated promising results [[2]](#2) [[3]](#3). These loss functions have been shown to enhance both the performance and stability of the neural networks to which they are applied.

For dose prediction, common loss functions include the mean absolute error (MAE) and mean squared error (MSE). TrDosePred utilizes the MAE loss function. In this context, $D_{pred}$ represents the predicted dose and $D_{true}$ is the ground truth.

$$L_{MAE}(D_{pred}, D_{true}) = \frac{1}{N}\sum_{i}|D^i_{pred} - D^i_{true}|$$

A frequently used technique involves modifying the loss function of a model by incorporating physics-based regularization terms. Specifically, this approach utilizes the MAE loss as a foundation, upon which a weighted physics-based loss component is added. This additional loss term can be domain-specific and encodes relationships that are particularly relevant to the domain of the model's application.

\begin{equation}
    Loss = L_{MAE}(D_{pred}, D_{true}) + w_{phy} Loss_{phy}(D_{pred}).
\end{equation}

Here,$Loss_{phy}$ is the physics-based loss and $w_{phy}$ is the weight given to the physics-based loss.

Typically, the weight $w_{phy}$ is selected such that the contribution of the physics-based loss is smaller than that of the primary loss function, which, in our case, is the MAE loss. This ensures that while the physics-based constraints influence the model, they do not overshadow the main predictive objective.

### DVH loss
<!-- In dose prediction, dose-volume histograms (DVHs) are commonly used to evaluate treatment plans [[6]](#6). DVHs are used to quantify the dose distribution around a target. They display the absorbed dose of an organ, over the relative volume of the organ that reached this dose. An example is shown in Figure 6.

<!-- ![Example-of-dose-volume-histogram-DVH-computed-with-MiM-Sureplan-701-software-research](https://hackmd.io/_uploads/H1RgWJSXA.png)
Figure 6: An example of a dose-volume histogram. Here, the x-axis displays the absorbed dose, while the y-axis explains the volume of the organ that absorbed that dose. -->

<!-- <div style="text-align: center;">
    <img src="https://hackmd.io/_uploads/H1RgWJSXA.png" alt="DVH example"/>
    <p>Figure 6: An exemplary dose-volume histogram. Here, the x-axis displays the absorbed dose and the y-axis explains the volume of the organ that absorbed that dose. Every line represents a different structure.</p>
</div> -->

DVHs are essential for ensuring that the prescribed radiation dose effectively targets the tumor while minimizing exposure to healthy tissues and the critical organs. Therefore, incorporating DVH information into the model training process can be beneficial. To this end, a DVH loss function has been proposed[[5]](#5), which is a differential approximation of the DVH.

Given a binary segmentation mask, $B_s$ for the $s$-th structure, along with the predicted and ground truth doses, the mean squared loss of the DVH can be defined as follows:

\begin{equation}
    L_{DVH}(D_{true}, D_{pred}, B_{s}) = \frac{1}{n_s}\frac{1}{n_t}\sum_s \lVert DVH(D_{true}, B_s) - DVH(D_{pred}, B_s) \rVert_2^2.
\end{equation}

where $n_s$ represents the number of structures and $n_t$ denotes the number of different bins in the histogram.
This loss function can be integrated into the total loss function as follows:
\begin{equation}
    Loss = L_{MAE} + w_{DVH}\cdot L_{DVH},
\end{equation} where $w_{DVH}$ is the weight given to the DVH loss.

The DVH of a dose and a structure mask for the $s$-th structure can be expressed as:

$$DVH(D, B_s) = (v_{s, d_1}, v_{s, d_2}, ..., v_{s, d_n})$$

where $v_{s, d_t}$ represents the volume-at-dose corresponding to the dose $d_t$. Each value in the DVH corresponds to a distinct bin in the histogram. Specifically, $v_{s, d_t}$ is defined as the fraction of the volume of a region-of-interest (ROI), which can be either an OAR or a PTV, receiving a dose of at least $d_t$. This value can be approximated as:

$$v_{s, t}(D, B_s) = \frac{\sum_i\sigma(\frac{D(i) - d_t}{\beta})B_s(i)}{\sum_iB_s(i)}.$$

Here, $\sigma$ denotes the sigmoid function, $\sigma(x) = \frac{1}{1+e^{-x}}$, $D(i)$ is the dose at voxel $i$, $d_t$ is the threshold dose, $\beta$ represents the histogram bin width and $B_s(i)$ whether voxel $i$ belongs to the $s$-th structure.

DVH loss can be regarded as a physics-based loss function because it directly incorporates the physical properties and constraints of radiation dose distribution into the loss calculation.

### Moment loss
Moment loss is a variant of the DVH loss. It is based on the concept that a DVH can be approximated using several moments of a structure, which are different quantative measures to represent a function, such as the mean or the maximum [[5]](#5). A DVH can be approximated using several moments as follows:

\begin{equation}
DVH \sim (M_1, M_2, ..., M_p).
\end{equation}

Here, $M_p$ represents the moment of order p, which is defined as:
$$M_p = \left(\frac{1}{|V_s|}\sum_{j\in V_s}d^p_j\right) ^\frac{1}{p}$$

where $V_s$ denotes the voxels belonging to the $s$th structure and $d$ represents the dose.

Different moments of a structure capture various characteristics of the structure. For instance, $M_1$ corresponds to the mean dose, while $M_\inf$ corresponds to the maximum dose. In practice, the 10th moment ($M_{10}$) can be used to approximate the maximum dose. In our experiments, the moments 1, 2 and 10 are used to compute the loss, following the work of [[5]](#5).

Based on the DVH approximation, the final moment loss is calculated as:
$$L_{moment} = \sum_{p\in P}||M_p - \tilde{M}_p ||_2^2$$

Here, $\tilde{M}_p$ denotes the corresponding moment derived from the predicted dose.

Integrating the moment loss function in the model training process is analogous to incorporating the DVH loss and is represented as follows:

$$Loss = L_{MAE} + w_{Moment}\cdot L_{Moment},$$

where $w_{Moment}$ denotes the weight assigned to the moment loss function. Lastly, following the research of [[5]](#5), the MAE, DVH loss and Moment loss can be combined into a unified loss function:

\begin{equation}
    Loss = L_{MAE} + w_{DVH}\cdot L_{DVH} + w_{Moment}\cdot L_{Moment}
\end{equation}

<!-- Explanation of Autoregression -->
## Autoregression
<!-- Two-three methods -->

Autoregressive methods are used to predict sequences by conditioning on previous predictions. In the context of dose prediction, autoregression helps in model dependencies between different slices of the predicted dose. These slices could be axial slices (across X, Y or Z) or can be along the beam eye view (BEV) axes, which correspond to the directions of the various radiation beams. There are multiple approaches to incorporating autoregression in this context.

#### 1. Autoregressive input masking
In the first method, an additional channel is added to the model input, which is a masked 3D dose. Thus, the input is a concatentation of CT, PTV, OAR and the masked 3D dose:

$$x = [CT, PTV, OAR, Mask],$$

where $Mask$ is the masked 3D dose. Based on this input, the model predicts a small slice of the dose at each step. The prediction is formulated as:

$$D_{\text{pred, slice}} = f(x).$$

Here, $f$ denotes the function implemented by the model to predict the dose slice from the concatenated input.

After predicting a slice, it is incorporated back into the masked 3D dose input for the subsequent prediction. This iterative process ensures that the model leverages its previous predictions to inform future ones. The process continues until the entire dose volume is predicted.
The loss function is calculated based on the incrementally predicted slices:

$$L_{\text{slice}}(D_{\text{pred, slice}}, D_{\text{true,slice}} ) = \frac{1}{K} \sum_{j}^K \left| D^j_{\text{pred, slice}} - D^j_{\text{true,slice}} \right|$$

where $D_{\text{pred, slice}}$ represents the predicted slice dose, $D_{\text{true,slice}}$ represents the ground truth slice dose and $K$ denotes the total number of elements in the slice.

##### Teacher forcing extension
Teacher forcing is a technique used in training autoregressive models to improve performance. During training, rather than using the model's own predictinos as inputs for subsequent steps, ground truth data is utilized. This involves feeding observed sequence values (i.e. ground-truth samples) back into the model after each step with a certain probability, guiding the model to remain close to the ground truth sequence. This method can teach the model to be inherently autoregressive by implicitly learning to make the next slice prediction.

In the context of our dose prediction model, teacher forcing can be applied by replacing the predicted dose slices with the ground truth dose slices during the training process. Mathematically, this implies that instead of updating the masked 3D dose input with $D_{pred, slice}$, it is updated with $D_{true, slice}$.

This apporach may be particularly beneficial when conditioning along BEV axes, as it allows for incremental updates to the model with information from the previous beam.

#### 2. RNN based neural network
Another technique to incorporate autoregressiveness into the model is by modifying the model's architecture. In the default model setup, the model uses a decoder that predicts the entire $D_{\text{pred}}$ at once. We aim to replace this decoder with an RNN-based decoder to introduce autoregression, enabling the model to predict dose slices sequentially. Instead of feeding the masked input back into the model, as in the first autoregressive method, the RNN leverages its hidden states to maintain context and continuity between predictions.

<div style="text-align: center;">
    <img src="https://hackmd.io/_uploads/rJNY4jeVA.png" alt="architecture"/>
    <p>Figure 3: Simple version of an RNN. <a href="#4"></a></p>
</div>

A typical RNN works by the following formula:
$$h_t = \tanh(x_t W_{ih}^T + b_{ih} + h_{t-1} W_{hh}^T + b_{hh}),$$ where $h_t$ is the hidden state at time $t$, $x_t$ is the input at time $t$, and $h_{t-1}$ is the hidden state from the previous time step ($t-1$) or the initial hidden state at time $0$. However, since we are dealing with 3D structures, we implement a convolutional approach (ConvRNN), which functions similarly but is adapted for 3D inputs.

<div style="text-align: center;">
    <img src="https://hackmd.io/_uploads/BJtmkTg40.png" alt="architecture"/>
    <p>Figure 3: A UNETR inspired Convolutional RNN. <a href="#4"></a></p>
</div>

The UNETR architecture is modified to use a ConvRNN in the output decoding. The ConvRNN process starts from the latent dimension produced by the UNETR encoder, which is a combination of three residual streams.

To make computation more feasible, the size of the residual stream is reduced from $768$ to $128$ and the patch size is increased from $16$ to $32$. The latent representation $\textbf{z}$ obtained from the encoder, includes features from CT, PTV, and OAR:

$$
\textbf{z} = [z_3, z_6, z_9] = \text{UNETR_Encoder}\left([CT, PTV, OAR]\right).
$$

The ConvRNN processes the latent features and maintains hidden states $h_t$ that capture information about previous predictions. The initial hidden state $h_0$ is initialized as $\textbf{z}$. This allows the model to perform a single forward pass to predict the entire dose volume, decoding it slice by slice.

Let $\mathbf{x}$ be the CT image and its features, $\text{Conv}(\mathbf{x}) = \mathbf{f}$ be the convolutional block producing the feature map $\mathbf{f}$, and $\text{Slice}(\mathbf{f}) = f_t$ be the operation that selects a specific slice, resulting in the slice feature map $f_t$.
$$\mathbf{x} \xrightarrow{\text{Conv}(\cdot)} \mathbf{f} \xrightarrow{\text{Slice}(\cdot)} f_t$$ At each time step $t$, the ConvRNN then updates its hidden state based on the previous hidden state, the latent representation and the slice-specific CT feature map:
\begin{equation}
o_t, h_t = \text{ConvRNN}\left(h_{t-1}, f_t\right),
\end{equation} where $h_{t-1}$ is the hidden state from the previous time step, $f_t$ are features specific to the current slice and $o_t$ is the output from the current ConvRNN step.

Each output is then concatenated to form $\textbf{o}$. This is then projected to $\mathbf{o}'$. Finally $\mathbf{o}'$ is jointly decoded with the CT feature map to a final $D_{pred}$:
$$D_{pred} = \text{DecoderConv} \left(\mathbf{f}, \mathbf{o}'\right)$$

<!-- A slight modification of this approach is to not use the CT feature map as $x_t$, but rather the previous ConvRNN output $o_{t-1}$.

\begin{equation}
h_t = \text{ConvRNN}(h_{t-1}, o_{t-1}).
\end{equation}  -->

<!-- #### 3.
3. Neural translation method
* Based on translation task
* Uses any type of encoder, and transformer based decoder
* Encoder can use full input
* Decoder has for each prediction acess to the encoded state which has global information
* Decoder predicts slices autoregressively using masked self-attention, thus only allowing the model to use information of previoulsy predicted slices for its predictions.
 -->

<!-- Explanation of KANs?-->

# Results
In summary, various experiments were performed. First of all, the original research of TrDosePred [[4]](#4) was reproduced, using the UNETR architecture instead of the original SWIN-based architecture. This reproduction was done using to variants of the UNETR architecture, one with three blocks and one using four blocks. The results of this reproduction are shown in Table 1.

| Architecture | DVH score | Dose score |
| -------- | -------- | -------- |
| 3-block UNETR     | #TODO     | #TODO     |
| 4-block UNETR     | #TODO     | #TODO     |

Table 1: results of the reproduction of TrDosePred using a UNETR architecture. Best results are denoted in bold.

TODO: comments about reproduction results -> which architecture is used for the following experiments?

After creating a 3D-dose prediction model, various experiments regarding incorporating physics-based elements were done. These experiments can be divided into two categories: using a physics-based loss and using an autoregression approach.

In Table 2, the results of using a physics-based loss are displayed. In total, three experiments were done. The baseline used is the model only using the MAE loss for training. Furthermore, a MAE+DVH loss was used, a MAE+Moment loss was used, and a MAE+DVH+Moment loss was used. The parameters used to weigh the different loss functions have been chosen such that the DVH loss and moment loss are roughly a factor ten smaller then the MAE loss. The $w_{DVH}$ that was used was $1 \times 10^{-5}$, $w_{Moment}$ was chosen to be $5 \times 10^{-6}$.
#TODO analyse results

| Loss | DVH score | Dose score |
| -------- | -------- | -------- |
| MAE     | #TODO     | #TODO     |
| MAE + DVH    | #TODO     | #TODO     |
| MAE + Moment    | #TODO     | #TODO     |
| MAE + DVH + Moment    | #TODO     | #TODO     |

Table 2: results of the different physics-based loss functions. Best results are denoted in bold.

<Note: while the final experiments are not done, from preliminary results was observed that the usage of both a DVH and a moment loss resulted in better scores).

Lastly, an autoregressive approach was tried. For this, two different techniques that incorporate autoregression in the model were used. These results are shown in Table 3 #TODO

| Autoregression | DVH score | Dose score |
| -------- | -------- | -------- |
| No autoregression     | #TODO     | #TODO     |
| Autoregression by applying ...    | #TODO     | #TODO     |
| Autoregression by applying ...    | #TODO     | #TODO     |

Table 2: results of the different autoregressive methods. Best results are denoted in bold.

# TODO
qualitative results for everything, at least some images of how the dose predictions look like, if possible some sort of animation over slices.

# TODO
Comparison with results from the original paper, and corresponding reflection

<!-- ## Analysis of TrDosePred -->
<!-- An analysis of the paper and its key components. Think about it as a nicely formatted review as you would see on OpenReview.net. It should contain one paragraph of related work as well. -->
<!-- TrDosePred achieves impressive performance on the OpenKBP dataset, with a dose score of 2.426 Gy and a DVH score of 1.592 Gy, ranking 1st and 3rd, respectively. The model demonstrates this performance with limited data and holds potential for further improvement with larger datasets. Additional ablation studies highlighted the effectiveness of key architectural components in enhancing performance, such as the convolutional sampling strategy and the depth-wise convolution in the multi-layer perceptron (MLP). -->

<!-- Exposition of its weaknesses/strengths/potential which triggered your group to come up with a response. -->

<!-- Explanation of strenghts -->
<!-- In summary, the strengths of the overall approach include the innovative use of transformers, making TrDosePred the first to demonstrate the transformer architecture in dose prediction, achieving state-of-the-art performance in accurate dose predictions, and validating the model's effectiveness through ablation studies that identify key components contributing to its performance. -->

# Conclusion

## Future work
When continuing the research on incorporating physics-based elements in the field of 3D dose distribution using transformers, various topics might be interesting to research. In this research, a main limitation was the limited dataset. The dataset used only contained the final true 3D dose distribution. However, in practice, there is a 3D dose distribution per radiation beam, on which the final treatment plan is based. This data allows for more physics-based elements to be introduced into the transformer. Two key elements that could be added are using a Lambert-based loss function and doing autoregression along the beam axis. The Lambert law is based on the idea that every structure absorbs a different amount of radiation. This is something that is extremely relevant in the field of dose prediction and could make the final predicted distribution more accurate. Secondly, the dose distribution for different beams rely heavily on each other, since one beam may be able to radiate a specific structure while another beam cannot radiate this structure without damaging an organ. Therefore, applying autoregression per beam might make the final distribution more accurate.

In addition to this, future research might look into reproducing the original work of TrDosePred with the SWIN-3D architecture. In this research, it was not possible to do this due to the lack of detail in the original paper. Contacting the authors was attempted, but unfortunately, no response was received. Reproducing the original work with help of the authors might make the field of 3D dose prediction using transformer-based architecture more accessible for further research.

## Individual contributions
<!-- Close the notebook with a description of each student's contribution. -->
This project can roughly be divided into three components: reproducing the original paper, creating physics-based loss functions and making the original model autoregressive. The main focus of Jakob and Gijs was the reproduction, the main focus of Macha was the physics-based losses and the main focus of Derck and Thijmen was the autoregression.

## References
<a id="1">[1]</a> Baskar, R., Lee, K. A., Yeo, R., & Yeoh, K. W. (2012). Cancer and radiation therapy: current advances and future directions. International journal of medical sciences, 9(3), 193.

<a id="2">[2]</a> List, B., Chen, L. W., & Thuerey, N. (2022). Learned turbulence modelling with differentiable fluid solvers: physics-based loss functions and optimisation horizons. Journal of Fluid Mechanics, 949, A25.

<a id="3">[3]</a> Raymond, S. J., & Camarillo, D. B. (2021). Applying physics-based loss functions to neural networks for improved generalizability in mechanics problems. arXiv preprint arXiv:2105.00075.

<a id="4">[4]</a> Hu, Chenchen, et al. "TrDosePred: A deep learning dose prediction algorithm based on transformers for head and neck cancer radiotherapy." Journal of Applied Clinical Medical Physics 24.7 (2023): e13942.

<a id="5">[5]</a> Nguyen, D., McBeth, R., Sadeghnejad Barkousaraie, A., Bohara, G., Shen, C., Jia, X., & Jiang, S. (2020). Incorporating human and learned domain knowledge into training deep neural networks: a differentiable dose‐volume histogram and adversarial inspired framework for generating Pareto optimal dose distributions in radiation therapy. Medical physics, 47(3), 837-849.

<a id="6">[6]</a> Drzymala, R. E., Mohan, R., Brewster, L., Chu, J., Goitein, M., Harms, W., & Urie, M. (1991). Dose-volume histograms. International Journal of Radiation Oncology* Biology* Physics, 21(1), 71-78.

<a id="7">[7]</a> Babier, A., Zhang, B., Mahmood, R., Moore, K. L., Purdie, T. G., McNiven, A. L., & Chan, T. C. (2021). OpenKBP: the open‐access knowledge‐based planning grand challenge and dataset. Medical Physics, 48(9), 5549-5561.

<a id="8">[8]</a> Moghaddasi, Leyla & Bezak, Eva & Marcu, Loredana. (2012). In Silico Modelling of Tumour Margin Diffusion and Infiltration: Review of Current Status. Computational and mathematical methods in medicine. 2012. 672895. 10.1155/2012/672895.

<a id="9">[9]</a> Ma, C. M., Li, J. S., Pawlicki, T., Jiang, S. B., Deng, J., Lee, M. C., ... & Brain, S. (2002). A Monte Carlo dose calculation tool for radiotherapy treatment planning. Physics in Medicine & Biology, 47(10), 1671.

<a id="10">[10]</a> Jabbari, K. (2011). Review of fast Monte Carlo codes for dose calculation in radiation therapy treatment planning. Journal of Medical Signals & Sensors, 1(1), 73-86.

<a id="11">[11]</a> Jia, X., Gu, X., Graves, Y. J., Folkerts, M., & Jiang, S. B. (2011). GPU-based fast Monte Carlo simulation for radiotherapy dose calculation. Physics in Medicine & Biology, 56(22), 7017.

<a id="12">[12]</a> Meerbothe, T. (2021). A physics guided neural network approach for dose prediction in automated radiation therapy treatment planning.

<a id="13">[13]</a> Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. In Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18 (pp. 234-241). Springer International Publishing.

<a id="14">[14]</a> Gronberg, M. P., Gay, S. S., Netherton, T. J., Rhee, D. J., Court, L. E., & Cardenas, C. E. (2021). Dose prediction for head and neck radiotherapy using a three‐dimensional dense dilated U‐net architecture. Medical physics, 48(9), 5567-5573.

<a id="15">[15]</a> Kontaxis, C., Bol, G. H., Lagendijk, J. J. W., & Raaymakers, B. W. (2020). DeepDose: Towards a fast dose calculation engine for radiation therapy using deep learning. Physics in Medicine & Biology, 65(7), 075013.

<a id="16">[16]</a> Nguyen, D., Jia, X., Sher, D., Lin, M. H., Iqbal, Z., Liu, H., & Jiang, S. (2019). 3D radiotherapy dose prediction on head and neck cancer patients with a hierarchically densely connected U-net deep learning architecture. Physics in medicine & Biology, 64(6), 065020.

<a id="17">[17]</a> Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2020). An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929.

<a id="18">[18]</a> Karpatne, A., Watkins, W., Read, J., & Kumar, V. (2017). Physics-guided neural networks (pgnn): An application in lake temperature modeling. arXiv preprint arXiv:1710.11431, 2.

<a id="19">[19]</a> Xiao, T., Singh, M., Mintun, E., Darrell, T., Dollár, P., & Girshick, R. (2021). Early convolutions help transformers see better. Advances in neural information processing systems, 34, 30392-30400.

<a id="20">[20]</a> Shanmugam, D., Blalock, D., Balakrishnan, G., & Guttag, J. (2021). Better Aggregation in Test-Time Augmentation. arXiv preprint arXiv:2011.11156.

<a id="21">[21]</a> Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., & Guo, B. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. arXiv preprint arXiv:2103.14030

<a id="22">[22]</a> Hatamizadeh, A., Tang, Y., Nath, V., Yang, D., Myronenko, A., Landman, B., Roth, H., & Xu, D. (2021). UNETR: Transformers for 3D Medical Image Segmentation. arXiv preprint arXiv:2103.10504

# Appendix

## Swin Components

This section goes into detail on specific components of the SWIN model architecture [[21]](#21).

### Patch Embedding and Expanding Block

Traditionally in ViTs, the input image is split and mapped to non-overlapping patches before being fed into the transformer encoder. However, recent research suggests that using overlapping patches can improve optimization stability and performance [[19]](#19). Inspired by this, TrDosePred's patch embedding block extracts patches from the input volume using stacked overlapping convolutional layers.

The patch embedding block comprises three submodules, each with a 3×3×3 convolution, an Instance Normalization, and a Gaussian Error Linear Units (GELU) activation function. A point-wise convolution with 96 filters projects these features into embedding tokens, reducing the feature dimensions by a factor of 2×4×4 (Figure 4a).

Symmetrically, a patch expanding block with a 2×4×4 transpose convolution and 3×3×3 convolutions is used to recover the resolution of feature maps after decoding. A point-wise convolution is then employed to generate the final dose prediction (Figure 4b).

<div style="display: flex; justify-content: center; align-items: center; gap: 20px;">
    <div style="text-align: center;">
        <img src="https://hackmd.io/_uploads/HkjhFIvmC.png" alt="patch-embedding" width="300"/>
        <p>Figure 4a: Patch embedding block <a href="#4">[4]</a></p>
    </div>
    <div style="text-align: center;">
        <img src="https://hackmd.io/_uploads/S1faKUPXC.png" alt="patch-expanding" width="300"/>
        <p>Figure 4b: Patch expanding block <a href="#4">[4]</a></p>
    </div>
</div>

### Transformer-Based Encoder and Decoder

After patch embedding, the tokens are fed into a U-shaped encoder and decoder, featuring multiple 3D Swin Transformer blocks. Compared to the vanilla transformer, the Swin Transformer is more efficient for medical image analysis due to its linear complexity relative to image size.

Each 3D Swin Transformer block consists of a window-based local multi-head self-attention (W-MSA) module and a Multi-layer Perceptron (MLP) module (Figure 5). Depth-wise convolution is added to the MLP to enhance locality, and Layer Normalization (LN) and residual connections are applied before and after each module.

The windows are cyclically shifted between consecutive transformer blocks to establish cross-window connections. The computational steps for two consecutive 3D Swin Transformer blocks are as follows:

1. The input to the first block is normalized using LN and then processed by the 3D W-MSA module.
2. The output of the W-MSA module is added to the input via a residual connection.
3. This output is then normalized again and passed through the MLP module.
4. The output of the MLP module is added to the input via another residual connection.

\begin{equation}
Z_i' = \text{3D W-MSA}(\text{LN}(Z_{i-1})) + Z_{i-1}
\end{equation}

\begin{equation}
Z_i = \text{MLP}(\text{LN}(Z_i')) + Z_i'
\end{equation}

For the next block, the same steps are repeated with a shifted window-based self-attention:

\begin{equation}
Z_{i+1}' = \text{3D SW-MSA}(\text{LN}(Z_i)) + Z_i
\end{equation}

\begin{equation}
Z_{i+1} = \text{MLP}(\text{LN}(Z_{i+1}')) + Z_{i+1}'
\end{equation}

Here, $Z_i'$ and $Z_i$ denote the output of the 3D(S)W-MSA and MLP module for the $i$-th block, respectively.

The attention in each 3D local window is computed as:

\begin{equation}
\text{Attention}(Q, K, V) = \text{SoftMax}\left(\frac{QK^T}{\sqrt{d_k}} + B\right)
\end{equation}

where $Q$, $K$, $V$ represent the query, key, and value matrices; $d_k$ is the dimension of the query and key, and $B$ is the bias matrix.

Between the encoder and decoder blocks, down-sampling and up-sampling layers are inserted to adjust the feature map sizes as described in the previous section.

<div style="text-align: center;">
    <img src="https://hackmd.io/_uploads/H126F8vmC.png" alt="swin-transformers"/>
    <p>Figure 5: Two consecutive Swin Transformer blocks <a href="#4">[4]</a></p>
</div>
