---
layout: single
toc: true
toc_sticky: true
---

> As discussed in previous Posts, segmentation tasks origins from semantic segmentation and developed to instance segmentation further to panoptic segmentation. In this post, we shall discuss the metrics to various metrics, including IOU, AP, PQ, etc.

## Basic Metrics and Definitions

### Confidence Score
Confidence score is not exactly a metric to describe the performance of the model. Still, it describes the probability that an anchor box contains object detection or a pixel value belonging to a specific class. It is usually the output of the softmax function from the `classifier`. Confidence score is an important value of classifier saying confidence on prediction, thus is often used to find the most confident detection. However, the corelation between a prediction(what is good?) and confidence score need to be examined, this is often ignored.

Confidence score are usually the output of softmax function, which is described as following:

\begin{equation}
P(y=j\mid \mathbf {x} )={\frac {e^{\mathbf {x} ^{\mathsf {T}}\mathbf {w} _{j}}}{\sum _{k=1}^{K}e^{\mathbf {x} ^{\mathsf {T}}\mathbf {w} _{k}}}}
\end{equation}

This can be seen as the composition of K linear functions ${ \mathbf {x} \mapsto \mathbf {x} ^{\mathsf {T}}\mathbf {w} _{1},\ldots ,\mathbf {x} \mapsto \mathbf {x} ^{\mathsf {T}}\mathbf {w} _{K}}$ and the softmax function (where ${ \mathbf {x} ^{\mathsf {T}}\mathbf {w} }$  denotes the inner product of ${ \mathbf {x}}$  and ${ \mathbf {w} }$, $\mathbf{x}$ is often called logits). The operation is equivalent to applying a linear operator defined by ${ \mathbf {w} }$  to vectors ${ \mathbf {x} }$ , thus transforming the original, probably highly-dimensional, input to vectors in a K-dimensional space ${ \mathbb {R} ^{K}}$.

We usually take a threshold to divide prediction as positive or negative(imaging that someone says i am positive that it is balabala). For example, a threshold 0.5 divide probabilities in [0.0, 0.5) and [0.5, 1.0] as negative, and positive.
### Confusion Matrix

A table of confusion (sometimes also called a confusion matrix) is a table with two rows and two columns that reports the number of false positives, false negatives, true positives, and true negatives. This allows more detailed analysis than mere proportion of correct classifications (accuracy). Accuracy will yield misleading results if the data set is unbalanced; that is, when the numbers of observations in different classes vary greatly. For example, if there were 95 cats and only 5 dogs in the data, a particular classifier might classify all the observations as cats. The overall accuracy would be 95%, but in more detail the classifier would have a 100% recognition rate (sensitivity) for the cat class but a 0% recognition rate for the dog class. F1 score is even more unreliable in such cases, and here would yield over 97.4%, whereas informedness removes such bias and yields 0 as the probability of an informed decision for any form of guessing (here always guessing cat).
<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-9wq8{border-color:inherit;text-align:center;vertical-align:middle}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow">Total population</th>
    <th class="tg-c3ow" colspan="2">Ground truth condition</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-9wq8" rowspan="2">Predicted<br>condition</td>
    <td class="tg-c3ow">True Positive(TP)</td>
    <td class="tg-c3ow">False Positive(FP)</td>
  </tr>
  <tr>
    <td class="tg-c3ow">False Negative(FN)</td>
    <td class="tg-c3ow">True Negative(TN)</td>
  </tr>
</tbody>
</table>


### Precision & Recall

Precision Recall curve(PR-curve) is used to measure the performance of a model by tuning confidence score, precision, and recall from a curve showing the compromise of precision and recall. The larger the area under that curve is, the better performance the model has. The area is what we call average precision. However, such a curve is not convenient to compare models, especially when the curve is noisy and intersects with the saw-tooth shape. 

Precision is defined as the number of true positives divided by the sum of true positives and false positives:
\begin{equation}
    Precision = \frac{|TP|}{|TP| + |FP|}
\end{equation}
The recall is defined as the number of true positives divided by the sum of true positives and false negatives:
\begin{equation}
    Recall = \frac{|TP|}{|TP| + |FN|}
\end{equation}
Users are usually interested in Precision, which indicates how many detections are right in all detection from models. The producers are interested in Recall, which indicated how many detections are right in all the dataset objects. Some other metrics, such as precision-recall curve recall-IoU-curve, are not what we are interested in. 

### IOU

\begin{equation}
    IoU = \frac{Area(B_p \cap B_{gt})}{Area(B_p \cup B_{gt})} = \frac{|TP|}{|TP| + |FP| + |FN|}
\end{equation}

Where $B_p$ is prediction and $B_{gt}$ is ground truth. In pixel-wise segmentation, the intersection and union count pixels of entities. In object detection, it is the object bounding box or bounding circles or any plane geometry that you like.


## Object Detection & Instance Segmentation Metrics
we take average precision(AP), mean average precision(mAP), average recall(AR), mean average recall(mAR) as metrics for Object Detection. We normally set IOU larger than 0.5 as positive instances and positive boxes, axis-aligned and rotated, IOU lower than 0.5 is seen as False Positive. For specific metrics w.r.t IOU, IOU value shall be named.

Average precision (AP) serves as a measure to evaluate the performance of object detectors. It is a single metric that encapsulates precision and recall and summarizes the Precision-Recall curve by averaging precision across recall values from 0 to 1.


\begin{align}
    AP &= \sum(r_n - r_{n-1})p_{interp}(r_n)
\end{align}
\begin{align}
    mAP &= \frac{1}{N}\sum_{n=1}^{N}AP_i
\end{align}
where

$p_{interp}(r_n)$ = $\max_{r^\prime > r_{n}} p(r^\prime)$

In PASCAL VOC \cite{Everingham2010PascalVOC} Dataset, if multiple detections correspond to 1 ground truth, only 1 detection with the highest score counts as positive, and the remaining counts as FP. Besides, PASCAL takes predictions with IOU larger than 0.5 to calculate AP. However, COCO takes AP's mean value with a different AP threshold interval [0.5,0.05,0.95]. Therefore, COCO not only averages AP over all classes but also on the defined IoU thresholds.

Average recall describes the area doubled under the Recall x IoU curve. The Recall x IoU curve plots recall results for each IoU threshold where IoU ∈ [0.5,1.0], with IoU thresholds on the x-axis and recall on the y-axis.

\begin{equation}
    AR = 2\int_{0.5}^{1} recall(IOU) d(IOU)
\end{equation}

We take metrics mostly from the COCO dataset, which takes metrics in more detail with different settings.
we take mAP as AP@[.50: .05: .95] as principle metric, and AP with IOU threshold [0.5, 0.75] respectively. considering building with different size, AP and AR is divided to small$[0,32^2]$, medium$[32^2,96^2]$, large$[96^2,\infty)$ w.r.t size of objects.

For instance Segmentation, the box is replaced with instance polygons and mapped to images as a mask.
\subsection{Semantic Metrics}
suppose $n_{jj}$ means the number of true positives for class j,$t_j$ total number of classes labeled as class j, $n_{ij}$ false positives,$n_{ji}$ false negatives, metrics can be represented as follows. PA \cref{eq:PA} is the number of correctly classified pixels over all pixels, MPA \cref{eq:MPA} is the averaged number of correctly classified pixels over pixels of a class. MIOU is given in \cref{eq:MIOU} in a more adaptable manner to pixel-wise calculation. FWIoU \cref{eq:FWIoU} is weighted IOU according to the frequency of each class.
\begin{equation}\label{eq:PA}
    Pixel Accuracy(PA) = \frac{\sum_{j=0}^k{n_{jj}}}{\sum_{j=0}^k t_j}
\end{equation}
\begin{equation}\label{eq:MPA}
    Mean Pixel Accuracy(MPA) = \frac{1}{k}\sum_{i=0}^k \frac{n_{jj}}{t_j}
\end{equation}
\begin{equation}\label{eq:MIOU}
    MIoU = \frac{1}{k}\sum_{i=0}^k \frac{n_{jj}}{ n_{ij} + n_{ji} + n_{jj}}
\end{equation}
\begin{equation}\label{eq:FWIoU}
    FWIoU = \frac{1}{\sum_{j=0}^k t_j}\sum_{i=0}^k t_j \frac{n_{jj}}{ n_{ij} + n_{ji} + n_{jj}}
\end{equation}
\subsection{Panoptic Metrics}

Panoptic Quality is used to evaluate the performance of the model in Panoptic Segmentation tasks.
\begin{align}
    PQ = \frac{\sum_{p,q \in TP} IoU(p,q)}{|TP| + \frac{1}{2}|FP| + \frac{1}{2}|FN|}
\end{align}
where
\begin{conditions}
\frac{\sum_{p,q \in TP} IoU(p,q)}{|TP|} & average IOU of matched segments\\
\frac{1}{2}|FP| + \frac{1}{2}|FN| & penalization of segments without matches
\end{conditions}
on the other hand, PQ can also be seen as segmentation quality(SQ) term and recognition quality(RQ)
\begin{align}
    PQ =  \underbrace{\frac{\sum_{p,q \in TP} IoU(p,q)}{|TP|}}_\text{segmentation quality}\times  \underbrace{\frac{|TP|}{|TP| + \frac{1}{2}|FP| + \frac{1}{2}|FN|}}_\text{recongnition quality}
\end{align}
SQ is the average IOU of matched segments, and RQ normally the F1 score. However, RQ and SQ are not independent since SQ is measured only on matched segments.