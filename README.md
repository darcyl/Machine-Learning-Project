# Machine-Learning-Project

<!DOCTYPE html>
<html lang="en-us">
  <head>
    <meta charset="UTF-8">
    <title>Machine-learning-project by darcyl</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" type="text/css" href="stylesheets/normalize.css" media="screen">
    <link href='http://fonts.googleapis.com/css?family=Open+Sans:400,700' rel='stylesheet' type='text/css'>
    <link rel="stylesheet" type="text/css" href="stylesheets/stylesheet.css" media="screen">
    <link rel="stylesheet" type="text/css" href="stylesheets/github-light.css" media="screen">
  </head>
  <body>
    <section class="page-header">
      <h1 class="project-name">Machine-learning-project</h1>
      <h2 class="project-tagline"></h2>
      <a href="https://github.com/darcyl/Machine-Learning-Project" class="btn">View on GitHub</a>
      <a href="https://github.com/darcyl/Machine-Learning-Project/zipball/master" class="btn">Download .zip</a>
      <a href="https://github.com/darcyl/Machine-Learning-Project/tarball/master" class="btn">Download .tar.gz</a>
    </section>

    <section class="main-content">
      <p>@@ -0,0 +1,253 @@
&lt;!DOCTYPE html&gt;</p>

<p></p>

<p></p>

<p>

</p>

<p></p>

<p></p>Machine Learning Project


code{white-space: pre;}

  pre:not([class]) {
    background-color: white;
  }



<p></p>

<p></p>


.main-container {
  max-width: 940px;
  margin-left: auto;
  margin-right: auto;
}
code {
  color: inherit;
  background-color: rgba(0, 0, 0, 0.04);
}
img { 
  max-width:100%; 
  height: auto; 
}


<div>


<div id="header">
<h1>
<a id="machine-learning-project" class="anchor" href="#machine-learning-project" aria-hidden="true"><span class="octicon octicon-link"></span></a>Machine Learning Project</h1>
<h4>
<a id="darcy-lewis" class="anchor" href="#darcy-lewis" aria-hidden="true"><span class="octicon octicon-link"></span></a><em>Darcy Lewis</em>
</h4>
<h4>
<a id="june-5-2015" class="anchor" href="#june-5-2015" aria-hidden="true"><span class="octicon octicon-link"></span></a><em>June 5, 2015</em>
</h4>
</div>

<div id="overview">
<h1>
<a id="overview" class="anchor" href="#overview" aria-hidden="true"><span class="octicon octicon-link"></span></a>Overview</h1>
<p>Objective of this analysis is to build a model which can predict the manner (A, B, C, D or E) in which the individuals performed bar bell lifts, given data recorded by accelerometers attached to the individuals.</p>
<p>The variable “classe”" in the data set correctly documents the manner in which the lift was performed.</p>
<p>Model Build Choices:</p>
<ol>
<li>Pre-Processing &amp; Predictor Elimination With 159 possible predictors, wanted to reduce the number of predictors to those most likely prior to attempting to find a model to predict the outcome. The steps to identifing good predictors took the field from 159 down to 14.<br>
</li>
</ol>
<ul>
<li>used only numeric fields; 56 predictors remaining<br>
</li>
<li>removed those columns which were contextual such as datetime stamps and row numbers; 52 predictors remaining<br>
</li>
<li>removed near zero columns (after replacing NAs with 0, replacing missing values with zero and centering and scaling the data); 26 predictors remaining</li>
<li>removed highly correlated predictors (&gt;75%); 24 predictors remaining</li>
<li>looked for predictors which were linear combinations of oneanother, but none were found<br>Note: centering and scaling was necessary as data valuess in various predictors ranged from fractions to values in the low hundreds</li>
</ul>
<ol start="2">
<li>Selecting a Model</li>
</ol>
<ul>
<li>linear regression was not an option as the outcome variable had 5 possible values<br>
</li>
<li>decision trees allow for more than two possible outcomes, but 40% accuracy just on the training data<br>
</li>
<li>random forest resulted in 100% accuracy on the training data and &lt;5% out of sample error rate.</li>
</ul>
<div id="setup-environment">
<h2>
<a id="1-setup-environment" class="anchor" href="#1-setup-environment" aria-hidden="true"><span class="octicon octicon-link"></span></a>1 Setup Environment</h2>
<pre><code>library(caret)</code></pre>
<pre><code>## Loading required package: lattice
## Loading required package: ggplot2</code></pre>
<pre><code>library(ggplot2)
library(rpart)</code></pre>
</div>

<div id="import-training-data">
<h2>
<a id="2-import-training-data" class="anchor" href="#2-import-training-data" aria-hidden="true"><span class="octicon octicon-link"></span></a>2 Import Training Data</h2>
<pre><code>setwd("C:/Users/Darcy/Documents/Coursera/Machine Learning")
#fileUrl&lt;-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
#download.file(fileUrl,destfile=".",method="curl")
pml_data&lt;-read.csv("pml-training.csv")</code></pre>
</div>

<div id="prepare-for-cross-validation">
<h2>
<a id="3-prepare-for-cross-validation" class="anchor" href="#3-prepare-for-cross-validation" aria-hidden="true"><span class="octicon octicon-link"></span></a>3 Prepare for Cross Validation</h2>
<pre><code>##Partition the training data into training and test (80/20 ratio) data sets 
set.seed(1234)
inTrain&lt;-createDataPartition(y=pml_data$classe,p=0.8,list=FALSE)
training&lt;-pml_data[inTrain,]
testing&lt;-pml_data[-inTrain,]</code></pre>
</div>

<div id="pre-processing-predictor-elimination">
<h2>
<a id="4-pre-processing--predictor-elimination" class="anchor" href="#4-pre-processing--predictor-elimination" aria-hidden="true"><span class="octicon octicon-link"></span></a>4 Pre-Processing &amp; Predictor Elimination</h2>
<pre><code>##create data frame of just the potential numeric predictors
NumericPredictors &lt;- sapply(training, is.numeric)
trainingPredictors&lt;-training[ , NumericPredictors]

##Remove the 1st 4 columns containing contextual data
trainingPredictors &lt;- trainingPredictors[,5:56]    

##replace NA values with 0
trainingPredictors[is.na(trainingPredictors)]&lt;-0

##center and scale the data
preProcValues &lt;- preProcess(trainingPredictors, method = c("center", "scale"))
trainingNorm &lt;- predict(preProcValues, trainingPredictors)

#remove near zero columns from the training dataset after replacing NA values with 0 to create dataframe of NonZero (NZ) predictors
nzv &lt;- nearZeroVar(trainingNorm)
trainingNZ &lt;- trainingNorm[, -nzv]

##remove highly correlated predictors
CorTrainingMatrix&lt;-cor(trainingNZ)
highlyCorPredictors &lt;- findCorrelation(CorTrainingMatrix, cutoff = 3/4)
trainingRelevant &lt;- trainingNZ[,-highlyCorPredictors]

##append outcome variable back into final dataset to be modeled, i.e. trainingClean
classe&lt;-training$classe
trainingClean&lt;-cbind(trainingRelevant,classe)</code></pre>
</div>

<div id="create-an-accurate-predictive-model">
<h2>
<a id="create-an-accurate-predictive-model" class="anchor" href="#create-an-accurate-predictive-model" aria-hidden="true"><span class="octicon octicon-link"></span></a>Create an Accurate Predictive Model</h2>
<pre><code>##After trying several models, random forest gave by far the best results
modelFit&lt;-train(classe~.,method="rf",data=trainingClean,prox=TRUE)</code></pre>
<pre><code>## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.</code></pre>
<pre><code>confusionMatrix(trainingClean$classe,predict(modelFit,trainingClean))</code></pre>
<pre><code>## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4464    0    0    0    0
##          B    0 3038    0    0    0
##          C    0    0 2738    0    0
##          D    0    0    0 2573    0
##          E    0    0    0    0 2886
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9998, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc &gt; NIR] : &lt; 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000</code></pre>
</div>

<div id="pre-process-and-apply-the-model-to-test-partition">
<h2>
<a id="pre-process-and-apply-the-model-to-test-partition" class="anchor" href="#pre-process-and-apply-the-model-to-test-partition" aria-hidden="true"><span class="octicon octicon-link"></span></a>Pre-process and Apply the Model to Test Partition</h2>
<pre><code>##preprocess testing data using training data
testingPredictors&lt;-subset(testing,select=colnames(trainingPredictors))
testingPredictors[is.na(testingPredictors)]&lt;-0
testingNorm &lt;- predict(preProcValues, testingPredictors)
classe&lt;-testing$classe
testingNorm&lt;-cbind(testingNorm,classe)
testingClean&lt;-subset(testingNorm,select=colnames(trainingClean))
##apply the model to the testing partion
confusionMatrix(testingClean$classe,predict(modelFit,testingClean))</code></pre>
<pre><code>## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1078   12   14   11    1
##          B   13  712   18   12    4
##          C    7   32  631   12    2
##          D    4    5   19  614    1
##          E    0    9   10    6  696
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9511          
##                  95% CI : (0.9438, 0.9576)
##     No Information Rate : 0.2809          
##     P-Value [Acc &gt; NIR] : &lt; 2.2e-16       
##                                           
##                   Kappa : 0.9381          
##  Mcnemar's Test P-Value : 0.003946        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9782   0.9247   0.9118   0.9374   0.9886
## Specificity            0.9865   0.9851   0.9836   0.9911   0.9922
## Pos Pred Value         0.9659   0.9381   0.9225   0.9549   0.9653
## Neg Pred Value         0.9914   0.9817   0.9812   0.9875   0.9975
## Prevalence             0.2809   0.1963   0.1764   0.1670   0.1795
## Detection Rate         0.2748   0.1815   0.1608   0.1565   0.1774
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9824   0.9549   0.9477   0.9643   0.9904</code></pre>
</div>

<p></p>
</div>

<p></p>
</div>







<p>
</p>

      <footer class="site-footer">
        <span class="site-footer-owner"><a href="https://github.com/darcyl/Machine-Learning-Project">Machine-learning-project</a> is maintained by <a href="https://github.com/darcyl">darcyl</a>.</span>

        <span class="site-footer-credits">This page was generated by <a href="https://pages.github.com">GitHub Pages</a> using the <a href="https://github.com/jasonlong/cayman-theme">Cayman theme</a> by <a href="https://twitter.com/jasonlong">Jason Long</a>.</span>
      </footer>

    </section>

  
  </body>
</html>
