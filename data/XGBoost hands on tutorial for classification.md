# XGBoost hands on tutorial for classification

[00:00:00] Hello and welcome to this video on XG Boost. Here we'll use XG Boost hands-on using psychic Learn in Visual Studio Code. We will work with the same heart dataset as we used in the random forest ~~video, the hands-on ~~video. Here we'll use XG Boost and we'll try also be Ian Optimization.

There's a little bit new things going on here. ~~In ~~instead of psychic Learn, we'll use the XG Boost package. This will be quite interesting. Let's move on to the code directly.

**Kokchun Giang-1:** here I am in Visual Studio Code, and let's start with setting up our environment. ~~In, ~~if you are in Mac, M1, M two, M 3M, four M, et cetera~~ we, ~~you need to install something called Lib comp. ~~This, ~~you need to do brew install lib comp. For me it's already done. ~~I've already done that. ~~It will work otherwise it won't work for you.

Then we will need to install a few packages because now we are not ~~in ~~using just Psyche Learn anymore. Instead we'll [00:01:00] use XT Boost package. Uuv V and V and then source that then being activate. Then we'll do UVP install XT boost Pandas. I need psych kit. Learn to do train test split.

I need mat lib. I need let's see. I need psych kit dash optimiz because I'll use the Ian optimization to show you how that works. Then I need ipi turnoff to work with Jupyter Notebook. Okay, that was quick. Now I have my heart data set. Let's start with create xg boost at I pin B.

Okay. XG Boost. Then I will remove this one and change this to Python. Okay. Import pandas as PD. Then we'll do df equals pd.read csv and we'll read this [00:02:00] Heart dataset let's say heart Do CSV index call equals zero. ~~It, ~~this is quite similar to what we've done before ~~in in what's called ~~in the random forest videos in the F head.

Okay. To see the data set now.

Okay. You can see this is the dataset and then we'll do DF dummies. This is also exactly like the video. P get dummies df drop first equals two, and then A XY equals two. DF, D. Drop HDS axis equals one, and then comma. ~~D, f dummy y a HD. Yes. ~~This is our target variable,

and then, ~~we'll, ~~I'll just copy this from the lecture note ~~for extra ~~for train test split. But we need to import the train test split. From SK Learn dot. Model selection import ~~TTT ~~train test bit and then I'll run this one. [00:03:00] Okay so this means that we have imported we divide the data set.

We have a trained test bit, and now it's time for our XG boost. We'll see how that will work. XG Boost and for XG boost algorithm, ~~we'll need, ~~we need to do something called a pipeline. I'll show you what that is right now. Pipe equals the pipeline. ~~You need to have the pipeline imported.~~

~~From xg, boost, import pi, let's see, from F. We need to have a psychic learn, actually. ~~From SK Learn pipeline, import pipeline. As from grid search video before Pipeline, you can put in several steps there and here we will put in XG Boost as one step. Pipe equals to pipeline of steps equals to a list of topples and our classifier.

~~We don't need to have ~~since XG boost and ~~three ~~based models don't need to scale the data. We don't put in a scaler, so we ~~do XTB dot x we need to import XTB as well. Import xg boost as xg B ~~do xg B classifier. Here I'll have random [00:04:00] state equals 42. This is my pipe. Cool.

As that we have XGB classifier, and here are different parameters that are by default that looks like this. A lot of none as you've seen. Not so many are set, but now we'll need to some hyper parameter tuning. ~~Hyper permitted tuning. Okay. ~~To do hyper pyramid tuning.

Let's see, remove that one. We'll need some from SK op. This is the psychic optimizer that we installed before import all the patient c such cv. This is a patient search algorithm to find different hyper parameters. From, so this is a more random, randomized approach. Compared to grid search, which was more the exhaustive searching. You can use grid search if you want, but I choose to use BA optimization here. SK op [00:05:00] space import real comma integer, and then we'll do a search space. These are the parameters that we'll search. This is a dictionary.

We'll start with cla. The name is the Classifier, classifier. Then we'll have max debt. Max debt integer two comma eight.

**Kokchun Giang-2:** the classifier max debt, it means that it's the tree debt. The higher, the more depth of the tree, and that means that we get more complex trees, which means that it's easier to find more complex structures, however. This will also lead to easier over fitting. That is a trade off here for max debt.

Then we have classifier let's see, learning rate. This is the learning rate, so how fast it learns and when it comes to trees. It's about, how much each [00:06:00] tree contributes to the ensemble. As in XG Boost and in random forest, we have several trees, in XG Boost, we are basically creating trees that will improve upon the last tree.

Attack the place where it has been weakest and to minimize the error. Then the learning rate is like, how much does each tree contribute to, to the fitting. It means that lower values, it becomes more robust because it learns so each tree contributes less, but then we need more trees, we need more depth.

That's the idea. Classifier learning rate, we start with real. It's a real value between 0.001, so it's very slow. Then dot 2.1, so 0.1. Then we have a prior it's we give it the log. Dash uniform. This is just what it should be as a prior, and then then it will search for this in this space.[00:07:00] 

Okay. Then we have classifier, CLF. Let's see. Subsample and subsample. What does this mean? It's a fraction of samples to use for the fitting. It means basically row sampling. You have your tablet data and then you cut out the rows, you pick out a piece or in your rows, and then you use this for the fitting.

This is to reduce over fitting as well, so we have real. Let's see real 0.5 comma one, and then we'll move on to classifier. Call sample by tree. What does this mean? This means it's a fraction of features, so of columns to use for each tree. Remember, we have several trees and then ~~we can, ~~we don't use all the columns in each tree.

Instead, it should be a fraction of it, so real. 0.5 to one. These are just [00:08:00] quite good~~ ba ~~baseline values that you can use when you're tuning the hyper parameters~~ and ~~and then you let the patient such do its work for you. Okay. Then let's move on. We have classifier, cold sample by level.

By level means that there is a fraction of feature used for each level of the tree. ~~For ~~for each level of the tree, we use a certain amount of features. We don't use all of them at once. It's also real 0.5 comm one. Okay. Then let's move on to classifier call, sample by node.

Here is that then you take a fraction of feature per split. Each node, it's a split, so here ~~you have you don't use all, ~~you don't consider all the samples either or all the. Features either instead. It's [00:09:00] just a few features. It's similar here, real 0.51. Now it's time for some regularization.

The regularization we have is CL fco rig alpha. Rig alpha is basically L one regularization. Then we have, ridge regularization. Then L two is the lesser one. These are also to penalize the weights or ~~pen, yeah, ~~penalize the weight. Here is the real between zero comma 10. Then let's see, classifier, reg lambda.

We have real also zero 10. Then we have, so these are L one, L two, and then we have Classifier Gamma. This is basically it's a minimum loss reduction that is required to make a new split. Here we have real of OMA 10. Okay. This is the search space that we'll use. Then what we do [00:10:00] is that we take optimizer.

Opt optimizer equals to base such cv, and then we put in our pipe. We put in our safe space, we take CV is the cross validation. We can pick it as five, for example, and iteration let's see, 20. The more you pick ~~the more ~~the the longer time it takes for to train, but maybe more robust. Scoring equals common scoring to use.

Usually. When it comes to more what's called more unbalanced data sets you use R-O-C-A-U-C. Actually this, you can use accuracy here since this quite balanced dataset. But this is a common more robust one to use. F1 is also used usually. For so random state, 42, then we have our optimizer.

Okay. ~~Okay, ~~so optimizer, what is this? It's this you can see it's a [00:11:00] pipeline of this XT B classifier with these parameters. Right now it hasn't set this parameters because you need to fit it in order to set those so.fit x train. X let's say Y train. Remember that we don't scale the data because it's a tree based model.

Here ~~we ~~we train it and yes, it takes a little bit of time, I think maybe 20, 30 seconds. Oh, okay. It's done. Yes. Then you see there's a lot of parameters that it found. Here are the parameters. You can see this by level, by node, by tree, et cetera, that it found different parameters. The learning rate is different except the different let's see max debt is free, et cetera.

Okay it has, found different parameters to use using the optimizer at the search space by using searching. Now we can do optimizer.best [00:12:00] estimator you can see here is the estimators. It is the same as before. Then let's do here to find the different scores you can do optimizer best.

Let's see. Or actually, I don't need to show that. I will show. I will do y bread actually. Optimizer dot predict X test. Then we have our Y bread is a lot of zeros and one. We can also do optimizer dot predict probability on X test. Then you can see here's the probability probability score, and then you can do colon five and you can see that yes, it has predicted different classes,

yes or no on a HD. Then [00:13:00] finally we do some evaluation. Here is let's see, training and then we're here in prediction. After prediction, we're in evaluation. For evaluation, it's the standards. From SK learn do metrics, import classification report. Confusion matrix. Confusion matrix display.

Then print classification report y test Y pre, and here you go. This is these are the results that we get. We have cm equals to Confucian matrix Y test, Y pre, and then we do Confucian matrix display on cm. Then display labels equals to ~~no comma. Yes. ~~No comma Yes. Then dot plot like this.

This is the confusion matrix that we have here. This is the final results. Yes, it might be somewhat [00:14:00] weaker than random forest if I remember correctly. However, quite close I think. We can also take a look into optimizer best. Estimator steps. Here you can see here's the classifier, and here is the ex.

The different steps that we have in our optimizer and in the pipeline we see we had the classifier, and here are. A lot of different values and we'll use these to pick out, let's see if I take out, oh. Zero is the first one, it's the top and then I pick out the one, and here you can see the parameters.

This is the XG Boost model. XG boost model equals to this one, and we can actually use. This for plotting the importance. We can [00:15:00] actually do this from XD boost, import plot importance, they have a convenience function for doing this because this is quite useful that we do. You don't need to do what we did before in random forest video.

XD boost model, we just plot it like this and then we can do PT grid. False. What do we need to import from import lib pipe, PLT. Okay, great. Here you can see that. Okay. Here H is the most h cole and cholesterol and ca is, are the most important features according to this model. This were a little bit different from the random forest model, I think the ca was most important at in that model.

in this video we have gone through XG Boost. This is a quite cool model and in XG Boost there's a lot of hyper parameters to tune and we tune them using BA optimization. This was quite a little bit different from [00:16:00] using normal Psyche Learn only, but we still use Psyche, learn for.

Creating the pipeline. We used it for ~~a mo ~~model selection. Train test split. Then finally we did the prediction and evaluation using psyche learn as normal. Then we used XD boost to plot the feature importances.

I hope that this was helpful for you. If you haven't seen my auto in random forest and decision trees, ~~take a look in, ~~take a look at them then I hope that you learned a lot. See you in the next video.

Thank you. Bye.

