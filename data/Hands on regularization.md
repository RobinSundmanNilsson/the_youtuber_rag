# Hands on regularization

[00:00:00] Hello and welcome to this video where ~~we've come into regularization and ~~we've come into the hands on the practical part of the regularization. This is to make the model less complex that we can get less over fitting. Remember that we've learned in the theory session about bias variance trade off.

Also we've learned about the regularization and we've learned about how to decrease the complexity using regularized models such as L one, L two, and elastic net, which is a combination of both of them. And then we talked about the K four cross validation in the theory as well.

We'll go through all this in code 

**Kokchun Giang-4:** now I am inside of the resources section of stat learning.com. The book that I referred to as ISLR. Introduction to Statistical Learning. Here in the resources session, if we go down here, we have the data sets and we have the advertising dot csv. Download this one advertising [00:01:00] csv, and we'll use it for our ~~regular sa re ~~regularization examples.

**Kokchun Giang-5:** Now I'm in Visual Studio Code and let's start with the terminal here and let's create a virtual environment. ~~UV, VNV source. ~~Then being activate u let's see. Now I've activated my virtual environment and VP install, we will need psychic learn. Of course we will need pandas and, I think that is it.

Maybe I need a mat lib or something. I think it's fine. We'll start with this one and IPAC kernel, of course. We need that for a Jupiter notebook. And then I will paste in my data. 

**Kokchun Giang-6:** now I've pasted in the data, the advertising csv. Now let's just close this terminal and create a new Jupiter notebook. Regularized linear models dot IPD, like this. And then we create a new markdown. Let's [00:02:00] see. We start with here regularized linear models, and we choose our virtual environment.

Let's choose this one like this and change this to Python. Okay, now we will run import mpi SNP, and then import pandas as PD. ~~And this is, ~~and we need to have DF equals to pd.read csv. ~~We'll read this. ~~Advertising, do csv let's see. Index call equals zero, and then DF equals to this one. Df, we can take DF head actually to not show all of it.

. Dear Head, run it again. ~~And ~~as we have worked with this in several videos before that we have three features and one labels here. We will. Take out all the features and put [00:03:00] it into an x capital X, which is the design, the feature matrix, or the design matrix, ~~and where we'll have all the features ~~and we'll have lowercase y for our label.

To do this, we'll do XY equals to df drop. And what do we want to drop on? We want to drop on sales capital letter and then let's see. Axis equals to one, and then we have df. ~~Of let's see. ~~And this one will give us the X and then comma we do topple unpacking here DF sales like this.

This in the right hand side becomes a topple. And here left hand side we do topple unpacking. X becomes this one and Y becomes this one. Okay, now we have X and Y. If I take a look into XX head, you can see that this is the X, without the sales. And then Y head, and you can [00:04:00] see that this is Y and this is just a se.

This is a data frame here, and a y is a series. Okay? And then using this one, we will ~~do some we will do, ~~use the polynomial features as we've learned ~~in ~~in the previous lectures about polynomial regression. We'll use polynomial features in order ~~to ~~to enlarge the feature space to make the model more complex.

~~That ~~maybe if we just use linear regression right away, ~~we will lose ~~we will lose some complex behaviors and~~ and maybe the, ~~we need to increase the complexity. We'll start with making the model more complex. What we do is from SK learn dot pre-processing ~~import. ~~Import polynomial features.

We'll have ~~model ~~polynomial, ~~poly, ~~poly equals to polynomial features of three. We'll ~~make it into, we have X one. We ~~have x, x squared and X cube. And then we'll do include bias equals [00:05:00] to. ~~Falls ~~like this. We have po. Then we create poly features. ~~Our features, ~~we do feature engineering now.

We do model poly.fit transform of x. We transform the X ~~that ~~the whole feature space becomes~~ where we get it into ~~polynomial features. I'll show you how that looks like. Remember that we had the X that looks like this, and now with poly features, it looks like this. We have 200 comma 19.

The shape is 200 comma 19. What does. That mean? It means that we have 200 samples. That's the row. And 19 are the features. From three features, we now have 19 features. And that is possible because we have all the different combinations. ~~We have x ~~we have for example, first we have tv.

We have TV squared, we have TV Cube. We have tv, [00:06:00] radio as one. We have tv. Tv radio, newspaper as one TV squared, radio as one, et cetera. ~~And this easily adds up to ~~and when we do all the combinations, exhaust, all of them, we come up to 19 features. You can see this is a rather complex model.

Let's see. We can do the shape here. And we see 200 comma 19. Okay, with this, we want to split the data now. We want to do train test split. From SK learn dot model selection, import. Train test split. And then I will do help train test split. And then I will go ~~and go ahead ~~and steal this in the text editor steal this one here that the, make sure to type them in the right order.

Okay,

Now we have X train, Y train and ~~x train, ~~X test y train y test. [00:07:00] Let's do this. And we want to do in not doing on x instead on poly features because we have done the ~~polynomial ~~polynomial feature engineering. We create more artificial features based on the features that already existed.

Okay? The model is more we have X train. We can take a look at extreme dot shape and it's 134 comma 19 and we can take a look at why train dot test as well. Do the shape and you can see it's 134. Okay what this means now is that ~~we, ~~we have our. Our training data, we have our test data, and we can move on to standardization.

We need to do feature standardization. Remember in theory that it, ~~for all the, ~~for the ~~re ~~regularization we require. The model to be scaled properly. We do feature standardization [00:08:00] and feature standardization. Let me just copy from my lecture note the equation you can look at it.

The equation is like this. You have. X is the matrix with our features. ~~And ~~and X prime is the standardized feature matrix or the design matrix. We have X minus mu, mu is the~~ mean ~~sample mean, and then divided by the sigma is the ~~standard. Is the standard deviation ~~sample.

Standard deviation. And by doing this, we will ~~get we'll get it in, we'll get, ~~transform the data into zero mean and one in standard deviation. That is the idea. ~~From, to make, ~~to do this in code. Let me see. From SK Learn, oh, not SK image. SK learn dot pre-processing import standard scaler, and then we do scaler equals to standard scaler, and we do scaled X train.[00:09:00] 

Equals to scaler, fit, transform, not that we can do fit transform. It's a fit and transform on the X strain. However, on the X test scaled X test, what we do is. S scaler dot transform only X test. And this is very important because ~~we, ~~when you do fit, transform, it means that we do fit on the X train and then we do transform on the X train.

While we, then we have the parameters calculated. And the parameters are sigma and mu. They're calculated from fitting the X train and using this. Parameters, we can calculate the transform the X test, we don't want to take the parameters directly from X test because that would be data leakage.

We want to use the parameters ~~in ~~in the training data, and then use that for transforming the test data. Okay, how far have you come? Now we can check, take a look at [00:10:00] scaled X test dot mean, for example, we can see that yeah, this is scale, this is~~ not ~~not zero. ~~And, oh, ~~not mean actually.

Mean that is what I mean. Okay. Mean is not zero because this is for the testing data. But s scale, the X. Strain dot mean this should be zero. . This is definitely zero. It's you can see that it's some rounding error. We have to the power of minus 17. It's very small.

We can see it as zero and scaled x strain dot D for standard deviation. ~~And ~~you can see it's one, and we can do scale X test, td. ~~And ~~you can see that's not one. It's close to one. Not really one. ~~Here we, ~~and this is what we expect. It's totally correct. And now we will go into our regularization techniques.

We'll start with reach regression

and rich regression is L two. Here we have the penalty time, what [00:11:00] we want to do is we want to. This will cause this will cause a lot of parameters that will be close to zero. Let's test that out. From SK learn dot linear model, import reach from SK Learn dot. Metrics. ~~We want to take a look at the metrics.~~

Import ~~means mean ~~mean squared error, and we can take a look at mean absolute error as well. Okay, now we'll create a, I will create a function now, deaf ridge regression, we'll put in the x. We'll put in the penalty equals to zero. We'll start there. And then model ridge equals to ridge [00:12:00] alpha.

Alpha is the ~~penalty. ~~Penalty. This is how in psychic learn they use Alpha~~ to ~~to denote the penalty. We place in the penalty, it starts with zero, and then we will test it out with different penalties. Okay. And then we have Model Ridge dot. Fit scale, X train, Y train, and then we do y red equals model ridge dot predict on our X.

We want to predict on our x,

Okay to get our Y spread. Y spread equals to model reach, do predict X, and then we return our Y bread. Return y pre like this and now we can use this. Y pre [00:13:00] equals to reach regression and note that we need to put in our X, scale X test, this is what we want to place in. We want to place in zero to make it predict with penalty zero or we didn't need to do it if we had penalty equal zero like that.

But it's fine to be. More clear. MSE mean squared error of YY tests. Comma y pre, let's see, Y pre. Then we have M, SE, we have RMSE equals to NP square root of MSE, ~~and we have. ~~And we can take a look into these ones. Let me check. MSE and RMSE. Let's see. Actually, we will take a look into MAA as well.

MAE is mean. Absolute error of y [00:14:00] test y spread like this. MAE. Okay, now you have three of them. ~~Let me see this. ~~

**Kokchun Giang-8:** I want to rewrite this into a function that we can reuse this metrics later on. In order to do that, let's do def metrics, and then I want to put in white test comma y print. This is just a wrapper function. To make to tab, tab those in, and then we return. And I want to return a dictionary, MSE of MSE MAE of MAE and RMSE for RMSE.

Like this and then remove that one. And we can do just metrics of y test Y spread.

**Kokchun Giang-9:** Now let's show you what happens [00:15:00] when we just take linear regression. Let's see here from sk learn dot linear model, import linear regression, ~~and then model model linear. Actually I can just do model. ~~Model is linear regression. And we can have model fit scaled X train, Y train, and then we have Y red equals to model predict scaled X test.

And then we do metrics of Y test, Y bread. Like this. And what you can see is that they're quite similar. They're actually the same. ~~There. Yeah, ~~there can, ~~could ~~be some rounding differences here, but ~~mostly ~~mostly it's the same. And the thing is that why is this? Because [00:16:00] linear regression or reach regression with penalty equals to zero is same as linear regression.

This is important to know because we have set the penalty term to zero. That is what I wanted to show, and then we ~~go ~~move on to lasso regression.

And for lasso regression, we have from SK Learn. Let's see. From sk learn, do linear model, import, lasso, and then we can do model equals to lasso. And here we have alpha equals to 0.1. We can choose one alpha and then we can do ~~lasso. Let's see, ~~model fifth scale X ~~strain, comma. ~~Y train and then we have ~~y red ~~y spread equals ~~the ~~model [00:17:00] dot predict scaled X test.

And then we do metrics of Y bread y test. Let's see. I had, oh, why test first? Okay. Why bread? Okay. And now it's a little bit different. You can see that this differs from what we had before. We can see that in this case, the linear regression, it worked better than the ~~lesser ~~regression, at least for this testing dataset.

But it could be that we had luck in the testing data set or like it favored the ~~linear model, more ~~linear regression model more than the lasso regression. That could be the case. We're not sure. What we want to do is that we want to do a K fold. Cross validation in order to make it [00:18:00] more robust.

What it does is that we shuffle the dataset and then we split it into K groups for K fold. And then for each group we take one test and ~~one ~~the rest is training. And then we fit the model ~~on the. ~~On the training and predict on the test, we get an evaluation metric, and then we take the mean of the evaluation metrics and we choose the parameters, and then we train on the entire training data set and we can use this ~~for eva ~~for predicting on new data.

That is the K fold cross validation. In order to do this in practice, we have K fold. ~~K fold, cross validation. ~~Cross validation let me change this to Python like this. From SK Learn dot linear model, import Ridge cv. We have model let's see, model Ridge CV [00:19:00] equals to ridge cv and we have different alphas to try out.

We have, for example, ~~0.001, we have ~~0.001 0.01, 0.1 0.51. Five and 10. We try out different types of alphas. Here you can see they, they have alphas here, and you can see that what does this alpha mean? You can read more about it and see what the suitable values are for alphas.

Here we'll do scoring equals two. Scoring. I'll choose here. Negative, mean squared, error. 

**Kokchun Giang-10:** ~~ the. ~~The reason for choosing negative means squared error is that in psychic learn ~~it, it ~~it tries to optimize by ~~maximizing the ~~maximizing the [00:20:00] score. However, as we know that ~~lower MSC, ~~lower means squared error is better, we want to maximize the negative means squared error.

That's why we have negative means squared error here. And also for the different alphas, it means that it will try different alpha values. For 0, 0, 0 1, 0 0 1, et cetera until it finds the one and it'll report the one ~~which ~~which was the best one. And~~ the. ~~This alpha, what it does is that it penalizes large coefficients.

When the coefficients or the parameters that we have trained become ~~when they're ~~large, it will become penalized. And the higher the alpha value the more it penalizes~~ this this, ~~these parameters. ~~And ~~and what that means is that we get less variance and higher bias, which means that we get the simpler model.

That is the core idea of this. Okay, with this, let's continue [00:21:00] with ~~we do. ~~Model Ridge cv, fit, scaled X train, Y train. And then ~~we can print, ~~we can take a look at the Model Ridge CV ~~do ~~Alpha to see how it looks like, it shows 0.1 as the best one out of all this. Okay. And then we can do y pre equals model~~ rich ~~CV dot predict on scaled x test, and then we can take a look into the metrics.

Y test, Y pre and this is the best metric that we got. Or this was the best from the ridge. And we can take a look into the model coefficients to see that. Model ridge cv, cof. And you can see that the coefficients some of them are. They become [00:22:00] smaller but they are not zero compared to, let's see we, we will show you with less how it looks like.

For lesser regression, less regression.

Here actually some of them, like this one has become close to zero, and this one is also close to zero. And this one's also close to zero, a few of them are close to zero. And let's take a look into lasso. From SK Learn linear. Model import lasso cv. ~~Sk actually like this SK learn. I forgot. Right here. ~~SK learn lasso cv. Model lasso O CV equals to less o cv. And here. We have we can take N Alphas to take a look into, for example, 100 here. And then it'll [00:23:00] try 100 different alphas along a special regularization path.

And then we can choose the cv. We can choose CV equals five, for example. We could choose CV up over there also as how many CV we want. Here we want to five to show you some of the parameters that we can take. And then we can also take max iteration to some value, for example, 10,000.

It tries 10,000 iterations and maximum and in case, because it's like taking, it's taking gradient descent. ~~It ~~it should stop somewhere in case it doesn't find the minimum. Here we take model lasso cv fit scaled X. Train y train and then we can take a look into model lasso cv dot alpha to take a look into those ones.

And here, [00:24:00] okay, it says N Alphas was deprecated in 1.7 and will be removed. Alphas now ~~accept, ~~accepts integer. Instead of n alphas, we have alphas like this. Okay. Then it found this one as the best alpha, 0.0 0 49. Note that this is a little bit different approach than before, before we chose different alphas to put in.

And here we let it choose for us. What you can do differently, like for example, if you choose this approach then you can start with a large range like this and it says okay, 0.1, then we can ~~narrow it down ~~narrow it down between 0.05, for example, and zero point 15 and then and then going more steps there to refine the such.

But here it finds this one for us, and then we can test it out. We can do y bread [00:25:00] equals to model lasso cv y, test y let's say y test scale x test. And then we take a look into the metrics Y test y print. Let's see, let's, CV is not defined. Let's see. Model. Let's cv we need to take dot Predict.

Okay. And here we get the scoring.

Okay. And ~~take a look into the. ~~Take a look into the different, model lasso, CV coefficient. And you can see how it looks like and ~~what it's, ~~what you can see clearly here is that several of them are set to zero. Lasso we make them into zero because they're not worth et all. According to this model.

These the other coefficients are those that are used in different extents depending on how large the values [00:26:00] are. Now the model becomes simpler and. And let's take a look into the last one, which is a combination of both lasso and ridge. It's called elastic net.

Elastic net Regression. Let's see, like this. Okay, from scale learn, linear model import, elastic net cv, and we can take a look into model elastic equals to elastic net cv. And here let me see. We have L one ratio. An L one ratio. Let's pick a few. One 0.5, zero one. 0.7, ~~0.9 ~~0.9 5.99.

Comma one for example. And let's take a look into, we can have N [00:27:00] Alphas it's Alphas of course they changed it. Alphas 100. It picks different alphas for us and ~~we can have ~~we can have max iteration as well as 10,000. It and then we take a look into model Elastic dot. Fit scaled X train, Y train and we take a look into model elastic dot L one ratio.

Let's say L one ratio. And you can see it's one. It means that this choses only L one and with only L one. ~~The, ~~it means that it will remove the ridge regression and only choose the lasso. Okay? It only chooses lasso. We can take a look into Model Elastic dot~~ alpha ~~alpha. And it chooses this one, which is the same alpha as the one we found ~~from ~~from Lasso,

what [00:28:00] it thinks based on the testing, on the training data it finds the best is it's the best one is lasso regression ~~with ~~with this alpha. Based on the parameters we used. Okay. And then after this one, what can we do more? We can do y bread equals the model elastic.

Do predict. We can predict on our Scaled X test. And then we take a look into the metrics of Y test from a Y bread. And this is the same as the one from Lasso. And we can also take a look into the Coefficients model Elastic coefficient,

and you can see is the same as the lasso one. Okay, based on this, you can see [00:29:00] which one we should pick. For example, we can take a look into RMSE. 0 57 8 5 is the one that we found with model Elastic. Let Elastic Net Lasso is the same as Elastic Net. And we have a, let's see, the ridge, what we did, we find there 0 56, it's somewhat lower.

It means that the ridge was somewhat better on this testing data. And let's look into linear regression here, linear regression. And reach with zero is linear regression. The simplest model it seemed to perform the best ~~for, ~~for this dataset after we had done feature engineering, however, you should also compare this result with the multiple linear ~~regression without not polynomial ~~regression and to see which one that performed the best.

And for example, say that this one performed the best. This polynomial regression. Then we pick this one and we train it [00:30:00] on the whole data set. Both the the X train and the Y train. The whole x we scale it and we train it on it. . 

In this video we have gone through using linear regression and polynomial regression and then regularized~~ the mo ~~the models. It means that we have~~ de ~~decreased the complexity and may increase the bias, decreasing the variance that the model becomes simpler. First we ~~created the ma made, ~~did ~~the polynomial, ~~what's called feature engineering using polynomial features that we get a more complex and then when we did linear ~~mo ~~regression on it means that we get polynomial regression, which was quite complex. And then ~~we ~~we tried to make it simpler ~~using, ~~using penalization, using rich regression, using lesser regression.

And then a combination of both with elastic net and then showing, ~~showing d ~~how to use cross validations to pick the models that you want. ~~And ~~and then afterwards~~ when, ~~when you have picked the model. This is the data science work. This is. [00:31:00] It was a quite a simple tutorial that didn't go through all the ~~met all ~~methodologies that existed, but some of them, ~~and ~~and then afterwards when you have picked the model that you want, you train on all your data and then this model can be deployed ~~and ~~and used for production for example.

~~And . , ~~And I hope that you've learned a lot from this video and see you in the next one. Bye.

