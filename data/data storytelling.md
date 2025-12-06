# data storytelling

[00:00:00] Hello and welcome to this lecture where we'll go into, data storytelling. So data storytelling is something that, is very important when you want to use your visuals, your data visualization for presentations, and when you're presenting it to, business stakeholders in particular, because, ~~when ~~when you are, 

making a lot of visualizations for yourself and for the data team. Maybe you are exploring the data, and this is, called exploratory data analysis, and this is very important to understand the data set that you're working with. However, I've seen a lot of people using the visualizations from exploratory data analysis and showing it directly to business stakeholders, which makes it a little bit.

Difficult for them to understand and difficult for them ~~to, un ~~to, know what you're talking about. And, gets, really ~~easy ~~confusing, especially when you have worked with, the data, in very long time while, the [00:01:00] ones you're presenting to maybe see this for the first time. And if you're showing them all your graphs from exploratory data analysis, then.

Probably they will, not understand all of them. So that's why we will go into something called explanatory data analysis. And ~~in, ~~in order to do explanatory data analysis, we need something called data storytelling to make it, ~~to become ~~more efficient. So we move into the keynote now, and I will show you what I mean.

So explaining your visualizations to business stakeholders using data storytelling. ~~Oh. It's about telling a story. So remember story. Okay, ~~a simplified data journey in data analysis and visualization. I'll show you a typical workflow.

Not, all workflows look like this, but this is one of, the workflows and it's simplified. So bear in mind. So we start with ~~rare ~~raw data. So you could get it from like CSV file, Excel file, you could get it from, the database, et cetera, but you [00:02:00] have some kind of data. And you need to clean it up, and you clean it up based on both, business requirements and also, data formats, requirements, et cetera.

~~So that you need for doing some types of transformations. ~~So you have transform data. And when you have transformed data, you visualize it. So ~~you visualize it for yourself. ~~You're doing different types of exploratory data analysis, also called EDA. Data scientist, data analyst usually talk a lot about ~~eeds.~~

~~So they're doing exploratory data analysis. ~~So you can ~~do ~~make correlations, you can make different distributions, you can, graph different variables against each other to understand the data. And to find different type of patterns. And then you create dashboards usually for, ~~four ~~different type of stakeholders, so that they can click around, on the dashboard and maybe to different other teams, or the only for the data teams, et cetera.

So they can click around. And it can also be, that you generate the BI report, like for example, saints Report once per week or once per month, et [00:03:00] cetera. 

And then based on this, we can use this for creating something called explanatory data analysis. So in explanatory data analysis, you pick out some visuals that you want and then you make a presentation. And in this, lecture here, in this video, we'll focus on the explanatory part. So this is, the ones that we're focusing on and the presentation.

These are first, all of these are shown, visible for the business stakeholders. All others are not. Remember that, and we will focus on explanatory data analysis. So when we have exploratory data analysis, we will go in, from exploratory to explanatory data analysis. And what does this mean? We start with EDA exploratory data analysis to understand our data.

And then we create visuals. So we have a, for example, scatter plot here we have some line [00:04:00] graph, and then we have, ~~yeah, ~~some kind of distribution. So we have a lot of visualizations, right? ~~So ~~1, 2, 3, blah, blah, blah to N. ~~Okay. ~~And then based on these, we'll pick out some relevant visualizations. So what are relevant?

That depends on what type of, explanations we, what type of, story we want to create, what type of, visualizations that are relevant for the target audience. So note that the target audience is different depending on, ~~different. ~~Different situations and depending on the target audience, we pick out the visual, right?

So maybe we pick out visual two and visual three and they look like this for example. And based on this, we create, we go through several steps of data storytelling. And when we have the data storytelling, we. Get something called explanatory data analysis, which we can proudly put into our presentations, for example, PowerPoint or [00:05:00] Keynote, and then we can present it to our target audience.

Okay, so this is the step from ~~X ~~exploratory data analysis here, down to explanatory data analysis. Note that there's difference here.

So exploratory data analysis, they're good for finding insights. They're bad for presenting to business stakeholders. I have unfortunately seen this, like this one cars sold in Norway. ~~I. They quite, they are okay for exploratory data analysis. You can find oh, okay. ~~Most of the, car brands that are sold are a few most that, are sold.

There's a lot of brands that sell very few. ~~Okay, good. ~~But ~~this will, ~~you should never present this for a business stakeholder. And you have this one yearly car sales in Norway. ~~Yes, ~~maybe it looks, quite okay, you might think, but we will improve on this one.

Okay, so what can we do? We [00:06:00] can remove cluster to simplify visualizations and improve readability. Okay, so what is clutter? Let's start with this graph. ~~We will, yeah I didn't say before, but we'll ~~focus on this visual and the other visual that is, quite hard to work with. I will give it for you as an exercise to clean it up and, I will, give the data a link to the data in the description so that you can find the raw data and try to, recreate that visual and clean it up.

Clutter are elements that don't add information and complicates the visuals. So for example, here we have a legend. ~~Okay. ~~You might think legend might, add information. Yeah, maybe if we have several lines here. Quantity, if you wanted to show quantity or you could have an arrow towards it, or you could have quantity in the axis.

So it's redundant [00:07:00] information, but if you have several lines, ~~that is ~~the legend is also cluttering because, the thing is. First you need to, when you're looking into a legend, you need to move your eye to the left and you need to move back to the right. And that is disturbing. You want to have something called ~~a pro, some, ~~proximity.

So that you want to have, for example, an arrow and a label to annotate that this correspond to this label here. So legend. I will consider it as clutter here. Grid lines clutter. You might think okay, we need grid lines to track exactly where it is. But that could also be, managed by, you could put, numbers close to where the data is ~~instead of having.~~

Instead of having to trace to the left, but that is a design that you could choose. ~~I haven't chosen anything in this in this lecture, but you could do that. I was showing on the lecture that I, where I have done it. So ~~here, spines? Yes. The right spine and the top spine, I would say clutter.

They [00:08:00] are ~~in they're ~~quite hard. So you could see they're quite, they're solid and black, so they're kinda standing out. Why they want to have the visual to standing out, the trend. Not the spines, so that one can be removed immediately.

So we also have ~~like ~~here, three zeros. They could be replaced by a prefix K for thousands, right? So you can write 150 K, 140 K, 130 K, et cetera. So we don't need to write out all the zeros because, ~~that one, ~~it could be simplified a little bit. So with this, I would say that, we have removed some clutters.

There might be some more things that you might think about. But for now that's fine. So let's see how the result will be. So we have removed the visual clutter in this line graph. So we start with this line graph. It'll become like [00:09:00] this.

So now I've removed some clutter by removing grid, we have removed the legend. We removed the right spine and the top spine. ~~Okay, so I. ~~And we have changed, ~~one, ~~for thousands we have changed to K, the prefix. ~~K. Okay. ~~So what else can we do? We could use colors and contrast to focus the audience attention.

So moving from this graph here, let's use colors sparingly and contrast. So what that means is that we can. Make this part here thicker and, clearer color. ~~You could change the color, but ~~I will actually use the same blue here. We can change the colors of, ~~these parts, ~~the x axis and y axis, to make them, into lower contrast, make them into gray.

~~So ~~this will make, ~~the visual ~~the actual line graph here. ~~It will become ~~[00:10:00] more, focused so that you can see it more clearly. Let's see what I mean. Like this, ~~so ~~you can see the visual is much clearer. What we've done is thicker line,

lower contrast on spines axis, and thick labels. So you can see here it's much lower contrast while ~~the gra, ~~the line here is much thicker. So this becomes much clearer than this one.

Okay. What else can we do? We can use a descriptive title to tell a story. ~~So ~~note that before we had yearly car sales in Norway can change it like this Number of cars sold in Norway have been increasing steadily after the dip in the financial crisis. So this one is much more descriptive title, and it has a story behind it.

It says that the number of cars sold in Norway [00:11:00] have increased. So we have descriptive title. We have also, we have positioned the title and labels, and I've also changed the labels. So we have a number of cars sold here and years since 2007. So it becomes a little bit clearer and when I positioned it, like this, so left the line here and, towards the top here, it becomes more clear in my opinion.

Now we have a descriptive title so we can also annotate to clarify or lift up different parts of the visuals. ~~So this is what we have, we put in an annotation. ~~Here

is.

Low sales due to automotive industry crisis.

So that's why we have lower sales here. So we have annotated this one. Okay, so this could be like finished, but there's some more things that could be done.

For example, here, we could use dash [00:12:00] lines with dot. To indicate that there's no data in between. No data between the dots. We could do like this. So this means that like we have a trend, but the data points are each year. So we don't have data in between. That's why we use dashed lines. And you can see here's the annotation.

With this, I would like to say that I'm finished with this graph. ~~We could make it a little bit different for, there, ~~there's a lot of different variations which you can make depending on the story.

So what have we done? We've applied data storytelling to turn exploratory data analysis into explanatory data analysis. We have remote clutter. We have used color, sparingly, leveraged, contrast [00:13:00] focused. The attention used, descriptive labels.

So these are a few things that we have done to turn ~~this, ~~exploratory data analysis into explanatory data analysis, ~~which we could be, ~~which we could use ~~to, ~~our, reports for presentations. We could put this into a slide. And ~~we could ~~explain this to our audience.~~ So ~~this video has been, focused on, working with, data storytelling, where we turned the, exploratory data analysis into explanatory data analysis, ~~u ~~using several different steps. ~~And ~~I hope that you can use these steps for your own visualizations. ~~And ~~the tool that I have been using, ~~used ~~for these ~~are, ~~simply lib in Python.

However, you could use whatever tool you want. This is perfectly viable using, for example, Excel or PowerPoint or Keynote or similar tool. But [00:14:00] there, it needs some customizations to make it. ~~But ~~however, when you know the theory, what to customize, then you just learn the tool that you want to use ~~to, ~~to make these visuals.

I use Python and ~~Matt Plotly ~~because, I'm more familiar with. Code, but, that is not, necessarily the case for you. However, if you want to learn more about how I make these visuals, I have other videos where I will go into Matplotlib in depth, ~~in this ~~in, this visualization course.

And you'll see me working step by creating, visuals like this. ~~And also I will, ~~I encourage you to work with the other visuals, and, try to make it, ~~into ~~into explanatory, data analysis so that you can, get, good visuals that you can show off and present for others. And also if you want to learn more, I really.

Strongly recommend this book. Let's see, like this. So data, storytelling, [00:15:00] storytelling with data from cold news, bomber, lic. So this is where I've learned all about, storytelling ~~and, ~~the thing is I've applied the knowledge that she, taught, into, these, map lib graphs and, also other, python graphs.

But, she has done it all with, Excel, to show that, the tool doesn't matter. With this, I would like to thank you for watching this video and see you in the next one. Bye.

