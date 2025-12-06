# Data platform course structure

[00:00:00] Hello and welcome to this lecture. We'll come into how this data platform course is structured and in this you will get to know the technology stack and you will get to know a little bit about the study techniques and how the course is structured in the GitHub repository. So everything is inside the GitHub repository where you'll find all the information.

So now we'll move on to the keynote.

So inside of the keynote here we have how this data platform cost is structured. Let's move on. So everything can be found in the course GitHub repository. So here is the main content in the GitHub repository. And you can see that ~~each ~~each folder or each directory has some content. ~~It will have a content on on ~~it will have a readme file.

It will have code. So all the materials is here and ~~in inside of the ~~inside of the read me, there's also video lectures. So there's links to video lectures and they [00:01:00] usually look like this. And if you see this video here, you can see there's a code alongs video which means that I will code live ~~and~~ or I will code inside of the video and you have the possibility to follow along.

So you should pause ~~and~~ and code it out, type it out, test it out and see that you understand and understand the code and make sure that You don't just copy the code from the GitHub repository or take a copy code directly from LLMs, et cetera. So typing out, you'll train your muscle memory.

And also we have exercises. We have several exercises and exercises includes practical ones. They include theoretical ones, and also they include glossary. So ~~I I suggest, ~~I highly recommend that you do the exercises and train so that you get some practice with this course.

Yes. And moving on to the technology stack, this is ~~quite, ~~quite a crowded course. I will say it includes a [00:02:00] lot of things. ~~Yes. ~~Yes. So first of all, we have Python and we have ~~UV~~ the ENV so that this is the virtual environment that we'll be working with and we'll connect it to Kafka. So actually the first two weeks are dedicated for Python repetition and pandas repetition in case you don't have enough foundations within Python and pandas. So this is very important that you make sure that you have some solid foundations. And then we'll come into Docker. This is really interesting to containerize our apps in order to get the isolated environments. So we'll come back to that later on.

And in this we'll work with images and containers. And we'll have several containers, actually multi container apps using Docker compose. This is really interesting. And in Docker, we have volumes to map to the local volumes as well. And we will map the ports as well. So right now in this course, we [00:03:00] will work locally ~~on the~~ on your local machine.

But however when you have Dockerized everything. It's very easy to just move this into the cloud so you could deploy this solution and it will work in the cloud. So that is quite cool. And also if you give this Docker containers to your colleagues, it will work on their computer as well. We'll come into Kafka for streaming data. So we'll ask in Kafka, we'll talk about the producers, consumers topics. We'll also come into some theoretical aspects about how it's working with how can it scale, how can it take care ~~of ~~of the data and the streaming data, but we will actually go into, we will use Kafka in together with Docker.

So we'll use a local Kafka environment instead of using the cloud. So usually Kafka is very popular in the cloud as well, and using [00:04:00] Confluent, for example, But we will work with it locally so that we train how to set everything up locally first before moving on to the cloud in later courses in your data engineering journey.

So it's good to understand how it works locally. And then we'll have Postgres as second. It's called, so it's a sync from Kafka so that you have the data stream into Kafka and then it's going into Postgres ~~to ~~to store it. So Postgres SQL is very common and very very popular database engine~~ which~~ which use a relational.

Tables. So it's a relational database and you will use it to connect to Kafka. So it's a sync. And then we'll also have MongoDB for no SQL so that we see how it works with connecting to no SQL as well, or no SQL. So MongoDB is no SQL. And then we'll serve this real. So real data is used for.

For dashboard, it's quite cool and quite [00:05:00] easy to set up dashboards that can update with certain frequency. So we start realtime dashboard and we have something called bi IS code because real data instead of~~ un ~~unlike many other data dashboard applications that~~ use ~~usually~~ you, ~~you drag and drop.

Here we have bi s code codes of business intelligence as code which means that we will write SQL code actually~~ to ~~to show different graphs. And ~~the, ~~one of the good aspect with this is that we can version control it. Another one is actually we can leverage ai.

Something that's rising up is AI within bi I actually forgot the term, but you, ~~you use, ~~basically use some kind of, lLM to help you create code that can directly be used to generate graphs. So ~~that, that~~ that is possible ~~using~~ using BIS code. And then we'll have GitHub, of course, and Git.

So we will version control through Git. [00:06:00] We'll work with git branches, but ~~this is~~ this is for the ones enrolled in the course. So if you're just taking this online, you won't be able to ~~you won't~~ participate in the projects of course. And we'll have a GitHub project for Kanban and ~~we'll have, ~~we'll work with the issues.

So we'll set up issues and work with GitHub projects and branches and the pull requests, et cetera, as in a real team. And also one thing to note is that the project is only available for those enrolled in this course and not ~~for~~ for those ~~taking~~ taking this course online.

As it looks right now we'll work with the bash of course, and here we'll have bash CLI for ~~term ~~terminals. Very important because we won't be having so many graphical user interfaces when we're going into containers, et cetera. And ~~we will have~~ if you're in windows, you'll also work with the WSL and WSL is windows Subsystem for Linux this is a requirement in order for us to work with Docker.

So then you have Linux [00:07:00] inside of ~~in in ~~Windows. And of course, we'll be in Visual Studio Code. This is the ~~IDE~~ Integrated Development Environment that we'll work with. You can of course use whatever ID you want. And now moving on to some study technique tips. ~~So ~~we have overview.

Usually you~~ usually ~~take a top down approach. ~~So we get you, ~~it's important to get some kind of overview. So ~~you have, ~~you look into the glossary in the exercises. And then you can watch the videos pause and code along. ~~So ~~type out the code. Don't copy paste. This has to do with ~~the ~~muscle memory.

The more you type it, the more you will learn how to do it. So I always type out the code. Even if I see some code snippets~~ I copy it. ~~I Type it out. Unless it's something super repetitive. I will copy it and then watch several times rewind rewatch. And then when you come to class, discuss with your classmates, they are your learning resources.

They will help you to grow and you will [00:08:00] help them to grow. And very soon they will become your colleagues. ~~They might help you finding jobs in the future. ~~You might help them finding jobs in the future and bring question to class, bring question to the teachers, bring question to your students and then you have the practical exercises.

Go back to the video and rewatch it in case there's something you're missing when you're doing the exercises and check with LLM for answers, check with LLM for different tips, etc. And check the resources, learn more. And then resources scheme through them. There's usually many resources and you may not have the time to go through them all.

Make sure to just scheme through some resources and that you find interesting and you think will fit with this course. So then we'll come to the next lecture. And in the next lecture, this process will start again. Okay. So thank you for watching this video and I hope ~~that you've learned some study technique tips that will ~~that you've learned some study technique [00:09:00] tips.

~~So thank you and see in the next video where we'll come into ~~see you in the next video where we'll come into Python repetition. See you. Bye.

