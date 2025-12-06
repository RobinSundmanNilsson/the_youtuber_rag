# postgres sink

[00:00:00] Hello and welcome to this video where we'll go into Postgres Sync and not only Postgres Sync, but we will actually go into getting data from an API from a REST API and we will loop it so that we will get the data in a certain frequency. So this is actually called polling data. So we're doing polling and this will give us streaming, but then we will, ~~we'll ~~poll it in with a certain frequency ~~if you want to stream based on when the event when it has changed then you're using other techniques which we'll come back to later on however so we'll pick the data ~~we will ingest the data from an api ~~into ~~Using a producer to publish events into our Kafka topics, and then the Kafka topics will also make sure that it is persistent, so they will make a ~~bind ~~bound, which we haven't done before, and then when it has been persisted into the topics, it means that ~~when ~~whenever we're closing it down with docker compose down and then we're opening up again, docker compose up then the events are still in [00:01:00] the topics.

And after that ~~we will do ~~we will consume this data. Consuming using a ~~consumer ~~consumer script and for the Kafka we will use a quick streams as we have done before which is very easy very good client for working with Kafka. And then. So we have now consumed the data.

We will do a sync. So that means we will store it into somewhere. And this somewhere is the Postgres, ~~so ~~a Postgres SQL, which will spin up in, it's another container. We have a multi app container here. ~~We have~~ we have four containers actually. So we have broker, we have Postgres. Postgres SQL, we also have control center and we have the schema registry.

So with this I would like to go directly to Visual Studio Code to spin up our containers starting with that. So see you in Visual Studio Code. So now I'm in Visual Studio Code. So let's do like this. I will do terminal, new terminal. And [00:02:00] here Postgres sync. Yes. So let's start with creating ~~a u, uvvnv. So I create ~~a virtual environment because I wouldn't use that. So bin vn, bin, activate. And then we will install some packages.

So ~~uvpip ~~install. What do I need? I need ipy. Ipy. Kernel because I want to go into Jupyter notebook to do some EDA on the actual API, and then we'll need pandas to work with pandas and we'll need the ~~quick streams. And the quick streams, we will actually need ~~quick streams and Postgres like this.

Let's see. I think it's like that actually. So what do you have? pip install ipykernel pandas quick streams Postgres. Do we need something? We need requests to get the data. So starting with this one these few things and we also need to have ~~e nv, so UVP installed ~~item nv, like that.

Okay, cool. So that with this, I'm finished with installation. I think there might be some other things. [00:03:00] However, let's do docker ps now to see, we don't have anything. So docker ps a, we don't have anything. So let's create the patch. Let's see, docker compose. yaml. And then I will open this one, docker compose.

yaml. And basically I'll just copy it from my lecture notes. ~~So you can just get it from there as well. So I'll copy it and I've already gone through most of it. ~~So here we have the broker and the new thing here is that I think we haven't gone through it before but basically I have ~~a ~~volumes here called dot slash Kafka data, and then I have slash temp slash craft dash combined log.

So this is the bind mount where I will bind mount where we have the logs from our Kafka so that we can actually get the get storing the offsets so that we store the data basically. So it will create a folder here called Kafka data. So when it creates a folder called Kafka data, this will generate a lot [00:04:00] of, files for Git to track if you're working with a GitHub repository.

~~Our git repository, which I truly recommend. So let's do that. ~~Let's do touch. gitignore. And probably you already have a gitignore if you're working with a github repository or git repository. But we have gitignore and I will ~~add ~~make sure I'll add this one ~~craft not craft ~~Kafka underscore data so that this folder here that is created, that one will be ignored.

Okay. Then we have schema registry and we have control center and then we have Postgres and in the Postgres is the same as in the last time. So we have a lot of environment files, right? And talking about the environment files, we need that as well. So code dot ENV. And I'll save this one.

And in here, you need to~~ paste ~~type in your environment variables. So ~~I will type that in actually I will give you, ~~I will type in this and also this one, and then we have the password as well. [00:05:00] So you're super safe. And also we will, you will also in our environment put in our API key for working with the API, as I said, but coming to that later on, let's start with this one.

And remember to change this Postgres password to your own. ~~And I'm also we'll go into the API. ~~So actually I will work with coin market.

So this is for getting ~~cryptocurrency so ~~cryptocurrency data, let's go into the browser here. ~~So coin market cap. ~~So if you go into this coin market cap create an account and login. ~~So yeah, exactly. Login here. ~~So when you have created an account and you have logged in, you can for the free account, the basic plan.

~~There, ~~there are some APIs which are allowed to use ~~it ~~and some that you're not allowed to use ~~it. ~~For me you can see that ~~I have a few, like ~~we have 10, 000 credits 10, 000 calls ~~per~~ per month. ~~And yeah, this is how many credits I've used ~~and this is how many credits I've used this month.

And these are [00:06:00] basically just for testing out. And here you can see my API key. Actually, it's not for you to see. So you should create your own and copy that one. Okay, moving back to Visual Studio Code. Then you have CoinMarket API and then you copy in your API key. So I will pause the video and I will do that. 

Great. So now I have copied in my API key. And~~ and now just ~~now let's spin up our container. So Docker compose up dash D. Let's see. Yeah, it needs to ~~pull a little bit thing. A little bit things. ~~Pull some images. ~~I think. ~~Yeah. Postgres pull. All other images I've already had. So this goes very quickly for you ~~may, ~~it may take some time for you to ~~in~~ download the images.

So if I do Docker PS, you can see here, we have four containers. Actually I'll do this so that you can see all of them. So here are them. They are control center, schema registry, Postgres and broker. So let's create [00:07:00] let's do make dir src for source and make dir explorations. And then I will do touch eda, let's see, coins, eda.

ipinb. Let's see, it should be in src. So I create basically I create or not in exar actually, I wanted it in explorations. Then move with the explorations.

So this coins, EDA, ~~I want to I want to go into the ~~I want to do some explorations ~~of ~~of the API. Before we actually~~ spin ~~build our pipeline. So we should understand how this API works first. So going into the browser now let's see if I go here, we can see coin ~~document ~~API documentation.

Perfect. And when going into let's see, ~~going into there should be, yeah, there should be here. ~~Cryptocurrency. [00:08:00] Yes, here you can see some interesting things. Here are some endpoints. Actually ~~we will ~~we will use latest. Actually ~~there, ~~there's also ~~V, ~~V2, but I have just V1. So let's do V1.

And you can see here in here, you can see the end point is available on the following API plans, and we actually have basic if you're not paying for it, so it says the update frequency and plan credits. ~~So we will actually go down here. And see ~~here are the parameters that you can read about query pair parameters and here you have some A.

P. I. Documentation. Let me zoom out a little bit and basically it looks like this. You have latest and you have data and here are the I. D. S. So you can get the name. The first one is Bitcoin and then you can see going on to the second one is actually a theorem ~~going on to the Yeah, ~~they only had two in this case, but it should be able to filter on which [00:09:00] cryptocurrency you want.

So I'll show you how to do that later on in this lecture. ~~And also there should be some code example, let me see. Let me see, code examples.~~

~~Categories, endpoints, let me go back here. Airdrop, let's see, airdrops, no. E1 latest. ~~If I search on Python, yeah, exactly here. I just searched for it and you can see here are some languages as we are working with Python. We go in here and this is the code that you want. So I'll just copy this one. And then I will go into Visual Studio Code.

~~and I will do like this. ~~Cryptocurrency. Let's see. Cryptocurrency coin market, EDA. Let's see. And the kernel, we choose our virtual environment. I paste this one in, and this should be in Python. Okay, so you can see the URL here. And here is an API key, and this is a fake API key. So if I just run this one, you can see there's a lot of Data, but these are fake data [00:10:00] based on this fake API key.

Okay, so how do I get the real API key? I do that through my dnv. So ~~from low~~ from. dnv import low. dnv. And then I will do low. dnv and I will also do import OS. And OS, let's see, then we'll go into the headers, you can see that this is the API key, right? So we'll just replace this one with what we have.

OS. getenv, and what is my API key called? I called it coinmarket. API or CoinMarketAPI key or CoinMarketAPI. I forgot, but let's try this out. Circulating. Yeah, it's still not working. Could it be API key? Status API key missing. Okay. [00:11:00] API, let me just to make sure what I got. Yeah. I call it coin market. API.

Does this work we have status, we have slug. It's weird. It's not the same. Let me restart this one. So I have my do d nv Here, let me run it again. Okay. Let me pause the video and come back. 

Okay, I've got it. It's the URL that needs to be changed. That this is the sandbox API URL. So just remove this ~~one. So here it's it's the quotes and it's the latest ~~one. And actually for the quotes, you need to change a little bit of the parameters. You cannot have these instead we will ~~have we ~~have symbol and what symbol do we have?

We have for example, the target symbol. And the target symbol is basically just a variable, ~~which you can~~ which you can set when you're putting this into a function later on, which is very good to do in case ~~you want to you want to change ~~you want to reuse this one. Of course. So for example, BTC, so that we can get [00:12:00] Bitcoin.

This is the symbol of Bitcoin. Let's see. ~~We have a target symbol. We have, ~~we need to put in the currency. What currency do we want? ~~So convert~~ we want to convert it into USD ~~US dollars. Let's see, US dollar like that. ~~And it takes the API keys always get in ~~the open coin ~~like this.

Actually we can clear it up a little bit. We can do like this, API key. And then we can place an API key here. ~~So these are ~~so headers and parameters needs to be placed into this. So session is used for opening up a session ~~of ~~to get the connection and~~ and having having the and having ~~using the session, we can get the response.

So we put in the URL, we put in the parameters and we have. Here we update the headers with headers. So I update it with these and then we get back the data basically. So if I run this one, I should get data. Yes, I get. So I see symbol BTC and then there's a lot of other things that is quite hard to understand, but we shouldn't print it out like that.

We can actually do data [00:13:00] like this so that we can get it out a little bit nicer. And I would also do the data dot keys.

Okay, so you have data, you got the keys, it's btc here and then you can continue going into the btc and you see all the keys here and also you can go into checking out like here's the name of the BTC and then you get Bitcoin and BTC data. ~~Let's see something else symbol. And we can get B, T, C.~~

Let's continue to write, , BTC. Let's see what we have here. ~~Sl, what is there? ~~Bitcoin. Okay. And let's see. Continue and date added. Let's see what that is. Okay. 2010. Yeah, that kind of makes sense. BTC data of let's see, quote is interesting, but that actually quote I'll put in the next one.

Let's [00:14:00] see like this. Okay. I will go into BTC data and I will go into quotes. Here, this is the latest price that we've got and yeah, and we can see here the latest price is this much, insanely much and then we have okay, you can see the percentage change in 24 hours, how many percent change, okay, it's actually quite volatile if it says minus 13 percent.

That's amazing. And then we have percent change one hour is this much percent change 24 hours, or actually this was volume change, ~~not~~ not percent change. Sorry, I couldn't read. Percent change minus 3. 36%. That is also quite much et cetera. But I don't know, I don't know the cryptocurrency market, but it seems to go up and down a lot.

Okay. So we have a BTC [00:15:00] data quote. So now you can see that, okay. If you're inside of the quote, you can go into USD and you can start to get the latest prices. So here you have the quote. And inside the quote, you can go into USD and you can see if you go into price here, you can get the latest price.

Cool. And also there's something else that is really cool to do, ~~but I haven't I won't show you here, but ~~you can actually start to explore other things. For example, you can go into finding all the listings and there's a lot of cryptocurrencies. Actually, you can get 5, 000 of them. ~~From this list you can maybe even get more.~~

I'm not sure. And also there is some cryptocurrency, some other end points that is worth to investigate and explore, but I will leave it to you so that I won't take out all the fun I'll just do BTC and interesting. ~~And I can show you ETH later. It doesn't matter. So I will close this down.~~

~~I'll close this down. I'll close this down. ~~So clear the terminal. Let's create our pipeline now. This is the interesting part according to me. [00:16:00] So ~~let's do like this. ~~We create coin consumer. py. We create coin producer. py. So we need to publish events from the API. And when we publish it, we want to. Store it in Kafka, right?

And then we want to consume it from Kafka topics, and we want to sync it into Postgres. That is the goal of this lecture. And also I will create something called constants, which I usually use when I have several constants that I want to reuse. ~~So this one, I'll just copy in because it's not very interesting.~~

So basically it's a OS, it's env and then a load. env and then here ~~we get~~ we get the coin market API, we get all this for the connection~~ in, ~~into Postgres, right? Because I will reuse this in several places [00:17:00] it's good to I separate into constants. So this gives me the freedom to just import it.

So also to not make the other modules bloat. So not make spaghetti code, not make them too large. Instead, divide into smaller modules and import stuffs. That is what I want to do. Okay. So going into the producer, let's create this one first. This one is quite simple. So import time. We need to do use time in order to do sleep frequency.

And then I will do from quick streams import application. Of course, that one we need. And then from constants. What do I need from constants? I want to import a coin market API and then from requests import session. ~~Remember that we needed this one in. From the we needed that one from you saw in the documentation, right?~~

So I'll just, I just copied it from the documentation import. Jason, we need this one also load it and then we can do like this app because the application [00:18:00] and then broker address. What do we have? We still have the same as before 1992. And then we have consumer group. What is our consumer group?

It's a coin group. You just give it a name and then you use the same one when you're consuming it. So it's called coin groups for us. And then we create the topics. Coins topic equals the app. topic. ~~And what do what does it call then? ~~It's called coins. And then we need to, we have the data in JSON, right?

So we need to serialize JSON data. Then I will create a function. I'll create like this. I'll go into my coins EDA and I will steal this code here, basically. API key, that one will import, so I will take this part, yeah, this part I'll copy. [00:19:00] And then I will do that, get latest coin data, symbol equals to BTC, and then I will paste this code in.

~~Let's see. Yeah, this two guys need to go in, this guy go in, and then I do black formatting. Yes, here. ~~Okay, so you can see target symbol is basically symbol API key. Let's see, ~~we go from, ~~we go into constants coin market API. So coin market, let's see API key, we can do like this API key. ~~Equals coin market API.~~

Actually, I would place it inside here. So coin market API goes into here. We have coin market, latest parameters. We get the data, right? And here we can basically just return this one. If it's successful, otherwise we have an exception. Then we need timeout and too many redirects. ~~Let's see.~~

Request too many redirects and we'll have a timeout like this. Yeah, basically like that. [00:20:00] So if do the Python preamble, if dunder name equals to dunder main, let's do get latest coin data. And~~ let's ~~Then ~~we have ~~we get the JSON, right? So result equals to that one. And I will do pretty print of my result.

And I need to import pretty print. So from pprint import pprint. Okay, like this. And then I will run this one. And you can see we get the data. Perfect. ~~Okay, so what do we get data? We get data like that. ~~Okay, then we need to go a little bit deeper here. Let's see. We have JSON. loads of this one. Then basically just pick out the data.

Getdata. And if I run this one again, we can see ~~we go. ~~We actually get into btc and then we can do get of the symbol, right? The symbol that we [00:21:00] put in is btc. If I run this one again, then we should go down into actually like this. Perfect! Now we're inside of it. Bitcoin, because you can see that the name is Bitcoin.

However, if I change this one to, for example, ETH, and I run this one, you can see now we're into Ethereum, right? ~~Okay, but I will keep it as VPC. Okay, ~~so I have now done a request, ~~and ~~a GET request, and it works. We got the data back. So let's now do the interesting part. ~~So now we have data from our API. ~~So now it's time for us to actually do some looping.

So def main and I will do like this with app dot get producer as producer. And you have seen this one before. And here now, while true, I want to run this forever, ever and ever. Okay and what [00:22:00] do I want to do? I want to do this coin latest equals to get latest coin data BTC.

And then we will create our Kafka message. Kafka message is just coins topic dot. serialize. And what I want to serialize the key is equals to coin latest of symbol. So now I create a symbol and~~ or I create, ~~I use the BTC then as the key here. And then the value is equals to coin latest.

So basically, I just pick everything and serialize it into binary and place it into my Gatopic. And the key is symbol, right? So the key is BTC in this case. And now I will just do a print here, [00:23:00] produce event with key equals to Kafka message and value equals to, and then I will just pick for example, the value not the whole value, actually, that is quite hard to, I will call with price price, let's do like this coin latest of quote, USD.

And then we go into the conversion and we go into the price like this. Okay, but we haven't converted yet, remember? We, or we haven't actually produced yet. So we do producer dot produce topic equals to coins. Topic dot name. ~~So the topic is remember we created a topic that was called Bitcoin.~~

~~And let me see, for example, or the name is coins actually. ~~So coins topic dot name. We have the coins. ~~We~~ we, Place it into this topic and then we have [00:24:00] a key e equals Kafka message dot p value equals to Kafka message dot value. So basically we produce this one and let's do time. sleeve 10 seconds. We let it produce a few.

Okay. Then we just comment this out and run the main function. And if I'm running this one, you can see that something is happening. Okay. It's produced event and we can see quick streams to model Kafka message. Okay. Not really correct. Dot e, let's see, Kafka message. Yeah, this is not really correct, actually.

Control C. Let me see why. We get the symbol. We get let me see. We get the symbol here. [00:25:00] And we get the symbol.

Oh, because print is not printed the key. Sorry. Now I print the key. Yeah, no, key equals BTC and price equals this much. And then in 10 seconds we get the next value.

Yeah, exactly, you can see the next value and it's the same because no one has bought yet or it hasn't updated. Probably it hasn't updated. But you can see after a while that it has changed. Yeah, now it changed. Cool. You can see that someone has bought or sold or both actually. Yeah, I'm not good at investment.

Okay. So now we have created our producer. So let's, let's consume this one. So go into consumer. I'll let it run for a while we're writing out the consumer. So inside of the consumer, let's do like this from QuickStreams [00:26:00] import application from QuickStreams dot things dot community dot.

Postgres Scale import Postgre S Scale Sync from Constant Import. What do I need here? I need to have Postgre db name Postgres, host Postgres password Postgres, port Postgres user like this, and now. Let's do like that for great creating a sync, we need to do or actually we can create a main first main and in the main, we create our application.

So app because the application broker address equals to local host. That's before 1992 comma consumer group consumer group equals to. [00:27:00] Coin group, and then we need to auto offset reset equals to earliest. Okay, and that means we'll pick all the data. Okay, so coins topic. Coins topic equals app.

~~topic. And then we have the name. What name did we have? We have coins, right? And then we need to deserialize the data. Remember, because we have serialized it, we need to get it back. ~~So now it's time to create our streaming data frame. So STF equals app. dot data frame of our topic equals to coins topic, right?

So now we have a streaming data frame. So what to do now? This is the interesting part. So stf equals stf dot apply. We need to apply a function. And I will call this function extract coin data. And then we'll create this function. def extract coin data. Basically what we need to put in is a message.[00:28:00] 

I will do pass for now actually I will do control C here or I will comment this one so that I will use it soon, but I will do stf. update update lambda message print message because we want to print it out in order to Pick out stuffs, right? So that we can know what to pick out. So I will leave this producer on.

It will be here. I don't care. I will let it collect some data. So if dunder name ~~equals to dunder main ~~equals dunder main. Let's see, I'll run main.

So then this will print out the message. So I run this one. Let's see what this says. Dicop G2 is missing. Run pip install quickstream Postgres. Yeah, I think I did that. [00:29:00] Let's do that again. uv pip install quickstreams.

Postgresql, like this. Ah, I need to have quotes here. Okay, now. So now I installed this one. Let's run it again. Yes, it worked. ~~However, let's see another main. name echo standard main main. ~~Yeah, we haven't run the app yet. Yeah, exactly. We need to do like this app dot run. Okay, perfect. Now I can see this is the data that we've got back.

And there's a lot of data here that was printed, right? So we need to go into in order for me to get some valuable data. What do I want to have data? I want to have the coin. Let's see if let's do pprint instead. From pprint, import pprint, ctrl c. I'll run this one again. [00:30:00] Oh yeah, now it has to wait for new messages to come.

Okay, because it already have processed.

~~So let's wait a little while. Let me see if the other one is running. Ah, it has been closed. Okay let me see. ls python src slash coin producer. Okay, so now I let this one run and I also have this one on. Okay, so now we've got it. We got the name. We can get extract the name here. Let's see. Then we do like this.~~

So latest. Quote equals to message of quote

and in the quote, we have USD, right? So quote USD and we get, we can get the latest quote and we will return now a dictionary of the coin is basically what is the coin? It's. Call, it's the message of name. So now we get the Bitcoin, right? This is the coin price. USD USD. This one is latest quote of price.

And then we have volume. This is latest quote of [00:31:00] volume 24 hour. Yeah, volume. We can't have volume 24 actually. We can have updated. When is this updated? That is also interesting. So message of last updated. So you can find it here. All of them. Yeah. Okay. So these are some coin data and of course, there's a lot more.

So to do for you is extract more relevant data. ~~Okay. So I will. So now I will extract a lot of data. So ~~if I close this down and I do SDF, I will do this extract data, and I will actually move this printing message down here. Pprint message pprint yeah, and this is the message after we have cleaned it.

So transform data. So if I, let me see if this one is still running. Yes. Actually, I need to do like this Python. src coin [00:32:00] consumer like this. Okay, now you can see it's much more beautiful and it's cleaner. Yes, this is the data that I've picked out. Now I've done this apply function to my SDF and I picked this data out.

But, we're not done yet. We want to place this into Postgres. That is our main goal. The only thing we need to do is this. It's so simple. It's ridiculous. So we have here. After we have done our apply, we do like this. stf, or we have postgres stf. sync And we placed in our sync. postgres sync And what is our postgres sync?

Postgres sync equals to [00:33:00] Create post rest sync and we need to implement this function that's, that is, it's create Postgres sync. And this is quite simple. I'll just, I'll copy it because it's just, you don't need to watch me type it out.

So you have Postgres SQL Sting, and then you have the host, the port, the DB name, the user, the password, what the table name should be and schema update. Actually, you don't need this one, I think. So these are all you need. That is it. So now the data will flow into. Postgres. So [00:34:00] let's try this one.

Close this one now. So I do stf. sync postgres. Python src coin consumer. py. Okay, I will let it run a little bit. I'll go into another terminal. Docker exec bash id postgres. Let's see postgres and then bash. I'm inside of my postgres container. P S Q L, Dash U, Postgres Dash D, CoinsDB, I'm inside of CoinsDB, backslash D T, I see Bitcoin table, select star from Bitcoin, ta da, it's there, four rows, we wait a little while, run ~~it again, I'm gross when it again, I rose.~~

Yeah we have we in just data in a frequency of 10 seconds. So just wait a little while and we'll get more data. [00:35:00] And as for now, it hasn't changed, but that will be interesting to see if it changes select star again, now we can see it has changed. Oh, cool. And now we have it in Postgres and Yes actually we want to try one more thing.

I will close this down Control D, and I'll do Docker Compose. Yeah, I will, I can close this one down, Producer and Consumer, Docker Compose down. So I closed down all my containers,

Docker Compose up dash d. I start them up again.

And if I'm I'll run the consumer. Yes, it doesn't consume anything because it's [00:36:00] waiting for messages. I'll run the producer. It produces. And I will do Postgres. And here psql u postgres d. Let's see, what did I call it? CoinsDB backslash DT. It's here. Select star from Bitcoin 13. You see it continues from where we were before.

And that is super cool. So the data has persisted and also the Kafka has persisted. We can show that as well. Create a new terminal. Let's see broker. ~~It's called right? ~~So docker exec dash it broker bash. Okay, [00:37:00] inside here we do let's see, dash consumer dash console, dash consumer dash bootstrap server, local host 1992.

Topic points from beginning and you should have more than 13 messages. So we cannot see how many messages we've got, right? But if I do Ctrl B now, or Ctrl C actually, processed 74 messages. So it has processed all those messages. And actually some messages I've had before as well.

Let me see. Coinbase. Yeah, it's hard to see actually, but [00:38:00] yeah, Bitcoin here. Data added.

Yeah, so I've run this before, so it has some previous messages. But no matter, you can see that the data has, it has persisted And that is a really cool thing. Because we have done some, we have done a bind mount for the Kafka data. We have done the vault. We have done a name volume for Postgres.

So you can see the data is persistent. ~~So I will just clean up. So Docker compose down. Let me see. Or I need to jump out first. Docker compose down. Yes. I stop all this. Yes. ~~ Okay, so you have now watched the whole pipeline ~~from go ~~from getting the data from the API, producing the data into and publishing it into a topic, consuming, reading the data from the topic, syncing it into Postgres, and then read it out. [00:39:00] So this is a pipeline for streaming data. How cool is that?

Okay. With this, I would like to thank you for watching this video and see you in the next one. Bye.

